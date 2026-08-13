//! pyre — A Rust meta-tracing JIT Python interpreter.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::OnceLock;

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
    /// Program read from stdin: an explicit `-` argument, or no script
    /// argument at all (app_main.py `options["run_stdin"]`). `argv0` is the
    /// literal `sys.argv[0]` the launcher records — `"-"` when the dash was
    /// given, `""` when it was not. Whether this drives the prompt or executes
    /// stdin as one unit is decided at dispatch by `stdin_is_interactive`.
    Stdin {
        argv0: String,
    },
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

// The option block and the environment fold over it live in the interpreter,
// beside the `SYS_*` statics they end up in, because the wasm32 embedding
// resolves the same block without a command line of its own.
use pyre_interpreter::launch_env::LaunchFlags;

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
-P     : don't prepend a potentially unsafe path to sys.path (also PYTHONSAFEPATH)
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
fn drain_args(parser: &mut lexopt::Parser) -> Result<Vec<std::ffi::OsString>, lexopt::Error> {
    Ok(parser.raw_args()?.collect())
}

/// Emit the `preconfig_init_utf8_mode` fatal error for an invalid PYTHONUTF8 /
/// `-X utf8` value and exit; the value is validated during pre-init config.
fn fatal_utf8_config_error(detail: &str) -> ! {
    eprintln!("Fatal Python error: preconfig_init_utf8_mode: {detail}");
    eprintln!("Python runtime state: preinitializing");
    eprintln!();
    std::process::exit(1);
}

/// Fold the process environment over the parsed options. The launcher owns a
/// real environment, so `launch_env` reads `std::env` directly here; an invalid
/// PYTHONUTF8 is fatal during pre-init config, as it is for `-X utf8`.
fn finalize_flags(flags: LaunchFlags) -> LaunchFlags {
    match pyre_interpreter::launch_env::finalize(flags) {
        Ok(flags) => flags,
        Err(e) => fatal_utf8_config_error(e.0),
    }
}

fn parse_args(
    binary_name: &str,
) -> Result<(RunMode, LaunchFlags, Vec<std::ffi::OsString>), lexopt::Error> {
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
                // The value reaches `sys._xoptions` in the host's own
                // spelling, so it is kept as an `OsString`; only the options
                // pyre acts on are matched, and every one of those is ASCII,
                // so a value with no UTF-8 form simply cannot be one of them.
                let option: std::ffi::OsString = parser.value()?;
                match option.to_str() {
                    Some("dev") => flags.dev_mode = true,
                    Some("utf8") | Some("utf8=1") => flags.utf8_mode = Some(1),
                    Some("utf8=0") => flags.utf8_mode = Some(0),
                    Some(value) if value.starts_with("utf8=") => {
                        fatal_utf8_config_error("invalid -X utf8 option value")
                    }
                    _ => {}
                }
                flags.xoptions.push(option);
            }
            // Accepted for CPython/PyPy command-line compatibility.  Warning
            // filter installation is handled by the warnings module; keeping
            // the option/value in launcher parsing lets stdlib subprocesses
            // reach their command or module.
            Short('W') => {
                flags.warnoptions.push(parser.value()?);
            }
            // `-O` / `-OO` raise the optimization level; each occurrence counts
            // (app_main.py `optimize`). PYTHONOPTIMIZE folds in during finalize.
            Short('O') => flags.optimize = flags.optimize.saturating_add(1),
            // PyPy app_main.py `simple_option`: repeated `-b` increments the
            // bytes-warning level (`-bb` therefore records 2).
            Short('b') => flags.bytes_warning = flags.bytes_warning.saturating_add(1),
            // `-B` disables writing bytecode caches (PYTHONDONTWRITEBYTECODE).
            Short('B') => flags.dont_write_bytecode = true,
            // PyPy app_main.py `unbuffered`: writing streams use raw FileIO
            // underneath TextIOWrapper and set write_through.
            Short('u') => flags.unbuffered = true,
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
                    // A leading `-` selects stdin and stays `sys.argv[0]`; the
                    // words after it are the program's arguments, exactly as
                    // for a script path.
                    let rest = drain_args(&mut parser)?;
                    return Ok((
                        RunMode::Stdin { argv0: script },
                        finalize_flags(flags),
                        rest,
                    ));
                }
                let rest = drain_args(&mut parser)?;
                return Ok((RunMode::Script(script), finalize_flags(flags), rest));
            }
            _ => return Err(arg.unexpected()),
        }
    }
    // No script argument at all reads stdin too, with an empty `sys.argv[0]`.
    Ok((
        RunMode::Stdin {
            argv0: String::new(),
        },
        finalize_flags(flags),
        vec![],
    ))
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
                // The controller's arguments are its own, not a Python
                // program's: they are rendered into the child's command line
                // as text, so one with no UTF-8 form is reported here rather
                // than carried the way `sys.argv` carries it.
                let args = drain_args(parser)?
                    .into_iter()
                    .map(|arg| arg.into_string().map_err(lexopt::Error::NonUnicodeValue))
                    .collect::<Result<Vec<String>, _>>()?;
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

/// `interactive or sys.stdin.isatty()` — a stdin run drives the prompt when fd
/// 0 is a terminal or `-i` was given, and executes stdin as a script
/// otherwise.
fn stdin_is_interactive(inspect: bool) -> bool {
    if inspect {
        return true;
    }
    #[cfg(feature = "sandbox")]
    {
        // Under sandbox the query goes through the seam, so a sandboxed child
        // observes its virtual console rather than the trusted parent's
        // terminal.
        pyre_interpreter::host_seam::ops::isatty(0).unwrap_or(false)
    }
    #[cfg(not(feature = "sandbox"))]
    {
        std::io::IsTerminal::is_terminal(&std::io::stdin())
    }
}

/// `sys.stdin.read()` — the whole of stdin as the program source.
fn read_stdin_source() -> std::io::Result<String> {
    // Under sandbox the read goes through the seam, so a sandboxed child
    // observes virtual stdin rather than the trusted parent's stdin.
    #[cfg(feature = "sandbox")]
    let data = {
        use pyre_interpreter::host_seam::{SeamError, ops};

        let mut data = Vec::new();
        loop {
            match ops::read(0, 65536) {
                Ok(chunk) if chunk.is_empty() => break,
                Ok(chunk) => data.extend_from_slice(&chunk),
                Err(SeamError::Os(errno)) => return Err(std::io::Error::from_raw_os_error(errno)),
                Err(_) => return Err(std::io::Error::other("stdin read failed")),
            }
        }
        data
    };
    #[cfg(not(feature = "sandbox"))]
    let data = {
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin(), &mut data)?;
        data
    };
    String::from_utf8(data)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "stdin not utf-8"))
}

/// Native stack for the interpreter thread spawned by [`main_entry`].
///
/// Well above the byte budget `sys.setrecursionlimit` can buy
/// (`stack_check::DEFAULT_RUNTIME_THREAD_STACK_SIZE`): `stack_check` gates
/// Python call levels, and the native recursion it does not gate — nested
/// container `repr`, the parser, GC tracing — draws on the rest.
const INTERPRETER_THREAD_STACK_SIZE: usize = 256 * 1024 * 1024;

pub fn main_entry(binary_name: &'static str) {
    configure_root_only_jit_stats();
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
    {
        real_main(binary_name);
        post_run_diagnostics();
    }

    // Block async signals on this (the process's original) thread so the
    // kernel delivers process-directed signals to the interpreter thread
    // spawned below, where they can interrupt blocking syscalls.  The
    // interpreter thread inherits this mask and unblocks them at the top of
    // `real_main`.
    #[cfg(not(feature = "sandbox"))]
    {
        pyre_interpreter::module::signal::signalstate::block_async_signals_on_origin_thread();
        std::thread::Builder::new()
            .stack_size(INTERPRETER_THREAD_STACK_SIZE)
            .spawn(|| {
                // Same first statement `_thread`'s worker runs
                // (`module/thread/mod.rs`): announce the stack this thread is
                // budgeted against, and capture the base here at the outermost
                // interpreter entry. Left out, `effective_stack_length` falls
                // back to `getrlimit(RLIMIT_STACK)`, which describes the
                // process's original thread and not this one, and the byte
                // guard — not the recursion limit — ends up deciding how deep
                // Python can recurse.
                //
                // The announced figure is the shared one, not
                // INTERPRETER_THREAD_STACK_SIZE: the budget it sizes is stored
                // in a process-global word every thread's inline probe reads.
                pyre_interpreter::stack_check::configure_current_thread_stack_size(
                    pyre_interpreter::stack_check::DEFAULT_RUNTIME_THREAD_STACK_SIZE,
                );
                real_main(binary_name);
                post_run_diagnostics();
            })
            .expect("spawn main thread")
            .join()
            .unwrap();
    }
}

/// Keep test-runner diagnostics on the pyre process the runner launched.
///
/// `test_asyncio` deliberately starts `sys.executable` children and asserts
/// their stdout/stderr protocol byte-for-byte. The CPython-suite runner still
/// needs `MAJIT_STATS` from its direct child to detect a silently recovered JIT
/// compile panic, but passing that variable through to grandchildren changes
/// the Python program's observable subprocess output. The runner opts into
/// this process-tree policy with `PYRE_MAJIT_STATS_ROOT_ONLY`; the inherited
/// marker distinguishes its direct child from later pyre descendants.
static PRINT_ROOT_ONLY_JIT_STATS: OnceLock<bool> = OnceLock::new();

fn configure_root_only_jit_stats() {
    let root_only = std::env::var_os("PYRE_MAJIT_STATS_ROOT_ONLY").is_some();
    let descendant = std::env::var_os("PYRE_MAJIT_STATS_ANCESTOR").is_some();
    let print_here = !root_only || !descendant;
    PRINT_ROOT_ONLY_JIT_STATS
        .set(print_here)
        .expect("JIT stats process policy configured twice");
    if root_only && !descendant {
        // main_entry runs before pyre starts its interpreter thread, so no
        // other thread can concurrently read or mutate the process environment.
        unsafe { std::env::set_var("PYRE_MAJIT_STATS_ANCESTOR", "1") };
    }
}

/// Diagnostics that have to read the descr universe *after* it has finished
/// growing, so they run once `real_main` returns rather than inside it. Each is
/// gated on its own environment variable internally.
///
/// The counts these name are already on the `[jit-stats]` line, which
/// `maybe_print_jit_stats` emits from `real_main`'s exit paths and so survives
/// a `process::exit` this hook does not reach. These print *which* members,
/// never whether the gate holds.
fn post_run_diagnostics() {
    if std::env::var_os("PYRE_FIELD_IDENTITY_CENSUS").is_some() {
        pyre_jit::field_descr_identity_census_now();
    }
    pyre_jit::descr_spelling_gate_recheck_now();
}

fn real_main(binary_name: &str) {
    // Ahead of everything else: the hash secret is fixed by the first digest
    // taken in the process, so `PYTHONHASHSEED` has to be resolved before any
    // Python object exists. `-E` does not reach it — the seed is settled before
    // command-line flags are, so an ignored environment still supplies it.
    if let Err(detail) = pyre_interpreter::builtins::init_hash_secret_from_env() {
        eprintln!("{detail}");
        std::process::exit(1);
    }
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
    // pypy/interpreter/app_main.py `entry_point`: preserve the executable and
    // every original launcher argument before command-line parsing rewrites
    // them into the run mode and `sys.argv`.
    importing::set_sys_orig_argv(std::env::args_os().collect());
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
        ..
    } = flags;

    // The `interact` controller does not run the embedded interpreter; it only
    // spawns and relays for an untrusted child, so it skips the interpreter
    // bootstrap (recursion budget, JIT hooks) entirely.
    let is_interact = matches!(mode, RunMode::Interact { .. });

    // The interactive REPL drives raw stdin/stdout directly, which under sandbox
    // are the controller's marshalling pipes — running it would corrupt the
    // protocol and bypass fd-0 mediation. Reject the prompt / `-i` in sandbox
    // builds; a piped stdin program reads through the seam and is fine.
    #[cfg(feature = "sandbox")]
    if !is_interact
        && (inspect || (matches!(mode, RunMode::Stdin { .. }) && stdin_is_interactive(false)))
    {
        eprintln!("{binary_name}: interactive mode is unavailable in the sandbox");
        std::process::exit(2);
    }

    if !is_interact {
        // Eagerly install pyre-jit's hooks into pyre-interpreter so that
        // sys.settrace / set_jit_param routing is live from the very first
        // user statement, not only after the first JIT-traced bytecode.
        pyre_jit::eval::init_jit_hooks();
    }

    // Record `-S` before the first `import sys` so `sys.flags.no_site`
    // reflects whether site initialization was skipped.
    importing::set_no_site(no_site);
    importing::set_runtime_flags(&flags);

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
            importing::init_sys_path(&cwd, std::ffi::OsStr::new(""));
            let mut argv = vec![std::ffi::OsString::from("-c")];
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
            importing::init_sys_path(&cwd, cwd.as_os_str());
            let mut argv = vec![std::ffi::OsString::from(&module)];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            run_module(&module, no_site);
            if inspect {
                repl::run_repl(true, no_site);
            }
        }
        RunMode::Script(path) => {
            let script_dir = script_startup_dir(&path);
            importing::init_sys_path(&script_dir, script_dir.as_os_str());
            // sys.argv[0] is the script path; remaining values go to argv[1:].
            let mut argv = vec![std::ffi::OsString::from(&path)];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            let compile_path = script_compile_path(&path);
            run_script_path(&path, &compile_path, no_site, binary_name);
            if inspect {
                repl::run_repl(true, no_site);
            }
        }
        RunMode::Stdin { argv0 } => {
            // The prompt and piped stdin both take "" for `sys.path[0]`
            // (`_PyPathConfig_ComputeSysPath0` on an argv[0] of "" / "-"); the
            // cwd stays the shadowing-check anchor.
            let cwd = sys_path_cwd();
            importing::init_sys_path(&cwd, std::ffi::OsStr::new(""));
            // `sys.argv` is `['']` with no script argument and `['-', …]` for
            // an explicit dash.
            let mut argv = vec![std::ffi::OsString::from(argv0)];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            if stdin_is_interactive(inspect) {
                // A tty (or `-i`) prints the banner unless `-q` and drops into
                // the prompt.
                repl::run_repl(quiet, no_site);
            } else {
                // Otherwise stdin is read to EOF and executed as a single
                // `exec` unit named `<stdin>`.
                let source = match read_stdin_source() {
                    Ok(source) => source,
                    Err(e) => {
                        eprintln!("{binary_name}: cannot read stdin: {e}");
                        std::process::exit(1);
                    }
                };
                run_source(&source, Mode::Exec, "<stdin>", no_site);
            }
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
// Keep the command-line options explicit at the sandbox-controller boundary;
// grouping them would only duplicate the CLI parser's argument structure.
#[allow(clippy::too_many_arguments)]
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
    if !PRINT_ROOT_ONLY_JIT_STATS.get().copied().unwrap_or(true) {
        return;
    }
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
    maybe_print_mc_diag();
    maybe_print_gc_diag();
    if std::env::var_os("MAJIT_STATS").is_none() {
        return;
    }
    // Gate tallies for the trace-entry and guard→bridge paths. These were
    // reachable only through the wasm runner's `pyre_jit_mc_diag` export, so a
    // native run had no way to see which gate declined a trace.
    //
    // `all_descrs` rides along because it is the one table the optimizer grows
    // per trace while `descr.py:47` asserts it stays under 2**15 — upstream
    // fills it once at translation time, so only pyre can run into that bound.
    //
    // Printed BEFORE the structural summary, which is where it used to have to
    // be: `check.py` `_jit_stats_snapshot` once kept only the LAST
    // `[jit-stats]` line, so a diag line emitted after the counters displaced
    // them and the loops_aborted / internal_compile_panics floor silently
    // stopped firing. That snapshot now merges every line, so the order is no
    // longer load-bearing — but the reading stays grouped diag-then-summary.
    // The wasm runner orders its diag lines the same way.
    let all_descrs_len = pyre_jit::eval::driver_pair()
        .0
        .meta_interp()
        .staticdata
        .all_descrs()
        .lock()
        .map(|d| d.len())
        .unwrap_or(0);
    eprintln!(
        "[jit-stats] mc_diag {} all_descrs={all_descrs_len}",
        majit_metainterp::mc_diag_summary()
    );
    eprintln!(
        "[jit-stats] fbw_escape_split portal_only={} published_callee_only={} portal_and_published_callee={}",
        pyre_jit::fbw_diag_counter(6),
        pyre_jit::fbw_diag_counter(7),
        pyre_jit::fbw_diag_counter(8),
    );
    eprintln!(
        "[jit-stats] fbw_force_attrib by_portal={} by_callee_only={}",
        pyre_jit::fbw_diag_counter(9),
        pyre_jit::fbw_diag_counter(10),
    );
    // Walks that ended uncommitted after a residual had already run an effect
    // no journal undoes — it wrote live heap or entered a Python frame — so the
    // replay the caller falls back to applies that effect twice.
    // `fbw_diag::ROLLED_BACK_WITH_EFFECTS`, which the wasm runner also exports;
    // a nonzero value names a walk-abort variant that reaches
    // `run_perfn_walk`'s epilogue without an adopt leg.
    //
    // `STORE_JOURNAL_ROLLBACK_FAILED` is the hole that subtraction would
    // otherwise open: a journaled store the rollback could not restore is
    // subtracted out of the count above, so it needs one of its own.
    //
    // The two adoption tallies are the same population read from the other end:
    // the walks that DID hand the interpreter a resumable image. Nothing else
    // counts them, so a change that silently sends a shape back to the legacy
    // replay moves no gated counter without these.
    eprintln!(
        "[jit-stats] fbw_rolled_back_with_effects={} \
         fbw_store_journal_rollback_failed={} \
         fbw_blackhole_adopted_single_frame={} \
         fbw_blackhole_adopted_multi_frame={}",
        pyre_jit::fbw_diag_counter(1),
        pyre_jit::fbw_diag_counter(11),
        pyre_jit::fbw_diag_counter(12),
        pyre_jit::fbw_diag_counter(13),
    );
    let stats = pyre_jit::eval::driver_pair().0.get_stats();
    eprintln!(
        "[jit-stats] loops_compiled={} bridges_compiled={} retraces_compiled={} loops_aborted={} \
         guard_failures={} internal_compile_panics={}",
        stats.loops_compiled,
        stats.bridges_compiled,
        stats.retraces_compiled,
        stats.loops_aborted,
        stats.guard_failures,
        stats.internal_compile_panics,
    );
    // How those `guard_failures` are distributed: a handful of guards eating
    // most of them means the deopt keeps returning to the runtime for a guard
    // that should already have a bridge, while a wide flat spread means many
    // guards each below `trace_eagerness`. Off unless `MAJIT_GUARD_CENSUS`.
    if std::env::var_os("MAJIT_GUARD_CENSUS").is_some() {
        eprintln!("[jit-stats] {}", majit_metainterp::guard_census_summary(12));
    }
    // What the minor collections those deopts drive actually reclaim. Off
    // unless `MAJIT_GC_DRAIN_CENSUS`.
    if majit_gc::drain_census_enabled() {
        eprintln!("[jit-stats] {}", majit_gc::drain_census_summary());
    }
    // The descr-universe invariants. `descr_set_stale_absent` re-asks the
    // `AbsentContainer` question against the universe as it stands now, so it
    // has to be read here, after the run published everything it is going to.
    eprintln!("[jit-stats] {}", pyre_jit::descr_set_jit_stats());
    // Whether those slots were filled CORRECTLY, which the line above
    // cannot say. See `field_position_jit_stats`.
    eprintln!("[jit-stats] {}", pyre_jit::field_position_jit_stats());
}

/// Names for the `MC_DIAG` tallies, in index order — the array
/// `majit_metainterp` declares beside the counters themselves, so a slot cannot
/// be added there and go unnamed here. The wasm runner prints the same names for
/// the `pyre_jit_mc_diag` export, making a native run and a wasm run of the same
/// program diffable field by field.
use majit_metainterp::MC_DIAG_LABELS as MC_DIAG_FIELDS;

/// Print the guard-failure → bridge-trace gate tallies. Until this existed the
/// counters were bumped on every backend but only readable through the wasm
/// guest export, so the question "did this guard failure ever reach a bridge,
/// or was it short-circuited by the decline blacklist" could not be asked of a
/// native run at all.
///
/// Deliberately NOT under the `[jit-stats]` prefix: `check.py`'s
/// `_jit_stats_snapshot` keeps the LAST such line, so a second one would
/// silently replace the committed per-bench baseline. Its own env gate keeps it
/// off the default `check.py` run, which sets `MAJIT_STATS` for every bench.
/// Print the GC's deterministic pressure counters.
///
/// The perf question this answers is the one a loaded machine's clock cannot:
/// "did that change actually allocate less?".  A minor collection happens once
/// per nursery-full, so `minor` is a direct proxy for cumulative allocation and
/// is identical across runs of the same program — unlike wall or even user CPU
/// time, which on this tree can swing by more than any single change is worth.
///
/// Same prefix discipline as [`maybe_print_mc_diag`]: not `[jit-stats]`, and
/// behind its own env gate, so the committed per-bench baselines never see it.
fn maybe_print_gc_diag() {
    if std::env::var_os("PYRE_GC_DIAG").is_none() {
        return;
    }
    let (minor, major, heap_bytes, nursery_bytes) = pyre_jit::gc_diag_counters();
    eprintln!(
        "[jit-gc-diag] minor={minor} major={major} heap_bytes={heap_bytes} \
         nursery_bytes={nursery_bytes}"
    );
}

fn maybe_print_mc_diag() {
    if std::env::var_os("PYRE_MC_DIAG").is_none() {
        return;
    }
    let mut line = String::from("[jit-mc-diag]");
    for (i, name) in MC_DIAG_FIELDS.iter().enumerate() {
        line.push_str(&format!(" {name}={}", majit_metainterp::mc_diag(i)));
    }
    eprintln!("{line}");
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
    let bootstrapped = bootstrap_importlib_modules(canonical, ec_ptr);
    // pylifecycle.c init_sys_streams, which follows init_importlib: the
    // standard streams can reach a text codec from here on.  A failed
    // bootstrap leaves the native importer serving imports, so the codec is
    // still reachable and the streams still want it.
    let stream_codecs = pyre_interpreter::module::sys::vm::init_stream_codecs();
    // A bootstrap failure is the more fundamental of the two, so it wins;
    // otherwise a codec the streams could not build is reported rather than
    // leaving a stream that reports itself unreadable.
    bootstrapped.and(stream_codecs)
}

fn bootstrap_importlib_modules(
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
/// callback before module teardown, then join native non-daemon handles.
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
}

/// `baseobjspace.py:finish` — after joining non-daemon threads, import the
/// builtin atexit module and run its app-level callback stack.  Any escaping
/// error is unraisable and must not prevent the remaining shutdown phases.
fn run_atexit_callbacks(canonical: pyre_object::PyObjectRef, ec_ptr: *const PyExecutionContext) {
    let result = importing::importhook("atexit", canonical, pyre_object::PY_NULL, 0, ec_ptr)
        .and_then(|module| {
            pyre_interpreter::getattr(module, pyre_object::w_str_new("_run_exitfuncs"))
        })
        .and_then(|callback| pyre_interpreter::call::call_function_impl_result(callback, &[]));
    if let Err(mut error) = result {
        error.write_unraisable(
            pyre_object::w_none(),
            rustpython_wtf8::Wtf8::new("_run_exitfuncs"),
            pyre_object::w_none(),
        );
    }
}

fn collect_and_run_finalizers(ec_ptr: *const PyExecutionContext) {
    pyre_object::gc_hook::try_gc_collect();
    if !ec_ptr.is_null() {
        unsafe { (&mut *(ec_ptr as *mut PyExecutionContext))._run_finalizers_now() };
    }
}

/// Whether releasing this binding can remove anything from the reachable set,
/// and so whether the collection that follows it could find garbage an earlier
/// one did not. Answering `false` skips a full mark-and-sweep of the whole
/// heap, which is what the release loop below otherwise costs per name.
///
/// Two values answer `false` outright:
///
/// * An exact `int`, `float`, `bool`, `str`, `bytes` or `None`. It holds no
///   reference to another object, so nothing but the value itself can lose its
///   last referrer, and no builtin scalar type defines `__del__`, so nothing
///   observes it going. `is_exact_type` is what makes this safe rather than
///   `is_str`/`is_int`, which key off the layout `ob_type` a subclass keeps: a
///   `class MyStr(str)` instance retags `w_class` and is rejected here, so its
///   `__del__` and its own attributes stay on the collecting path.
/// * A module still registered in `sys.modules` under its own `__name__`. It
///   stays reachable from there no matter what `__main__` does, so the release
///   removes no object at all from the reachable set. This is every `import`
///   name. Reading the live `sys.modules` rather than assuming is what keeps a
///   program that replaced or deleted the entry on the collecting path.
///
/// The point is not that these values are cheap to collect — it is that the
/// collection cannot reach a different answer, so every `__del__` still runs at
/// exactly the same place in the loop.
fn release_frees_nothing(value: pyre_object::PyObjectRef) -> bool {
    if value.is_null() {
        return true;
    }
    unsafe {
        if pyre_object::is_none(value)
            || pyre_object::is_exact_type(value, &pyre_object::INT_TYPE)
            || pyre_object::is_exact_type(value, &pyre_object::BOOL_TYPE)
            || pyre_object::is_exact_type(value, &pyre_object::FLOAT_TYPE)
            || pyre_object::is_exact_type(value, &pyre_object::STR_TYPE)
            || pyre_object::is_exact_type(value, &pyre_object::BYTES_TYPE)
        {
            return true;
        }
        if pyre_object::is_module(value) {
            // `Module.w_name` is the interpreter's own field, not the
            // program-writable `__name__` attribute. `module.__new__` leaves it
            // null until `__init__` seeds it, and a valid Python name may carry
            // a lone surrogate that cannot key pyre's native UTF-8 registry.
            // Either shape proves nothing about reachability and therefore
            // takes the collecting path.
            let w_name = pyre_object::w_module_get_name(value);
            if w_name.is_null() || !pyre_object::is_str(w_name) {
                return false;
            }
            let Ok(name) = pyre_object::w_str_get_wtf8(w_name).as_str() else {
                return false;
            };
            return importing::get_sys_module(name).is_some_and(|m| m == value);
        }
    }
    false
}

fn shutdown_module_private_name(name: &rustpython_wtf8::Wtf8) -> bool {
    let bytes = name.as_bytes();
    bytes.first() == Some(&b'_') && bytes.get(1) != Some(&b'_')
}

fn clear_shutdown_module_name(dict: pyre_object::PyObjectRef, name: &rustpython_wtf8::Wtf8) {
    let should_clear = unsafe { pyre_object::w_dict_getitem_wtf8(dict, name) }
        .is_some_and(|value| unsafe { !pyre_object::is_none(value) });
    if should_clear {
        unsafe {
            pyre_object::w_dict_setitem_wtf8(dict, name, pyre_object::w_none());
        }
    }
}

/// `_PyModule_ClearDict`: clear string-keyed module globals in two name passes.
fn clear_shutdown_module_dict(dict: pyre_object::PyObjectRef) {
    if dict.is_null() {
        return;
    }
    let keys: Vec<rustpython_wtf8::Wtf8Buf> = unsafe { pyre_object::w_dict_str_entries_wtf8(dict) }
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    for name in &keys {
        if shutdown_module_private_name(name) {
            clear_shutdown_module_name(dict, name);
        }
    }
    for name in &keys {
        if name.as_bytes() != b"__builtins__" {
            clear_shutdown_module_name(dict, name);
        }
    }
}

/// `finalize_modules`: clear detached modules newest-first while their peers
/// remain available to finalizers that run between module dictionaries.
fn clear_shutdown_modules(
    modules: Vec<(rustpython_wtf8::Wtf8Buf, pyre_object::PyObjectRef)>,
    ec_ptr: *const PyExecutionContext,
) {
    let _roots = pyre_object::gc_roots::push_roots();
    let roots_start = pyre_object::gc_roots::shadow_stack_len();
    let mut names = Vec::with_capacity(modules.len());
    let mut sys_module_slot = None;
    let mut builtins_module_slot = None;
    for (name, module) in modules {
        let index = names.len();
        let bytes = name.as_bytes();
        if bytes == b"sys" {
            sys_module_slot = Some(index);
        } else if bytes == b"builtins" {
            builtins_module_slot = Some(index);
        }
        names.push(name);
        pyre_object::gc_roots::pin_root(module);
    }
    collect_and_run_finalizers(ec_ptr);
    for index in (0..names.len()).rev() {
        let module = pyre_object::gc_roots::shadow_stack_get(roots_start + index);
        let is_core_module = sys_module_slot.is_some_and(|slot| {
            module == pyre_object::gc_roots::shadow_stack_get(roots_start + slot)
        }) || builtins_module_slot.is_some_and(|slot| {
            module == pyre_object::gc_roots::shadow_stack_get(roots_start + slot)
        });
        if is_core_module {
            continue;
        }
        if module.is_null() || !unsafe { pyre_object::is_module(module) } {
            continue;
        }
        let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        clear_shutdown_module_dict(dict);
    }
    // One collection for the whole walk, not one per module. `finalize_modules`
    // clears the module dictionaries and lets refcounting release what they
    // held; a sweep per module buys no ordering here, because a finalizer that
    // reads a global reaches its own already-cleared namespace either way, and
    // it costs a full mark-and-sweep for each of the ~100 modules a bare
    // `import unittest` loads.
    collect_and_run_finalizers(ec_ptr);
}

/// PyPy `ObjSpace.finish()` / module teardown ordering: join non-daemon
/// threads, collect already-unreachable cycles, then release `__main__`
/// globals from newest to oldest while the older globals their `__del__`
/// methods may reference are still present.
///
/// Newest-to-oldest is what keeps those references working, and it is not
/// interchangeable with the insertion order `_PyModule_ClearDict` uses. A name
/// is bound before every name that could be finalized while reading it — most
/// of all `import sys`, which is usually the very first — so releasing in
/// insertion order strands `sys` at `None` and every finalizer that writes to
/// `sys.stderr` dies with an `AttributeError` instead of running. Refcounting
/// makes the question moot upstream: a finalizer there runs from the decref of
/// the name being released, while every other slot still holds its original
/// value, which is not reproducible without leaving dangling pointers in the
/// dict.
///
/// The value is rebound to `None` rather than deleted, as `_PyModule_ClearDict`
/// does: a `__del__` that reads an already-released name then sees `None`, the
/// way it would upstream, instead of raising `NameError` at a name the program
/// can see is still defined.
///
/// `__main__` is not the whole reachable set: an object stored in another
/// module's namespace stays alive through that module's dict. The
/// `finalize_modules` phase that follows detaches `sys.modules` and clears the
/// remaining module dictionaries newest-first.
fn finalize_runtime(canonical: pyre_object::PyObjectRef, ec_ptr: *const PyExecutionContext) {
    run_threading_shutdown();
    run_atexit_callbacks(canonical, ec_ptr);
    // PyPy sets `sys.finalizing` only after atexit callbacks.  Those callbacks
    // may still start threads; reject new starts only when module/finalizer
    // teardown is actually about to begin.
    pyre_interpreter::module::thread::set_finalizing();
    // baseobjspace.py:498-501 `finish()` runs every started module's shutdown
    // hook; `_io`'s (moduledef.py:37-40) flushes the streams that are still
    // alive.  The per-global teardown below reaches only the ones `__main__`
    // itself holds, so without this a stream owned by any other module loses
    // its buffered writes.
    pyre_interpreter::module::_io::flush_all_streams();
    collect_and_run_finalizers(ec_ptr);
    let mut entries = unsafe { pyre_object::w_dict_str_entries(canonical) };
    entries.reverse();
    for (name, _) in entries {
        if name == "__builtins__" {
            continue;
        }
        // Re-read rather than trusting the snapshot: a `__del__` already run by
        // this loop may have rebound the name, and the decision below is only
        // sound about the value actually being released.
        let value = unsafe { pyre_object::w_dict_getitem_str(canonical, &name) };
        let frees_nothing = value.is_none_or(release_frees_nothing);
        unsafe {
            pyre_object::w_dict_setitem_str(canonical, &name, pyre_object::w_none());
        }
        if !frees_nothing {
            collect_and_run_finalizers(ec_ptr);
        }
    }
    // The loop ends on a collection whenever it releases anything collectable;
    // this is the one for the case where the last releases were all skipped, so
    // teardown still finishes with the heap swept and the queue drained.
    collect_and_run_finalizers(ec_ptr);
    let shutdown_modules = pyre_interpreter::importing::release_sys_modules_for_shutdown();
    clear_shutdown_modules(shutdown_modules, ec_ptr);
}

/// Resolve a pending `SystemExit`'s status, then finalize and exit with it.
///
/// The order matters: `app_main.py:114-129 handle_sys_exit` runs at
/// application level, i.e. *before* `targetpypystandalone.py:88
/// finally: space.finish()`. Resolving `e.code` is an ordinary attribute
/// lookup that walks the exception's type and that type's dict, and
/// `finalize_runtime` collects — pinning the three raw `PyObjectRef` fields
/// of the Rust `PyError` (which, unlike PyPy's GC-visible `OperationError`,
/// the collector cannot see) does not keep that type's dict alive. Reading
/// the code after the collection spun forever inside the type dict's
/// `IndexMap` probe for a user-defined `SystemExit` subclass.
fn finalize_system_exit(
    error: pyre_interpreter::PyError,
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) -> ! {
    let code = pyre_interpreter::system_exit_code(&error);
    finalize_runtime(canonical, ec_ptr);
    maybe_print_jit_stats();
    std::process::exit(code);
}

/// Die by SIGINT for an uncaught KeyboardInterrupt after shutdown has run
/// (`app_main.py:1133-1153`).
fn terminate_by_sigint() -> ! {
    // A Win32 process has no SIGINT to die of: restoring `SIG_DFL` and calling
    // `raise(SIGINT)` runs the CRT's default action, which returns and ends the
    // process with status 3 rather than the interrupt a shell reads. The status
    // that reads as one is `STATUS_CONTROL_C_EXIT`, so it is exited with
    // directly. Both callers have already finalized and flushed.
    #[cfg(windows)]
    {
        const STATUS_CONTROL_C_EXIT: u32 = 0xC000_013A;
        std::process::exit(STATUS_CONTROL_C_EXIT as i32);
    }
    #[cfg(not(windows))]
    {
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            if libc::kill(libc::getpid(), libc::SIGINT) != 0 {
                std::process::exit(1);
            }
        }
        std::process::abort();
    }
}

/// Whether the raised exception is an instance of the named builtin exception
/// class or of a subclass of it.
fn raised_is_instance_of(error: &pyre_interpreter::PyError, class_name: &str) -> bool {
    let exc = error.exc_object;
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return false;
    }
    let Some(raised_type) = pyre_interpreter::typedef::r#type(exc) else {
        return false;
    };
    pyre_interpreter::builtins::lookup_exc_class(class_name).is_some_and(|target| unsafe {
        pyre_interpreter::baseobjspace::exception_issubclass_w(raised_type.as_ptr(), target)
    })
}

fn is_keyboard_interrupt(error: &pyre_interpreter::PyError) -> bool {
    raised_is_instance_of(error, "KeyboardInterrupt")
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
        runpy_run_module_as_main(canonical, ec_ptr, module, true)?;
        Ok(())
    })();

    if let Err(e) = result {
        if e.kind == PyErrorKind::SystemExit {
            finalize_system_exit(e, canonical, ec_ptr);
        }
        let is_keyboard_interrupt = is_keyboard_interrupt(&e);
        // targetpypystandalone.py:88 `finally: space.finish()` — finalize on
        // every exit path, not only on SystemExit.  Print first: the raw
        // `PyObjectRef` fields of `e` are not GC-visible and `finalize_runtime`
        // collects.
        pyre_interpreter::eprint_exception(&e, true);
        finalize_runtime(canonical, ec_ptr);
        maybe_print_jit_stats();
        if is_keyboard_interrupt {
            terminate_by_sigint();
        }
        std::process::exit(1);
    }
    finalize_runtime(canonical, ec_ptr);
    maybe_print_jit_stats();
}

/// `os.path.abspath` for the run filename, or `None` for the `-c "<string>"`
/// command path, which has no script.
///
/// A relative path resolves against `sys_path_cwd()` — the seam-provided
/// virtual cwd under sandbox — before normalizing, so `std::path::absolute`
/// never consults (and leaks) the trusted process cwd. Off sandbox this is
/// identical to `absolute(filename)`.
fn absolute_script_path(filename: &str) -> Option<String> {
    if filename == "<string>" {
        return None;
    }
    if filename == "<stdin>" {
        // A stdin run records the literal `<stdin>`: there is no path to
        // absolutize.
        return Some(filename.to_string());
    }
    let path = Path::new(filename);
    let to_absolutize = if path.is_absolute() {
        path.to_path_buf()
    } else {
        sys_path_cwd().join(path)
    };
    Some(match std::path::absolute(&to_absolutize) {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(_) => filename.to_string(),
    })
}

fn script_startup_dir(path: &str) -> PathBuf {
    // Initialize sys.path with the script's directory. Under sandbox,
    // `canonicalize()` issues raw host-FS syscalls (realpath) past the
    // seccomp lockdown and resolves against the real filesystem, not the
    // controller VFS; use the virtual path as given instead.
    #[cfg(feature = "sandbox")]
    {
        Path::new(path)
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
    }
    #[cfg(not(feature = "sandbox"))]
    {
        // `parent()` of a bare `x.py` is the empty path;
        // `_PyPathConfig_ComputeSysPath0` absolutizes the script's
        // directory, so resolve the empty case against the cwd rather
        // than leaving a relative `.` on `sys.path`.
        let parent = Path::new(path).parent().unwrap_or(Path::new("."));
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        parent.canonicalize().unwrap_or_else(|_| sys_path_cwd())
    }
}

fn script_compile_path(path: &str) -> String {
    // A script is compiled with the same absolute path exposed as
    // `__main__.__file__`; warnings use `code.co_filename`, so the two must
    // not diverge when argv[0] was relative.
    #[cfg(not(feature = "sandbox"))]
    {
        std::path::absolute(Path::new(path))
            .unwrap_or_else(|_| Path::new(path).to_path_buf())
            .to_string_lossy()
            .into_owned()
    }
    #[cfg(feature = "sandbox")]
    {
        path.to_string()
    }
}

/// `app_main.py:1057` treats only `ImportError` as "this hook does not handle
/// the path" and lets anything else propagate.  `zipimport.ZipImportError` is
/// a subclass, so a hook that rejects an ordinary file lands here too.  The
/// kind test comes first because a natively-raised error carries its class in
/// `kind` and may have no exception object yet.
fn is_import_error(error: &pyre_interpreter::PyError) -> bool {
    matches!(
        error.kind,
        PyErrorKind::ImportError | PyErrorKind::ModuleNotFoundError
    ) || raised_is_instance_of(error, "ImportError")
}

fn path_hook_accepts(
    filename: &str,
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) -> Result<bool, pyre_interpreter::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let sys = importing::importhook("sys", canonical, pyre_object::PY_NULL, 0, ec_ptr)?;
    let path_hooks = pyre_interpreter::baseobjspace::getattr_str(sys, "path_hooks")?;
    // `for hook in sys.path_hooks` is a plain iteration, so a `sitecustomize`
    // that replaces the list with a tuple or any other iterable still gets its
    // hooks called.
    let hooks = pyre_interpreter::baseobjspace::unpackiterable(path_hooks, -1)?;

    let _roots = push_roots();
    let filename_slot = shadow_stack_len();
    pin_root(pyre_object::w_str_new(filename));
    // A hook call runs arbitrary Python and can drive a collection, which moves
    // the hooks themselves. A Rust vector is not walked, so publish each hook in
    // a shadow-stack slot and read it back after the previous call.
    let hooks_base = shadow_stack_len();
    for hook in &hooks {
        pin_root(*hook);
    }
    for i in 0..hooks.len() {
        match pyre_interpreter::call::call_function_impl_result(
            shadow_stack_get(hooks_base + i),
            &[shadow_stack_get(filename_slot)],
        ) {
            // `importer = hook(filename); break` — the first hook that does not
            // raise `ImportError` ends the walk, and `importer is None` then
            // means no importer claimed the path.
            Ok(importer) => return Ok(!unsafe { pyre_object::is_none(importer) }),
            Err(error) if is_import_error(&error) => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(false)
}

fn runpy_run_module_as_main(
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
    module: &str,
    alter_argv: bool,
) -> Result<(), pyre_interpreter::PyError> {
    let runpy = importing::importhook("runpy", canonical, pyre_object::PY_NULL, 0, ec_ptr)?;
    let func = pyre_interpreter::getattr(runpy, pyre_object::w_str_new("_run_module_as_main"))?;
    let args = if alter_argv {
        vec![pyre_object::w_str_new(module)]
    } else {
        vec![
            pyre_object::w_str_new(module),
            pyre_object::w_bool_from(false),
        ]
    };
    let res = pyre_interpreter::call_function(func, &args);
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
}

fn prepare_main_module(
    execution_context: &Rc<PyExecutionContext>,
) -> (pyre_object::PyObjectRef, pyre_object::PyObjectRef) {
    let w_globals = execution_context.fresh_module_globals();
    let _root = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_globals);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            w_globals,
            "__name__",
            pyre_object::w_str_new("__main__"),
        )
    };
    let canonical = w_globals;
    let main_module = pyre_object::module::w_module_new_aliasing_dict("__main__", canonical);
    importing::set_sys_module("__main__", main_module);
    (canonical, main_module)
}

fn handle_main_error(
    e: pyre_interpreter::PyError,
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) -> ! {
    if e.kind == PyErrorKind::SystemExit {
        finalize_system_exit(e, canonical, ec_ptr);
    }
    let is_keyboard_interrupt = is_keyboard_interrupt(&e);
    // targetpypystandalone.py:88 `finally: space.finish()` — finalize
    // on every exit path, not only on SystemExit.  Print first: the raw
    // `PyObjectRef` fields of `e` are not GC-visible and
    // `finalize_runtime` collects.
    pyre_interpreter::eprint_exception(&e, true);
    finalize_runtime(canonical, ec_ptr);
    maybe_print_jit_stats();
    if is_keyboard_interrupt {
        terminate_by_sigint();
    }
    std::process::exit(1);
}

fn finish_main(canonical: pyre_object::PyObjectRef, ec_ptr: *const PyExecutionContext) {
    finalize_runtime(canonical, ec_ptr);
    maybe_print_jit_stats();
}

fn eval_source_in_main(
    source: &str,
    mode: Mode,
    filename: &str,
    main_file: Option<&str>,
    execution_context: Rc<PyExecutionContext>,
    canonical: pyre_object::PyObjectRef,
    main_module: pyre_object::PyObjectRef,
    no_site: bool,
) {
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

    let ec_ptr = Rc::as_ptr(&execution_context);
    let mut frame = match PyFrame::new_with_context_and_globals(code, execution_context, canonical)
    {
        Ok(frame) => frame,
        Err(e) => {
            pyre_interpreter::eprint_exception(&e, true);
            std::process::exit(1);
        }
    };
    // The frame is a GC object, and the only root that reaches one is the
    // `CURRENT_FRAME` / `f_backref` chain the frame walker follows — which it
    // joins when `eval_with_jit` below installs it.  Everything between here
    // and that call runs Python (`site`) and can therefore drive a collection
    // that would reclaim the frame and hand its storage to the next frame
    // allocation.  RPython keeps a local holding a GC object alive through the
    // translated shadow stack; publish it explicitly for the same span.
    let _main_frame_root = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(&*frame as *const PyFrame as pyre_object::PyObjectRef);

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

    // A script run by path gets `__file__` / `__cached__` in `__main__`;
    // the `-c "<string>"` command path does not. `__file__` is absolutized
    // while `sys.argv[0]` keeps the literal command-line path. A stdin run
    // records the literal `<stdin>`: there is no path to absolutize and no
    // file to bind a `SourceFileLoader` to.
    if let Some(file) = main_file {
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            main_module,
            "__file__",
            pyre_object::w_str_new(file),
        );
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            main_module,
            "__cached__",
            pyre_object::w_none(),
        );
    }
    let script_file = if filename == "<stdin>" {
        None
    } else {
        main_file
    };
    seed_main_loader(canonical, script_file, ec_ptr);
    import_site(no_site, canonical, ec_ptr);

    match eval_with_jit(&mut frame) {
        Ok(result) => {
            if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                println!("{}", PyDisplay(result));
            }
        }
        Err(e) => handle_main_error(e, canonical, ec_ptr),
    }
    finish_main(canonical, ec_ptr);
}

fn read_script_source(path: &str, binary_name: &str) -> String {
    // The script is read through the import machinery's source provider, so
    // under sandbox the controller VFS mediates it over the same channel
    // module imports use.  The bytes then go through the tokenizer's BOM /
    // PEP 263 decoding in every build.
    match importing::read_source_bytes(Path::new(path)) {
        Ok(bytes) => match pyre_interpreter::decode_source_bytes(
            &bytes,
            rustpython_wtf8::Wtf8::new(path),
            false,
        ) {
            Ok(source) => source,
            Err(error) => {
                pyre_interpreter::eprint_exception(&error, false);
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("{binary_name}: cannot open '{path}': {e}");
            std::process::exit(1);
        }
    }
}

fn run_script_path(path: &str, filename: &str, no_site: bool, binary_name: &str) {
    let execution_context = setup_exec_context();
    let ec_ptr = Rc::as_ptr(&execution_context);
    let (canonical, main_module) = prepare_main_module(&execution_context);

    let result = (|| -> Result<bool, pyre_interpreter::PyError> {
        // Import `sys` up front so its creation flushes the native search-path
        // seed into `sys.path` before `site` and user code read it.
        let _ = importing::importhook("sys", canonical, pyre_object::PY_NULL, 0, ec_ptr);
        // pylifecycle.c init_importlib before site: install the importlib
        // bootstrap so `sys.meta_path` / `sys.path_hooks` are populated before
        // the run target is chosen.
        if let Err(e) = init_importlib_bootstrap(canonical, ec_ptr) {
            eprintln!("pyre: importlib bootstrap failed: {}", e.message_text());
        }
        path_hook_accepts(filename, canonical, ec_ptr)
    })();

    match result {
        Ok(true) => {
            importing::restage_sys_path_0(std::ffi::OsStr::new(filename));
            import_site(no_site, canonical, ec_ptr);
            if let Err(e) = runpy_run_module_as_main(canonical, ec_ptr, "__main__", false) {
                handle_main_error(e, canonical, ec_ptr);
            }
            finish_main(canonical, ec_ptr);
        }
        Ok(false) => {
            let source = read_script_source(path, binary_name);
            let main_file = absolute_script_path(filename);
            let filename = main_file.as_deref().unwrap_or(filename);
            eval_source_in_main(
                &source,
                Mode::Exec,
                filename,
                main_file.as_deref(),
                execution_context,
                canonical,
                main_module,
                no_site,
            );
        }
        Err(e) => handle_main_error(e, canonical, ec_ptr),
    }
}

fn run_source(source: &str, mode: Mode, filename: &str, no_site: bool) {
    // `config_run_filename_abspath` absolutizes the run filename before the
    // module is compiled, so `co_filename` — and with it every `File "…"` line
    // a traceback prints — carries the absolute path, not the literal argv
    // spelling.  `sys.argv[0]` keeps the spelling and is set elsewhere.
    let main_file = absolute_script_path(filename);
    let filename = main_file.as_deref().unwrap_or(filename);
    let execution_context = setup_exec_context();
    let ec_ptr = Rc::as_ptr(&execution_context);
    let (canonical, main_module) = prepare_main_module(&execution_context);

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

    eval_source_in_main(
        source,
        mode,
        filename,
        main_file.as_deref(),
        execution_context,
        canonical,
        main_module,
        no_site,
    );
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
