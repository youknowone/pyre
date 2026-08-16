//! Native host that runs the wasm32 build of pyre under wasmtime so the full
//! interpreter+JIT path executes outside a browser.
//!
//! It is the non-JS counterpart of `majit-backend-wasm/js/jit_glue.js` and
//! implements the same host-import contract:
//!
//!   * the main module (`pyre-wasm` built with `--features wasm-host`) imports
//!     `pyre_jit.{jit_compile_wasm, jit_execute_wasm, jit_free_wasm}` and
//!     exports `memory`, `__indirect_function_table`, `pyre_alloc`,
//!     `pyre_dealloc`, and `pyre_run_python`;
//!   * each JIT-emitted trace module imports `env.memory` (shared with the
//!     main module) plus an optional `env.jit_call` trampoline, and exports a
//!     `trace` function `(i32) -> i32`;
//!   * the trampoline reads func_ptr / args from the frame call area, dispatches
//!     through the main module's indirect function table (a `fn as usize` is a
//!     table index on wasm32), and writes the result back.
//!
//! CLI: `pyre-wasm-runner <script.py>` runs the script and writes its output to
//! stdout, matching the `pyrex` backend interface `pyre/check.py` drives. The
//! wasm module is located via `$PYRE_WASM_MODULE` or `--module <path>`, else the
//! default release artifact path.

mod wasmi_host;

use std::path::{Path, PathBuf};

use wasmtime::error::Context;
use wasmtime::{
    AsContext, AsContextMut, Caller, Config, Engine, Error, Extern, Func, Instance, Linker, Memory,
    Module, Ref, Result, Store, Table, Val, ValType,
};

// Frame call-area offsets — must match `majit-backend-wasm/src/codegen.rs`.
// Shared with `wasmi_host`, which mirrors the same call-area protocol.
pub(crate) const CALL_RESULT_OFS: usize = 2000;
// The arg count also lives at offset 2016, but the trampoline derives arity
// (and the exact value types) from the resolved function's wasm signature
// instead, which is authoritative on wasm32. Kept for layout documentation.
#[allow(dead_code)]
pub(crate) const CALL_NARGS_OFS: usize = 2016;

pub(crate) const DEFAULT_MODULE: &str = "target/wasm32-unknown-unknown/release/pyre_wasm.wasm";

/// Which wasm runtime executes the module. `wasmtime` (cranelift) is fast in
/// steady state but compiles the whole ~14MB module on load; `wasmi` is a
/// pure-Rust interpreter with near-zero load cost but slower hot loops.
/// Selected by `--engine` or `$PYRE_WASM_ENGINE` (CLI wins), default wasmtime.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum WasmEngine {
    Wasmtime,
    Wasmi,
}

impl WasmEngine {
    fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "wasmtime" => Ok(WasmEngine::Wasmtime),
            "wasmi" => Ok(WasmEngine::Wasmi),
            other => Err(format!("unknown engine {other:?} (want wasmtime|wasmi)")),
        }
    }
}

/// Per-store host state shared by all import callbacks.
#[derive(Default)]
struct Host {
    /// The main module's exported linear memory, shared with every trace.
    memory: Option<Memory>,
    /// The main module's `__indirect_function_table`, used by the trampoline.
    table: Option<Table>,
    /// First `__indirect_function_table` slot that is a JIT trace. The main
    /// module's table is pre-populated with its own functions (`[0,
    /// trace_base)`); `jit_compile` only ever appends (`table.grow`), so every
    /// slot `>= trace_base` is a trace. That, plus the table itself (a freed
    /// trace's slot is reset to `Func(None)`), is the sole record of trace
    /// liveness — no id→trace map is kept, since the slot IS the id and the
    /// table is the single source of truth: `jit_execute` accepts an `id >=
    /// trace_base` whose slot is still a function, and `jit_free` clears only
    /// such slots (nulling a runtime slot would corrupt dispatch). The table
    /// also roots each trace's instance for the `Store`'s lifetime.
    trace_base: u64,
    /// Real stdlib root the wasm module's `pyre_host.*` imports read source
    /// from (`$PYRE_STDLIB`, forwarded by `pyre/check.py`). The wasm side
    /// seeds it on `sys.path`, so the host serves genuine absolute paths.
    stdlib_root: Option<String>,
    /// `PYRE_WASM_JIT_STATS` diagnostic counters: trace modules compiled /
    /// executed this run. Per-store (not a global static): the increment sites
    /// already hold the `Caller<Host>`, and the runner is single-threaded.
    jit_compile_count: u64,
    /// `PYRE_WASM_JIT_STATS` diagnostic: total wall time spent inside
    /// `Module::new` (wasmtime Cranelift-compiling guest-emitted trace
    /// modules), in nanoseconds. Isolates host-compile latency — the warmup
    /// tax — from steady-state execution.
    jit_compile_time_ns: u128,
    jit_execute_count: u64,
    /// Diagnostic: per-op residual-call host crossings (`env.jit_call`
    /// trampoline invocations). Compare against `jit_execute_count` to test
    /// whether per-op crossings or per-guard-exit crossings dominate.
    jit_call_count: u64,
    /// Wall time inside `jit_call_trampoline`, in nanoseconds.
    jit_call_time_ns: u128,
    /// Diagnostic (PYRE_WASM_EXEC_TRACE=1): histogram of (trace func_id,
    /// guard-exit fail_index) over every host round-trip, so we can see which
    /// guard keeps returning to the host instead of chaining in-module.
    exec_hist: std::collections::BTreeMap<(u32, u32), u64>,
    /// `PYRE_WASM_GUEST_PROFILE` sampling profiler; taken/restored around each
    /// epoch tick so `sample` can borrow the store it lives in.
    guest_profiler: Option<wasmtime::GuestProfiler>,
}

/// Report `PYRE_*` / `MAJIT_*` settings the guest cannot see.
///
/// The module is built for `wasm32-unknown-unknown`, whose `std::env` is
/// permanently empty: every `std::env::var_os` reached from inside the guest
/// returns `None` regardless of this process's environment. That covers every
/// JIT knob and probe that lives in the guest crates — `PYRE_NO_JIT`,
/// `PYRE_GC_INTERP`, `MAJIT_STRICT`, `MAJIT_LOG`, the `PYRE_FBW_*` /
/// `PYRE_P2_DIAG` / `PYRE_JD1*` diagnostics — so setting one here changes
/// nothing at all. Silently ignoring them is the trap: an A/B run through such
/// a knob measures the same build twice and reads as a result. Name them
/// instead, once, on stderr (`check.py` shows a run's stderr only when it
/// fails, so this is invisible to a green suite and present in every failure
/// dump and interactive run).
///
/// Where a guest-side probe has a host-side replacement, use that instead:
/// `PYRE_FBW_DEBUG_ABORT`'s decline census is `PYRE_WASM_FBW_CENSUS` here,
/// read back through the `pyre_fbw_census` export, and `MAJIT_GUARD_CENSUS`'s
/// per-guard deopt census is `PYRE_WASM_GUARD_CENSUS`, armed through
/// `pyre_jit_guard_census_enable` and read through `pyre_jit_guard_census`.
/// `PYRE_WASM_TRACE_ENTRY_CENSUS` similarly arms the emitted-module entry
/// census before tracing starts.
///
/// Exempt: the names this runner interprets host-side (`PYRE_WASM_*`,
/// `PYRE_STDLIB`, `MAJIT_STATS`) and `check.py`'s own `PYRE_CHECK_*`
/// interpreter paths. A knob that later becomes host-interpreted must be added
/// here; the prefix match needs no upkeep for new guest-side knobs.
fn warn_inert_guest_env() {
    const HOST_HANDLED: &[&str] = &["PYRE_STDLIB", "MAJIT_STATS"];
    // `to_string_lossy`, not `into_string().ok()`: a name the platform allows
    // but UTF-8 does not is still a setting the guest silently ignores, and
    // dropping it here would hide exactly the case worth reporting.
    let mut inert: Vec<String> = std::env::vars_os()
        .map(|(name, _)| name.to_string_lossy().into_owned())
        .filter(|name| name.starts_with("PYRE_") || name.starts_with("MAJIT_"))
        .filter(|name| {
            !name.starts_with("PYRE_WASM_")
                && !name.starts_with("PYRE_CHECK_")
                && !HOST_HANDLED.contains(&name.as_str())
        })
        .collect();
    if inert.is_empty() {
        return;
    }
    inert.sort();
    inert.dedup();
    eprintln!(
        "[pyre-wasm-runner] ignoring {}: the wasm guest has no environment, so \
         guest-side knobs and probes do nothing here",
        inert.join(", ")
    );
}

fn main() {
    let t0_main = std::time::Instant::now();
    warn_inert_guest_env();
    let startup_trace = std::env::var_os("PYRE_WASM_STARTUP_TRACE").is_some();
    let mut module_path: Option<PathBuf> = None;
    let mut script: Option<PathBuf> = None;
    let mut inspect = false;
    let mut engine: Option<WasmEngine> = None;

    // `args_os`, not `args`: the latter unwraps the UTF-8 conversion, so a
    // script path spelling a byte with no UTF-8 form aborts the runner before
    // it can name the file. Every flag below is ASCII by construction, so a
    // value that does not convert is never one and takes the positional arm.
    let mut argv = std::env::args_os().skip(1);
    while let Some(arg) = argv.next() {
        let flag = arg.to_str().map(str::to_owned);
        match flag.as_deref() {
            Some("--inspect") => inspect = true,
            Some("--module") => {
                module_path = Some(PathBuf::from(
                    argv.next()
                        .unwrap_or_else(|| fatal("--module needs a path")),
                ))
            }
            Some("--engine") => {
                let v = argv
                    .next()
                    .unwrap_or_else(|| fatal("--engine needs a value"));
                let v = v
                    .to_str()
                    .unwrap_or_else(|| fatal("--engine needs a value"));
                engine = Some(WasmEngine::parse(v).unwrap_or_else(|e| fatal(&e)));
            }
            Some("-h") | Some("--help") => {
                eprintln!(
                    "usage: pyre-wasm-runner [--module <pyre_wasm.wasm>] \
                     [--engine wasmtime|wasmi] [--inspect] <script.py>"
                );
                std::process::exit(2);
            }
            Some(other) if other.starts_with('-') => fatal(&format!("unknown flag {other}")),
            _ => script = Some(PathBuf::from(arg)),
        }
    }

    let module_path = module_path
        .or_else(|| std::env::var_os("PYRE_WASM_MODULE").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODULE));

    // CLI `--engine` wins over `$PYRE_WASM_ENGINE`; default wasmtime.
    let engine = engine
        .or_else(|| {
            std::env::var("PYRE_WASM_ENGINE")
                .ok()
                .map(|v| WasmEngine::parse(&v).unwrap_or_else(|e| fatal(&e)))
        })
        .unwrap_or(WasmEngine::Wasmtime);

    // The wasm runs on the calling thread's stack (sync wasmtime), so the
    // deep interpreter recursion needs a large host stack to back the
    // generous `max_wasm_stack` set in `run`. wasmi runs on its own heap
    // stacks, but the large reservation is harmless there too.
    let worker = std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || -> std::result::Result<i32, String> {
            if inspect {
                return match engine {
                    WasmEngine::Wasmtime => {
                        inspect_module(&module_path).map(|()| 0).map_err(fmt_err)
                    }
                    WasmEngine::Wasmi => wasmi_host::inspect(&module_path),
                };
            }
            let script = script.ok_or_else(|| "no script given".to_string())?;
            let source = std::fs::read_to_string(&script)
                .map_err(|e| format!("read script {}: {e}", script.display()))?;
            match engine {
                WasmEngine::Wasmtime => run(&module_path, &source, &script).map_err(fmt_err),
                WasmEngine::Wasmi => wasmi_host::run(&module_path, &source, &script),
            }
        })
        .expect("spawn worker thread");

    match worker.join() {
        Ok(Ok(code)) => {
            if startup_trace {
                eprintln!(
                    "[startup] TOTAL(main-entry..worker-joined) {:.1}ms",
                    t0_main.elapsed().as_secs_f64() * 1e3
                );
            }
            std::process::exit(code)
        }
        Ok(Err(e)) => fatal(&e),
        Err(_) => fatal("worker thread panicked"),
    }
}

/// Format a wasmtime error with its `{:?}` chain for the fatal handler.
fn fmt_err(e: Error) -> String {
    format!("{e:?}")
}

fn fatal(msg: &str) -> ! {
    eprintln!("pyre-wasm-runner: {msg}");
    std::process::exit(1);
}

fn run(module_path: &Path, source: &str, script: &Path) -> Result<i32> {
    let mut config = Config::new();
    // Allow the interpreter's deep recursion before wasmtime raises a stack
    // overflow trap; the interpreter's own recursion limit normally fires
    // first. Kept below the worker thread's stack reservation. wasmtime
    // requires `max_wasm_stack <= async_stack_size` even for sync execution,
    // so the async stack is sized just above it.
    const WASM_STACK: usize = 256 * 1024 * 1024;
    config.max_wasm_stack(WASM_STACK);
    config.async_stack_size(WASM_STACK + 1024 * 1024);
    // JIT trace modules emit `return_call_indirect` to chain a loop-closing bridge
    // back into its loop at constant stack depth (the tail-call proposal).
    config.wasm_tail_call(true);
    // Optional instruction budget: PYRE_WASM_FUEL=N traps the guest after N fuel
    // units so a diagnostic run that livelocks (e.g. a buggy loop-closing bridge)
    // still reaches the stats readout instead of hanging forever.
    let fuel_limit: Option<u64> = std::env::var("PYRE_WASM_FUEL")
        .ok()
        .and_then(|s| s.parse().ok());
    // Fuel doubles as an exact executed-instruction counter, reported as
    // `wasm_ops` in the stats line below. It is the one wasm measurement that
    // does not move with machine load, which makes it the right thing to
    // compare a codegen change against. Metering costs time on every
    // instruction, so it stays confined to the two diagnostic modes that ask
    // for it — a timed run sets neither and pays nothing.
    let jit_stats = std::env::var_os("PYRE_WASM_JIT_STATS").is_some();
    let meter_fuel = fuel_limit.is_some() || jit_stats;
    if meter_fuel {
        config.consume_fuel(true);
    }
    // Diagnostic: PYRE_WASM_GUEST_PROFILE=<out.json> writes a sampling profile
    // of the guest in the Firefox processed format (main-module symbols only;
    // JIT trace modules appear as omitted frames under `execute_token`). View
    // at https://profiler.firefox.com/. Samples land on epoch checkpoints
    // (function entries / loop back-edges), so call-dense small functions are
    // over-represented and bulk ops (memory.fill) are attributed to the next
    // checkpoint. Epoch interruption changes main-module codegen, so pair with
    // PYRE_WASM_NO_CACHE=1 to avoid clobbering the shared .cwasm cache.
    let guest_profile_out = std::env::var("PYRE_WASM_GUEST_PROFILE").ok();
    if guest_profile_out.is_some() {
        config.epoch_interruption(true);
    }
    // Diagnostic: PYRE_WASM_STARTUP_TRACE=1 prints a fixed-startup breakdown
    // (module load/deserialize, main-module instantiate, guest bootstrap+run)
    // so the warmup tax can be attributed to host module-load vs guest
    // interpreter bootstrap.
    let startup_trace = std::env::var_os("PYRE_WASM_STARTUP_TRACE").is_some();
    let mut startup_mark = std::time::Instant::now();
    let mut startup_lap = |label: &str| {
        if startup_trace {
            eprintln!(
                "[startup] {label} {:.1}ms",
                startup_mark.elapsed().as_secs_f64() * 1e3
            );
            startup_mark = std::time::Instant::now();
        }
    };
    let engine = Engine::new(&config)?;
    startup_lap("engine_new");

    let module = load_main_module(&engine, module_path)?;
    startup_lap("load_module");

    let mut store = Store::new(&engine, Host::default());
    if guest_profile_out.is_some() {
        const SAMPLE_INTERVAL: std::time::Duration = std::time::Duration::from_micros(200);
        let profiler = wasmtime::GuestProfiler::new(
            &engine,
            "pyre-wasm",
            SAMPLE_INTERVAL,
            vec![("pyre_wasm".to_string(), module.clone())],
        )?;
        store.data_mut().guest_profiler = Some(profiler);
        store.set_epoch_deadline(1);
        store.epoch_deadline_callback(|mut ctx| {
            if let Some(mut p) = ctx.data_mut().guest_profiler.take() {
                p.sample(&ctx, std::time::Duration::ZERO);
                ctx.data_mut().guest_profiler = Some(p);
            }
            Ok(wasmtime::UpdateDeadline::Continue(1))
        });
        let ticker_engine = engine.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(SAMPLE_INTERVAL);
                ticker_engine.increment_epoch();
            }
        });
    }
    store.data_mut().stdlib_root = std::env::var("PYRE_STDLIB").ok();
    // Counting-only runs get the whole budget: `wasm_ops` is a subtraction, and
    // a run that exhausts its fuel reports the limit rather than its own cost.
    let fuel_start = meter_fuel.then(|| fuel_limit.unwrap_or(u64::MAX));
    if let Some(n) = fuel_start {
        store.set_fuel(n)?;
    }

    let linker = build_linker(&engine)?;
    let instance = linker
        .instantiate(&mut store, &module)
        .context("instantiate main module")?;
    startup_lap("instantiate");

    let memory = instance
        .get_memory(&mut store, "memory")
        .context("main module is missing its `memory` export")?;
    let table = instance
        .get_table(&mut store, "__indirect_function_table")
        .context("main module is missing its `__indirect_function_table` export (build with --export-table)")?;
    // The table's current size is the first slot a later `table.grow` will
    // return, i.e. the first JIT-trace id; everything below it is a main-module
    // function that must never be dispatched as, or freed like, a trace.
    let trace_base = table.size(&store);
    store.data_mut().memory = Some(memory);
    store.data_mut().table = Some(table);
    store.data_mut().trace_base = trace_base;

    let alloc = instance.get_typed_func::<u32, u32>(&mut store, "pyre_alloc")?;
    let run_python = instance.get_typed_func::<(u32, u32), u64>(&mut store, "pyre_run_python")?;
    let dealloc = instance.get_typed_func::<(u32, u32), ()>(&mut store, "pyre_dealloc")?;
    if let Some(selector) = std::env::var("PYRE_WASM_FORCE_CA_TERMINAL_DECLINE")
        .ok()
        .filter(|value| !value.is_empty())
    {
        let selector = selector.parse::<u64>().context(
            "PYRE_WASM_FORCE_CA_TERMINAL_DECLINE must be 1 or a decimal JitCellToken number",
        )?;
        let set_force = instance
            .get_typed_func::<u64, ()>(&mut store, "pyre_set_force_ca_terminal_decline")
            .context("wasm module lacks terminal-decline regression hook")?;
        set_force.call(&mut store, selector)?;
    }

    // The environment `-P`, `-O`, PYTHONWARNINGS and the rest resolve against.
    // The guest's own is permanently empty, so without this every one of them
    // reads as unset there while working natively; the runner only forwards the
    // values, the fold stays in `launch_env::finalize` where the launcher's is.
    // The module names which variables it wants, so the two never drift apart.
    //
    // A module predating `pyre_set_launch_env` still understands the `-P` half
    // through `pyre_set_safe_path`; one predating both keeps its previous
    // behaviour of always seeding `sys.path[0]`. Both halves are resolved before
    // either is used, so a module carrying only one degrades to the fallback
    // instead of failing the run.
    let launch_env_names = instance
        .get_typed_func::<(), u64>(&mut store, "pyre_launch_env_names")
        .ok();
    let set_launch_env = instance
        .get_typed_func::<(u32, u32), ()>(&mut store, "pyre_set_launch_env")
        .ok();
    if let (Some(names), Some(set_launch_env)) = (launch_env_names, set_launch_env) {
        let packed = names.call(&mut store, ())?;
        let (nptr, nlen) = ((packed >> 32) as u32, packed as u32);
        let mut buf = vec![0u8; nlen as usize];
        memory.read(&store, nptr as usize, &mut buf)?;
        dealloc.call(&mut store, (nptr, nlen))?;

        // Values are forwarded undecoded. The fold reads all but one of these
        // names through `env::var` and so drops a value that is not UTF-8 on
        // its own, matching the native launcher; the exception is the
        // `PYTHONSAFEPATH` presence flag, which `_Py_GetEnv` tests on raw
        // bytes. Decoding here would make such a value read as unset and leave
        // `sys.path[0]` seeded rather than suppressed, so the decision belongs
        // to the fold, not the transport. `into_encoded_bytes` round-trips a
        // valid-UTF-8 `OsString` to exactly its UTF-8 bytes, so the guest's
        // `from_utf8` reproduces `env::var`'s accept/reject split unchanged.
        let mut blob: Vec<u8> = Vec::new();
        for name in String::from_utf8_lossy(&buf)
            .split('\0')
            .filter(|name| !name.is_empty())
        {
            let Some(value) = std::env::var_os(name) else {
                continue;
            };
            if !blob.is_empty() {
                blob.push(0);
            }
            blob.extend_from_slice(name.as_bytes());
            blob.push(b'=');
            blob.extend_from_slice(&value.into_encoded_bytes());
        }

        let blen = blob.len() as u32;
        if blen == 0 {
            set_launch_env.call(&mut store, (0, 0))?;
        } else {
            let p = alloc.call(&mut store, blen)?;
            memory.write(&mut store, p as usize, &blob)?;
            set_launch_env.call(&mut store, (p, blen))?;
            dealloc.call(&mut store, (p, blen))?;
        }
    } else if std::env::var_os("PYTHONSAFEPATH").is_some_and(|value| !value.is_empty())
        && let Ok(set_safe_path) =
            instance.get_typed_func::<u32, ()>(&mut store, "pyre_set_safe_path")
    {
        set_safe_path.call(&mut store, 1)?;
    }

    // The environment the collector sizes itself from, forwarded the same way
    // and for the same reason. It is not a diagnostic knob: `PYPY_GC_MIN` and
    // `PYPY_GC_NURSERY` fix where the major-collection threshold falls, the
    // major step arms the eval-breaker word, and every compiled loop's back edge
    // polls that word through a real guard — so a guest that cannot read them
    // counts a different number of guard failures than the native backends run
    // beside it, from the same settings. Absent on a module predating the
    // export, which then keeps its built-in defaults.
    let gc_env_names = instance
        .get_typed_func::<(), u64>(&mut store, "pyre_gc_env_names")
        .ok();
    let set_gc_env = instance
        .get_typed_func::<(u32, u32), ()>(&mut store, "pyre_set_gc_env")
        .ok();
    if let (Some(names), Some(set_gc_env)) = (gc_env_names, set_gc_env) {
        let packed = names.call(&mut store, ())?;
        let (nptr, nlen) = ((packed >> 32) as u32, packed as u32);
        let mut buf = vec![0u8; nlen as usize];
        memory.read(&store, nptr as usize, &mut buf)?;
        dealloc.call(&mut store, (nptr, nlen))?;

        // `env::var`, not `var_os`: the guest parses these as numbers, so a
        // value that does not decode could not have been one and is left unset
        // exactly as it would be natively.
        let blob = String::from_utf8_lossy(&buf)
            .split('\0')
            .filter(|name| !name.is_empty())
            .filter_map(|name| {
                std::env::var(name)
                    .ok()
                    .map(|value| format!("{name}={value}"))
            })
            .collect::<Vec<_>>()
            .join("\0");

        let blen = blob.len() as u32;
        if blen != 0 {
            let p = alloc.call(&mut store, blen)?;
            memory.write(&mut store, p as usize, blob.as_bytes())?;
            set_gc_env.call(&mut store, (p, blen))?;
            dealloc.call(&mut store, (p, blen))?;
        }
    }

    // Name the script so the guest compiles it under its real path: that is
    // what a traceback prints, what its source-line lookup reads back through
    // `pyre_host.host_read`, and the directory that heads `sys.path`. Absent
    // on a module predating the export, leaving the guest on `<string>`.
    if let Ok(set_path) =
        instance.get_typed_func::<(u32, u32), ()>(&mut store, "pyre_set_script_path")
    {
        let name = script.to_string_lossy();
        let nlen = name.len() as u32;
        if nlen != 0 {
            let p = alloc.call(&mut store, nlen)?;
            memory.write(&mut store, p as usize, name.as_bytes())?;
            set_path.call(&mut store, (p, nlen))?;
            dealloc.call(&mut store, (p, nlen))?;
        }
    }

    // Arm the per-guard deopt census before the run, not after: it records on
    // the deopt path and stays off until asked. `MAJIT_GUARD_CENSUS` is inert
    // here for the usual reason, so the switch travels as a call. Absent on a
    // module predating the export, in which case the readout below finds
    // nothing and says so.
    if std::env::var_os("PYRE_WASM_GUARD_CENSUS").is_some()
        && let Ok(arm) =
            instance.get_typed_func::<(), ()>(&mut store, "pyre_jit_guard_census_enable")
    {
        arm.call(&mut store, ())?;
    }
    if std::env::var_os("PYRE_WASM_TRACE_ENTRY_CENSUS").is_some()
        && let Ok(arm) =
            instance.get_typed_func::<(), ()>(&mut store, "pyre_jit_trace_entry_census_enable")
    {
        arm.call(&mut store, ())?;
    }
    if std::env::var_os("PYRE_WASM_REEMIT").is_some()
        && let Ok(arm) = instance.get_typed_func::<(), ()>(&mut store, "pyre_jit_reemit_enable")
    {
        arm.call(&mut store, ())?;
    }
    if std::env::var_os("PYRE_WASM_INLINE_BRIDGE").is_some()
        && let Ok(arm) =
            instance.get_typed_func::<(), ()>(&mut store, "pyre_jit_inline_bridge_enable")
    {
        arm.call(&mut store, ())?;
    }
    // Parameter bridge entries are the default. The guest has no environment,
    // so an explicit host-side opt-out must travel through this export before
    // tracing begins.
    if std::env::var_os("PYRE_WASM_BRIDGE_PARAMS")
        .is_some_and(|value| matches!(value.to_str().map(str::trim), Some("0" | "false" | "off")))
        && let Ok(arm) =
            instance.get_typed_func::<(), ()>(&mut store, "pyre_jit_bridge_params_disable")
    {
        arm.call(&mut store, ())?;
    }

    let src = source.as_bytes();
    let len = src.len() as u32;
    let in_ptr = if len == 0 {
        0
    } else {
        let p = alloc.call(&mut store, len)?;
        memory.write(&mut store, p as usize, src)?;
        p
    };

    // Keep the run result so the diagnostic stats can be read out even when the
    // guest traps (a panic, or a PYRE_WASM_FUEL exhaustion that interrupts a
    // livelock): the JIT counters are populated at compile time, before the run
    // finishes, and the diag exports just read statics that survive a trap.
    let run_result = run_python.call(&mut store, (in_ptr, len));
    startup_lap("run_python(bootstrap+script)");
    // After a fuel-exhaustion trap the store has no fuel, so the diagnostic
    // export calls below would themselves immediately trap and read as 0.
    // Refill so the readout reflects the real (compile-time) counter values.
    // Read the remaining budget first: `start - remaining` is the executed
    // instruction count, which makes "how much does one loop iteration cost"
    // answerable by differencing two runs at different trip counts. A run that
    // exhausted its budget reports `used == limit` and says nothing, so size the
    // limit above what the script needs.
    let wasm_ops = fuel_start.map(|start| {
        let remaining = store.get_fuel().unwrap_or(0);
        let used = start.saturating_sub(remaining);
        if let Some(limit) = fuel_limit {
            eprintln!("[fuel] used={used} remaining={remaining} limit={limit}");
        }
        let _ = store.set_fuel(u64::MAX);
        used
    });
    if let Some(path) = &guest_profile_out
        && let Some(p) = store.data_mut().guest_profiler.take()
    {
        match std::fs::File::create(path).context("create guest profile output") {
            Ok(f) => match p
                .finish(std::io::BufWriter::new(f))
                .context("write guest profile")
            {
                Ok(()) => eprintln!("[guest-profile] wrote {path}"),
                Err(err) => eprintln!("[guest-profile] failed to write {path}: {err:#}"),
            },
            Err(err) => eprintln!("[guest-profile] failed to create {path}: {err:#}"),
        }
    }
    if std::env::var_os("PYRE_WASM_JIT_STATS").is_some() {
        let lin_mem = memory.data_size(&store);
        // Split linear-memory growth into GC-retained vs. host-heap: a leak that
        // shows up here but NOT in oldgen/nursery is a Rust-heap leak, not GC
        // false-retention. The exports are diagnostic; tolerate their absence.
        let gc_oldgen = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_gc_oldgen_bytes")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(0);
        let gc_nursery = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_gc_nursery_bytes")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(0);
        let gc_minors = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_gc_minor_collections")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(0);
        let gc_majors = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_gc_major_collections")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(0);
        // `heap-prof` builds only: net-live guest-heap bytes/count. Distinguishes
        // a true not-freed leak (live grows with executes) from fragmentation.
        let heap_live_bytes = instance
            .get_typed_func::<(), i64>(&mut store, "pyre_heap_live_bytes")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(-1);
        let heap_live_count = instance
            .get_typed_func::<(), i64>(&mut store, "pyre_heap_live_count")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(-1);
        // Per-size-class net-live histogram (heap-prof builds): bucket i covers
        // sizes (8*(i-1), 8*i]. Surfaces the exact leaking size class.
        if let Ok(bucket) = instance.get_typed_func::<u32, i64>(&mut store, "pyre_heap_bucket") {
            let mut parts = Vec::new();
            for i in 0u32..64 {
                let n = bucket.call(&mut store, i).unwrap_or(0);
                if n > 1000 {
                    parts.push(format!("≤{}B:{}", i * 8, n));
                }
            }
            if !parts.is_empty() {
                eprintln!("[jit-stats] heap_buckets {}", parts.join(" "));
            }
        }
        // The `Counters.ABORT_*` split behind `loops_aborted` (diagnostic).
        // The native backends get this from MAJIT_LOG's summary, which the
        // guest cannot read; without it a wasm-only `loops_aborted` regression
        // names no reason. Labels mirror
        // `majit_metainterp::jitprof::ABORT_COUNTER_KINDS`, which is
        // append-only, so a guest older than this runner just reads 0 for a
        // trailing label.
        if let Ok(ab) = instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_abort_diag") {
            let labels = [
                "too_long",
                "bridge_or_generic",
                "bad_loop",
                "escape",
                "force_quasiimmut",
                "segmented_trace",
            ];
            let mut parts = Vec::with_capacity(labels.len());
            for (i, lbl) in labels.iter().enumerate() {
                let n = ab.call(&mut store, i as u32).unwrap_or(0);
                parts.push(format!("{lbl}={n}"));
            }
            eprintln!("[jit-stats] abort_diag {}", parts.join(" "));
        }
        // compile_bridge outcome tallies (diagnostic). 0=entered 1=declCALL_ASM
        // 2=declMultiPeel 3=declNotDirect 4=declRefHome 5=BRIDGE_OK
        // 6=loopClosing 7=srcHasPreamble 15=declCAHostTrampoline,
        // 16=forced terminal-decline regression hook.
        if let Ok(diag) = instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_bridge_diag") {
            let labels = [
                "entered",
                "decl_callasm",
                "decl_multipeel",
                "decl_notdirect",
                "decl_refhome",
                "BRIDGE_OK",
                "loopclosing",
                "src_preamble",
                "ml_descr_none",
                "ml_unsafe_label",
                "ml_arity_mismatch",
                "decl_noadvance",
                "ca_cell_set",
                "ca_cells_zero",
                "accepted_ca",
                "decl_ca_trampoline",
                "forced_ca_terminal_decline",
                "ml_no_descr",
                "ml_unpublished",
                "pub_peeled",
                "pub_flat",
                "pub_flat_skipped",
                "label_retracted",
                "cl_entered",
                "cl_ok",
                "cl_decl_unsupported",
                "cl_decl_host_reject",
                "cell_set",
                "cell_missing",
                "cell_rebridge",
                "reemit_failed",
                "reemit_ok",
                "inline_ok",
                "inline_decl_not_direct",
                "inline_decl_not_loop_closing",
                "inline_decl_not_reemittable",
                "inline_decl_already_owned",
                "inline_decl_frame",
                "inline_decl_not_header",
                "inline_decl_no_loop_label",
                "inline_decl_value_layout",
                "inline_decl_ref_layout",
                "inline_decl_missing_label",
                "inline_decl_other",
                "bridge_param_ok",
                "bridge_param_decl_source_frame",
                "bridge_param_decl_arity",
                "bridge_param_label_suppressed",
            ];
            let mut parts = Vec::new();
            for (i, lbl) in labels.iter().enumerate() {
                let n = diag.call(&mut store, i as u32).unwrap_or(0);
                parts.push(format!("{lbl}={n}"));
            }
            eprintln!("[jit-stats] bridge_diag {}", parts.join(" "));
        }
        if let Ok(geometry) =
            instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_inline_geometry_diag")
        {
            let mut parts = Vec::new();
            for i in 0..3 {
                let packed = geometry.call(&mut store, i).unwrap_or(0);
                if packed != 0 {
                    parts.push(format!("{}:{}/{}", i + 1, packed >> 32, packed as u32));
                }
            }
            if !parts.is_empty() {
                eprintln!("[jit-stats] inline_geometry {}", parts.join(" "));
            }
        }
        // Per-walk full-body-walk census (diagnostic). Prints the same record
        // the native backends print as `[fbw-census]` under PYRE_FBW_CENSUS,
        // which the guest cannot read. Slot layout in
        // `pyre_jit_trace::trace::fbw_diag`.
        if let Ok(fbw) = instance.get_typed_func::<u32, u64>(&mut store, "pyre_fbw_diag") {
            const RING_BASE: u32 = 14;
            const RING_ENTRIES: u32 = 24;
            const RING_STRIDE: u32 = 5;
            const NAME_SLOTS: u32 = 4;
            let mut slot = |i: u32| fbw.call(&mut store, i).unwrap_or(0);
            let walks = slot(0);
            eprintln!(
                "[jit-stats] fbw_diag walks={walks} ROLLED_BACK_WITH_EFFECTS={} \
                 midbody_latch={}/{} escape_plain_fallback={}/{} \
                 fbw_store_journal_rollback_failed={}",
                slot(1),
                slot(3),
                slot(2),
                slot(5),
                slot(4),
                slot(11),
            );
            for entry in 0..RING_ENTRIES.min(walks as u32) {
                let base = RING_BASE + entry * RING_STRIDE;
                let mut end = String::new();
                for name_slot in 0..NAME_SLOTS {
                    let packed = slot(base + name_slot);
                    for byte in 0..8 {
                        let b = ((packed >> (byte * 8)) & 0xff) as u8;
                        if b != 0 {
                            end.push(b as char);
                        }
                    }
                }
                let flags = slot(base + NAME_SLOTS);
                if flags & 1 == 0 {
                    continue;
                }
                let field = |shift: u32| (flags >> shift) & 0xffff;
                eprintln!(
                    "[fbw-census] end={end} committed={} leg={} bridge={} exec_mf={} \
                     effects={} journaled={}",
                    flags & (1 << 1) != 0,
                    (flags >> 56) & 0xff,
                    flags & (1 << 2) != 0,
                    field(40),
                    field(8),
                    field(24),
                );
            }
        }
        // must_compile / start_retrace gate tallies (diagnostic). Positional
        // mirror of `majit_metainterp::MC_DIAG_LABELS`; the runner links no
        // majit crate, so a new slot must be appended in both. Appending keeps
        // the existing indices, which is the only edit that cannot silently
        // rename every tally after it.
        if let Ok(mc) = instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_mc_diag") {
            let labels = [
                "mc_entered",
                "decl_shortcircuit",
                "descr0_skip",
                "busy_skip",
                "FIRED",
                "stack_full",
                "retrace_entered",
                "retrace_bailed",
                "cb_entered",
                "cb_invalidloop",
                "cb_retrace_req",
                "cb_arity_giveup",
                "sbt_entered",
                "sbt_not_faildescr",
                "sbt_no_jct",
                "sbt_no_meta",
                "sbt_cant_trace",
                "sbt_short_vals",
                "caro_backedge",
                "caro_funcentry",
                "caro_no_portal",
                "caro_no_merge_entry",
                "caro_not_tracing",
                "stfe_cell_busy",
                "stfe_dead_token",
                "stfe_tick",
                "toolong_suppressed",
                "wct_declined",
                "cl_hct_giveup",
                "ct_no_front_token",
                "ct_origin_loop_gone",
                "ct_entry_bridge_failed",
                "ct_no_entry_data",
                "ct_not_tracing",
                "ceb_no_loop",
                "ceb_dead_token",
                "ceb_invalidloop",
                "ceb_retrace_req",
                "ceb_backend_panic",
                "ceb_backend_err",
                "abrt_too_long",
                "abrt_bridge",
                "abrt_bad_loop",
                "abrt_escape",
                "abrt_force_qmut",
                "abrt_segmented",
                "abrt_other",
                "giveup_invalidloop",
                "giveup_compileloop_err",
                "giveup_bridge_aborted",
                "bridge_declined_close",
                "bridge_no_targets_close",
                "abort_after_declined",
                "ct_compile_bridge_false",
                "mp_shape_filtered",
                "retrace_mp_untyped",
                "close_hdr_fallback",
                "retrace_arity_giveup",
                "ptp_push",
                "ptp_pop",
                "forced_never_compiled",
                "mst_entered",
                "mst_sync_before_false",
                "mst_live_values_mismatch",
                "stfe_declined_compiled",
                "stfe_declined_tracing",
                "stfe_declined_tracing_stale",
                "bridge_unattempted_close",
                "xloop_close_decision_reached",
                "xloop_close_target_compiled",
                "xloop_close_published",
                "abrt_unclassified_default",
                "unroll_cancelled_invalid_loop",
                "unroll_free_retry_rescued",
                "unroll_free_retry_failed",
                "qmut_deps_simple_loop",
                "qmut_deps_entry_bridge",
                "qmut_deps_blackhole_arm",
                "retrace_close_resumed",
            ];
            let mut parts = Vec::new();
            for (i, lbl) in labels.iter().enumerate() {
                parts.push(format!(
                    "{lbl}={}",
                    mc.call(&mut store, i as u32).unwrap_or(0)
                ));
            }
            eprintln!("[jit-stats] mc_diag {}", parts.join(" "));
        }
        // Full-body-walk decline census: the `DispatchError` variant behind
        // every aborted walk, which is what `loops_aborted` counts. Unlike the
        // fixed-slot tallies above this one is name-keyed, so the guest hands
        // back `(ptr << 32) | len` into its own memory and the name is sliced
        // out here — `PYRE_FBW_DEBUG_ABORT`, the native channel for the same
        // attribution, never reaches a guest with no environment.
        if let (Ok(len_fn), Ok(name_fn), Ok(count_fn)) = (
            instance.get_typed_func::<(), u32>(&mut store, "pyre_jit_fbw_census_len"),
            instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_fbw_census_name"),
            instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_fbw_census_count"),
        ) {
            let n = len_fn.call(&mut store, ()).unwrap_or(0);
            let mut parts = Vec::new();
            for i in 0..n {
                let packed = name_fn.call(&mut store, i).unwrap_or(0);
                let count = count_fn.call(&mut store, i).unwrap_or(0);
                let (ptr, len) = ((packed >> 32) as usize, (packed & 0xffff_ffff) as usize);
                let data = memory.data(&store);
                let name = data
                    .get(ptr..ptr + len)
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .unwrap_or("<unreadable>");
                parts.push(format!("{name}={count}"));
            }
            if !parts.is_empty() {
                eprintln!("[jit-stats] fbw_census {}", parts.join(" "));
            }
        }
        let guest_jit_execute_count = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_jit_execute_count")
            .and_then(|f| f.call(&mut store, ()))
            .ok();
        let guest_jit_compile_count = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_jit_compile_count")
            .and_then(|f| f.call(&mut store, ()))
            .ok();
        let guest_jit_compile_cache_hits = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_jit_compile_cache_hits")
            .and_then(|f| f.call(&mut store, ()))
            .ok();
        let host = store.data();
        eprintln!(
            "[jit-stats] compiles={} compile_ms={:.1} executes={} jit_calls={} jit_call_ms={:.1} \
             wasm_ops={} linear_mem={} gc_oldgen={} gc_nursery={} \
             gc_minors={} gc_majors={} heap_live_bytes={} heap_live_count={}",
            host.jit_compile_count,
            host.jit_compile_time_ns as f64 / 1.0e6,
            guest_jit_execute_count.unwrap_or(host.jit_execute_count),
            host.jit_call_count,
            host.jit_call_time_ns as f64 / 1.0e6,
            wasm_ops.map_or(-1, |n| n as i64),
            lin_mem,
            gc_oldgen,
            gc_nursery,
            gc_minors,
            gc_majors,
            heap_live_bytes,
            heap_live_count,
        );
        if let (Some(materialized), Some(cache_hits)) =
            (guest_jit_compile_count, guest_jit_compile_cache_hits)
        {
            eprintln!(
                "[jit-stats] materialized={} compile_cache_hits={}",
                materialized, cache_hits
            );
        }
        if !host.exec_hist.is_empty() {
            let mut v: Vec<_> = host.exec_hist.iter().collect();
            v.sort_by(|a, b| b.1.cmp(a.1));
            for ((func_id, fail_index), n) in v.into_iter().take(8) {
                eprintln!(
                    "[jit-stats] exec_hist func_id={func_id} fail_index={fail_index} roundtrips={n}"
                );
            }
        }
    }
    probe_call_hist_dump();
    // Emit the sign-stable badness counters under MAJIT_STATS (the same env
    // check.py sets for every backend), so the regression floor gates wasm on
    // the same fields as the native backends. Kept separate from the verbose
    // PYRE_WASM_JIT_STATS block above; which line they land on no longer
    // matters, since check.py merges every [jit-stats] line into one snapshot.
    // This line is the ONLY thing standing between a wasm-only descr-universe
    // regression and a green gate, because `_jit_stats_change` reads a
    // field absent from the current run as zero — the healthy value. So a
    // counter that cannot be read must not resolve to zero: a module that
    // predates these exports would then look perfectly healthy while gating on
    // nothing at all, which is the exact hole this block exists to close.
    if std::env::var_os("MAJIT_STATS").is_some() {
        let mut missing: Vec<&str> = Vec::new();
        let mut counter = |name: &'static str, missing: &mut Vec<&'static str>| match instance
            .get_typed_func::<(), u64>(&mut store, name)
            .and_then(|f| f.call(&mut store, ()))
        {
            Ok(v) => v,
            Err(_) => {
                missing.push(name);
                0
            }
        };
        let loops_compiled = counter("pyre_jit_loops_compiled", &mut missing);
        let bridges_compiled = counter("pyre_jit_bridges_compiled", &mut missing);
        let retraces_compiled = counter("pyre_jit_retraces_compiled", &mut missing);
        let loops_aborted = counter("pyre_jit_loops_aborted", &mut missing);
        let guard_failures = counter("pyre_jit_guard_failures", &mut missing);
        let back_edge_polls = counter("pyre_jit_back_edge_polls", &mut missing);
        let internal_compile_panics = counter("pyre_jit_internal_compile_panics", &mut missing);
        let descr_set_resolved = counter("pyre_jit_descr_set_resolved", &mut missing);
        let descr_set_absent = counter("pyre_jit_descr_set_absent", &mut missing);
        let descr_set_ambiguous = counter("pyre_jit_descr_set_ambiguous", &mut missing);
        let descr_set_stale_absent = counter("pyre_jit_descr_set_stale_absent", &mut missing);
        let field_pos_spec_checked = counter("pyre_jit_field_pos_spec_checked", &mut missing);
        let field_pos_spec_misplaced = counter("pyre_jit_field_pos_spec_misplaced", &mut missing);
        let field_pos_attached_checked =
            counter("pyre_jit_field_pos_attached_checked", &mut missing);
        let field_pos_attached_misplaced =
            counter("pyre_jit_field_pos_attached_misplaced", &mut missing);
        // The rest of the `field_position_census`. `field_pos_unresolved` is
        // the only counter that sees fields whose parent does not contain them:
        // both halves of the `attached_*` pair are counted inside a lookup that
        // must first succeed, so that population is missing from numerator and
        // denominator alike and the pair reads healthy while blind.
        //
        // Asking for them is what makes them gateable. The refusal below fires
        // only for names in THIS list, so a counter nobody asks for is not
        // caught by it — it simply never appears, and `_jit_stats_change` reads
        // an absent field as 0, the healthy value.
        let field_pos_parent_absent = counter("pyre_jit_field_pos_parent_absent", &mut missing);
        let field_pos_parent_empty = counter("pyre_jit_field_pos_parent_empty", &mut missing);
        let field_pos_rederived = counter("pyre_jit_field_pos_rederived", &mut missing);
        let field_pos_unresolved = counter("pyre_jit_field_pos_unresolved", &mut missing);
        // Walks that ended uncommitted after a residual had already run an
        // irreversible effect. Reached through the slot-indexed `pyre_fbw_diag`
        // export rather than a counter of its own; slot 1 is
        // `pyre_jit_trace::trace::fbw_diag::ROLLED_BACK_WITH_EFFECTS`. It carries
        // the same key the native backends print, because `_jit_stats_change`
        // compares by name and a name only one backend emits gates nothing on
        // the others.
        // Slot 11 is `STORE_JOURNAL_ROLLBACK_FAILED`, the store restores the
        // rollback could not perform — the one journaled effect the walk-end
        // subtraction must not silently absorb. Both slots come from one
        // export lookup so an absent export names itself once.
        let (
            fbw_rolled_back_with_effects,
            fbw_store_journal_rollback_failed,
            fbw_blackhole_adopted_single_frame,
            fbw_blackhole_adopted_multi_frame,
        ) = match instance
            .get_typed_func::<u32, u64>(&mut store, "pyre_fbw_diag")
            .and_then(|f| {
                Ok((
                    f.call(&mut store, 1)?,
                    f.call(&mut store, 11)?,
                    f.call(&mut store, 12)?,
                    f.call(&mut store, 13)?,
                ))
            }) {
            Ok(slots) => slots,
            Err(_) => {
                missing.push("pyre_fbw_diag");
                (0, 0, 0, 0)
            }
        };
        if !missing.is_empty() {
            eprintln!(
                "pyre-wasm-runner: MAJIT_STATS is set but the module exports no {} — \
                 refusing to report a gated counter as 0, which reads as healthy",
                missing.join(", "),
            );
            std::process::exit(1);
        }
        eprintln!(
            "[jit-stats] loops_compiled={loops_compiled} \
             bridges_compiled={bridges_compiled} \
             retraces_compiled={retraces_compiled} \
             loops_aborted={loops_aborted} \
             guard_failures={guard_failures} \
             back_edge_polls={back_edge_polls} \
             internal_compile_panics={internal_compile_panics} \
             descr_set_resolved={descr_set_resolved} \
             descr_set_absent={descr_set_absent} \
             descr_set_ambiguous={descr_set_ambiguous} \
             descr_set_stale_absent={descr_set_stale_absent} \
             fbw_rolled_back_with_effects={fbw_rolled_back_with_effects} \
             field_pos_spec_checked={field_pos_spec_checked} \
             field_pos_spec_misplaced={field_pos_spec_misplaced} \
             field_pos_attached_checked={field_pos_attached_checked} \
             field_pos_attached_misplaced={field_pos_attached_misplaced} \
             field_pos_parent_absent={field_pos_parent_absent} \
             field_pos_parent_empty={field_pos_parent_empty} \
             field_pos_rederived={field_pos_rederived} \
             field_pos_unresolved={field_pos_unresolved} \
             fbw_store_journal_rollback_failed={fbw_store_journal_rollback_failed} \
             fbw_blackhole_adopted_single_frame={fbw_blackhole_adopted_single_frame} \
             fbw_blackhole_adopted_multi_frame={fbw_blackhole_adopted_multi_frame}"
        );
    }
    let packed = match run_result {
        Ok(p) => p,
        Err(e) => {
            // wasm32-unknown-unknown has no stderr, but pyre-wasm's panic hook
            // writes "panicked at …" into linear memory before the trap.
            // Recover the formatted message (the heap String, not the static
            // format template) so the real cause is visible.
            for msg in recover_panic_messages(memory.data(&store)) {
                eprintln!("pyre-wasm-runner: recovered panic: {msg}");
            }
            // Everything the run wrote to fd 2 sits in the guest's buffer until
            // `pyre_take_stderr` drains it, and the success path below is the
            // only place that used to do so — so a trap discarded the whole
            // channel at the one moment it carries the most: the diagnostics a
            // fatal path emits just before aborting. A trap does not poison the
            // store, so the export is still callable; ignore it if the module
            // predates the export or the call itself fails.
            if let Ok(bytes) = take_guest_stderr(&mut store, &instance, &memory)
                && !bytes.is_empty()
            {
                use std::io::Write;
                let _ = std::io::stderr().write_all(&bytes);
            }
            return Err(e);
        }
    };
    let out_ptr = (packed >> 32) as u32;
    let out_len = (packed & 0xffff_ffff) as u32;

    let mut out = vec![0u8; out_len as usize];
    if out_len != 0 {
        memory.read(&store, out_ptr as usize, &mut out)?;
        dealloc.call(&mut store, (out_ptr, out_len))?;
    }
    // The guest has no descriptors, so its fd-2 bytes and its exit status come
    // back through their own exports. Absent on a module predating them, in
    // which case the run keeps the old stdout-only, always-0 behaviour.
    let err_bytes = take_guest_stderr(&mut store, &instance, &memory)?;
    if let Ok(errors) =
        instance.get_typed_func::<(), u64>(&mut store, "pyre_jit_inline_trial_errors")
        && let Ok(packed) = errors.call(&mut store, ())
    {
        let (ptr, len) = ((packed >> 32) as u32, packed as u32);
        if len != 0 {
            let mut bytes = vec![0u8; len as usize];
            memory.read(&store, ptr as usize, &mut bytes)?;
            dealloc.call(&mut store, (ptr, len))?;
            eprintln!(
                "[jit-stats] inline_trial_errors {}",
                String::from_utf8_lossy(&bytes)
            );
        }
    }
    // `PYRE_FBW_DEBUG_ABORT` cannot select the walker's decline census in the
    // guest, and its `eprintln!` would reach nothing anyway, so the census
    // comes back through its own export and is printed here. Absent on a
    // module predating the export, in which case there is nothing to print.
    if std::env::var_os("PYRE_WASM_FBW_CENSUS").is_some()
        && let Ok(census) = instance.get_typed_func::<(), u64>(&mut store, "pyre_fbw_census")
    {
        let census_result: Result<()> = (|| {
            let packed = census.call(&mut store, ())?;
            let (ptr, clen) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
            if clen != 0 {
                let mut bytes = vec![0u8; clen as usize];
                memory.read(&store, ptr as usize, &mut bytes)?;
                dealloc.call(&mut store, (ptr, clen))?;
                eprint!("{}", String::from_utf8_lossy(&bytes));
            } else {
                eprintln!("[fbw-census] (no declines recorded)");
            }
            Ok(())
        })();
        if let Err(err) = census_result {
            eprintln!("pyre-wasm-runner: fbw census failed: {err}");
        }
    }
    // The per-guard deopt census armed before the run, printed in the shape
    // `pyrex` prints it natively so the two are directly comparable. A bare
    // `guard_failures` total cannot say whether the excess is one guard the
    // guest keeps returning through or a spread of cold ones.
    if std::env::var_os("PYRE_WASM_GUARD_CENSUS").is_some() {
        match instance.get_typed_func::<(), u64>(&mut store, "pyre_jit_guard_census") {
            Ok(census) => {
                let census_result: Result<()> = (|| {
                    let packed = census.call(&mut store, ())?;
                    let (ptr, clen) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
                    let mut bytes = vec![0u8; clen as usize];
                    if clen != 0 {
                        memory.read(&store, ptr as usize, &mut bytes)?;
                        dealloc.call(&mut store, (ptr, clen))?;
                    }
                    eprintln!("[jit-stats] {}", String::from_utf8_lossy(&bytes));
                    Ok(())
                })();
                if let Err(err) = census_result {
                    eprintln!("pyre-wasm-runner: guard census failed: {err}");
                }
            }
            Err(_) => eprintln!("[jit-stats] guard_census=unexported"),
        }
    }
    if std::env::var_os("PYRE_WASM_TRACE_ENTRY_CENSUS").is_some() {
        match instance.get_typed_func::<(), u64>(&mut store, "pyre_jit_trace_entry_census") {
            Ok(census) => {
                let census_result: Result<()> = (|| {
                    let packed = census.call(&mut store, ())?;
                    let (ptr, clen) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
                    if clen != 0 {
                        let mut bytes = vec![0u8; clen as usize];
                        memory.read(&store, ptr as usize, &mut bytes)?;
                        dealloc.call(&mut store, (ptr, clen))?;
                        eprint!("{}", String::from_utf8_lossy(&bytes));
                    }
                    Ok(())
                })();
                if let Err(err) = census_result {
                    eprintln!("pyre-wasm-runner: trace entry census failed: {err}");
                }
            }
            Err(_) => eprintln!("[trace-entry-census] unexported"),
        }
    }
    let exit_code = instance
        .get_typed_func::<(), i32>(&mut store, "pyre_exit_code")
        .and_then(|f| f.call(&mut store, ()))
        .unwrap_or(0);
    if len != 0 {
        dealloc.call(&mut store, (in_ptr, len))?;
    }

    use std::io::Write;
    std::io::stdout().write_all(&out)?;
    std::io::stdout().flush()?;
    if !err_bytes.is_empty() {
        std::io::stderr().write_all(&err_bytes)?;
    }
    startup_lap("dealloc+stdout");
    // Exit without running the wasmtime `Store`/`Module`/`Engine` destructors.
    // Dropping them frees ~40MB of mapped code + linear memory that the OS
    // reclaims on process exit anyway; on a short run that teardown is ~0.2s —
    // the dominant fixed startup tax, larger than the module load and far
    // larger than trace compilation. stdout is already flushed above and the
    // diagnostic stats print to (unbuffered) stderr, so nothing is lost.
    // `PYRE_WASM_FULL_TEARDOWN=1` restores the drops for leak diagnostics.
    if std::env::var_os("PYRE_WASM_FULL_TEARDOWN").is_none() {
        std::io::stderr().flush().ok();
        std::process::exit(exit_code);
    }
    Ok(exit_code)
}

/// Load the main module, using a compiled `<module>.cwasm` cache to skip
/// cranelift recompilation of the ~14MB module on every process start.
///
/// The cache is the engine's own `Module::serialize` artifact, so it is only
/// usable by a byte-compatible engine build; `Module::deserialize_file`
/// validates that and errors on mismatch. `deserialize_file` runs trusted
/// precompiled native code and the `.cwasm` is otherwise independent of the
/// `.wasm` contents, so the cache is bound to the exact bytes it was produced
/// from: a sidecar `<module>.cwasm.sha256` records the SHA-256 of those bytes,
/// and the cache is deserialized only when it matches the current module's
/// hash. A rebuilt module or a pre-placed `.cwasm` therefore recompiles instead
/// of running a stale or untrusted artifact. Set `PYRE_WASM_NO_CACHE` to bypass
/// the cache entirely.
fn load_main_module(engine: &Engine, module_path: &Path) -> Result<Module> {
    let cache_disabled = std::env::var_os("PYRE_WASM_NO_CACHE").is_some();
    let wasm_bytes = std::fs::read(module_path)
        .with_context(|| format!("read wasm module {}", module_path.display()))?;
    let hash = wasm_content_hash(&wasm_bytes);
    let cache_path = cache_path_for(module_path);
    let key_path = cache_key_path_for(module_path);

    if !cache_disabled && cache_key_matches(&key_path, &hash) {
        // SAFETY: the artifact was produced by this runner's own engine via
        // `Module::serialize`; `deserialize_file` re-checks engine/version
        // compatibility and returns Err (not UB) if it cannot be trusted. The
        // key check above further proves it was compiled from the exact bytes
        // we just read.
        match unsafe { Module::deserialize_file(engine, &cache_path) } {
            Ok(m) => return Ok(m),
            Err(_) => { /* incompatible cache; recompile below */ }
        }
    }

    let module = Module::new(engine, &wasm_bytes[..])
        .with_context(|| format!("load wasm module {}", module_path.display()))?;
    if !cache_disabled && let Ok(bytes) = module.serialize() {
        // Best-effort: a failed cache write only forgoes the speedup. Write
        // the artifact before the key so an interrupted write never leaves a
        // key pointing at a half-written `.cwasm`.
        if std::fs::write(&cache_path, bytes).is_ok() {
            let _ = std::fs::write(&key_path, &hash);
        }
    }
    Ok(module)
}

/// `<module>.cwasm` next to the module (full name kept, so
/// `pyre_wasm.wasm-host.wasm` → `pyre_wasm.wasm-host.wasm.cwasm`).
fn cache_path_for(module_path: &Path) -> PathBuf {
    let mut s = module_path.as_os_str().to_owned();
    s.push(".cwasm");
    PathBuf::from(s)
}

/// Sidecar recording the SHA-256 of the `.wasm` its `.cwasm` was compiled from.
fn cache_key_path_for(module_path: &Path) -> PathBuf {
    let mut s = module_path.as_os_str().to_owned();
    s.push(".cwasm.sha256");
    PathBuf::from(s)
}

/// Hex SHA-256 of the module bytes, used as the cache key.
fn wasm_content_hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// The cache is usable only if its key sidecar exists and matches `hash`.
fn cache_key_matches(key_path: &Path, hash: &str) -> bool {
    matches!(std::fs::read_to_string(key_path), Ok(k) if k.trim() == hash)
}

fn build_linker(engine: &Engine) -> Result<Linker<Host>> {
    let mut linker = Linker::new(engine);

    linker.func_wrap(
        "pyre_jit",
        "jit_compile_wasm",
        |mut caller: Caller<'_, Host>, bytes_ptr: u32, bytes_len: u32| -> u32 {
            match jit_compile(&mut caller, bytes_ptr, bytes_len) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("[jit_compile_wasm] {e:?}");
                    0
                }
            }
        },
    )?;

    linker.func_wrap(
        "pyre_jit",
        "jit_replace_wasm",
        |mut caller: Caller<'_, Host>, func_id: u32, bytes_ptr: u32, bytes_len: u32| -> u32 {
            match jit_replace(&mut caller, func_id, bytes_ptr, bytes_len) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("[jit_replace_wasm] {e:?}");
                    0
                }
            }
        },
    )?;

    linker.func_wrap(
        "pyre_jit",
        "jit_execute_wasm",
        |mut caller: Caller<'_, Host>, func_id: u32, frame_ptr: u32| -> u32 {
            match jit_execute(&mut caller, func_id, frame_ptr) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[jit_execute_wasm] {e:?}");
                    0
                }
            }
        },
    )?;

    linker.func_wrap(
        "pyre_jit",
        "jit_free_wasm",
        |mut caller: Caller<'_, Host>, func_id: u32| {
            // Only a trace slot may be cleared; nulling a main-module slot
            // (`id < trace_base`) would corrupt the shared dispatch table.
            if (func_id as u64) >= caller.data().trace_base {
                // Release the table's hold on the trace function; the slot
                // itself stays (wasm tables cannot shrink).
                if let Some(table) = caller.data().table {
                    let _ = table.set(&mut caller, func_id as u64, Ref::Func(None));
                }
            }
        },
    )?;

    // Reflective residual-call trampoline for the *recording* / blackhole
    // path. The compiled trace reaches residual targets through `env.jit_call`
    // on the child module; the in-module metainterp cannot reflect a function's
    // wasm type to build a matching `call_indirect`, so it routes residual
    // calls here instead. Reuses the same call-area protocol and signature
    // reflection as `jit_call_trampoline` (the child's `env.jit_call`), so a
    // residual target whose real signature is not the uniform `(i64…) -> i64`
    // (e.g. the void `set_current_exception_fn`, or the `-> i64`
    // `store_subscr_fn` invoked in a void context) is coerced correctly
    // rather than trapping on an indirect-call type mismatch.
    linker.func_wrap(
        "pyre_jit",
        "jit_call_host",
        |mut caller: Caller<'_, Host>, frame_ptr: u32| {
            if let Err(e) = jit_call_trampoline(&mut caller, frame_ptr, CALL_RESULT_OFS as u32) {
                eprintln!("[jit_call_host] {e:?}");
            }
        },
    )?;

    // Host-filesystem imports for the wasm-host build's module loader. The wasm32
    // module has no filesystem of its own; these serve module source from the
    // host's real stdlib (`$PYRE_STDLIB`). See `pyre-wasm`'s `host_fs_provider`.
    linker.func_wrap(
        "pyre_host",
        "host_stdlib_root",
        |mut caller: Caller<'_, Host>, buf_ptr: u32, buf_cap: u32| -> i64 {
            host_stdlib_root(&mut caller, buf_ptr, buf_cap)
        },
    )?;
    linker.func_wrap(
        "pyre_host",
        "host_is_dir",
        |mut caller: Caller<'_, Host>, path_ptr: u32, path_len: u32| -> u32 {
            match host_path(&mut caller, path_ptr, path_len) {
                Some(p) => PathBuf::from(p).is_dir() as u32,
                None => 0,
            }
        },
    )?;
    linker.func_wrap(
        "pyre_host",
        "host_file_size",
        |mut caller: Caller<'_, Host>, path_ptr: u32, path_len: u32| -> i64 {
            match host_path(&mut caller, path_ptr, path_len) {
                Some(p) => match std::fs::metadata(&p) {
                    Ok(m) if m.is_file() => m.len() as i64,
                    _ => -1,
                },
                None => -1,
            }
        },
    )?;
    linker.func_wrap(
        "pyre_host",
        "host_read",
        |mut caller: Caller<'_, Host>,
         path_ptr: u32,
         path_len: u32,
         buf_ptr: u32,
         buf_cap: u32|
         -> i64 { host_read(&mut caller, path_ptr, path_len, buf_ptr, buf_cap) },
    )?;

    Ok(linker)
}

/// Read a host path argument out of wasm linear memory as a `String`.
fn host_path(caller: &mut Caller<'_, Host>, path_ptr: u32, path_len: u32) -> Option<String> {
    let memory = caller.data().memory?;
    let mut bytes = vec![0u8; path_len as usize];
    memory.read(&*caller, path_ptr as usize, &mut bytes).ok()?;
    String::from_utf8(bytes).ok()
}

/// `pyre_host.host_stdlib_root`: write `$PYRE_STDLIB` into the wasm buffer.
fn host_stdlib_root(caller: &mut Caller<'_, Host>, buf_ptr: u32, buf_cap: u32) -> i64 {
    let Some(root) = caller.data().stdlib_root.clone() else {
        return -1;
    };
    let bytes = root.as_bytes();
    if bytes.len() > buf_cap as usize {
        // Report the needed length without writing; the caller can retry.
        return bytes.len() as i64;
    }
    let Some(memory) = caller.data().memory else {
        return -1;
    };
    if memory.write(&mut *caller, buf_ptr as usize, bytes).is_err() {
        return -1;
    }
    bytes.len() as i64
}

/// `pyre_host.host_read`: read the host file into the wasm-provided buffer.
fn host_read(
    caller: &mut Caller<'_, Host>,
    path_ptr: u32,
    path_len: u32,
    buf_ptr: u32,
    buf_cap: u32,
) -> i64 {
    let Some(path) = host_path(caller, path_ptr, path_len) else {
        return -1;
    };
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(_) => return -1,
    };
    let n = data.len().min(buf_cap as usize);
    let Some(memory) = caller.data().memory else {
        return -1;
    };
    if memory
        .write(&mut *caller, buf_ptr as usize, &data[..n])
        .is_err()
    {
        return -1;
    }
    n as i64
}

/// Compile and instantiate a JIT-emitted trace module, sharing the main
/// module's linear memory and wiring the `jit_call` trampoline.
fn jit_compile_trace(
    caller: &mut Caller<'_, Host>,
    bytes_ptr: u32,
    bytes_len: u32,
) -> Result<(Table, Func)> {
    caller.data_mut().jit_compile_count += 1;
    let memory = caller
        .data()
        .memory
        .context("main memory not initialized")?;
    let table = caller.data().table.context("main table not initialized")?;

    let mut bytes = vec![0u8; bytes_len as usize];
    memory
        .read(&*caller, bytes_ptr as usize, &mut bytes)
        .context("read trace module bytes")?;

    let engine = caller.engine().clone();
    let compile_start = std::time::Instant::now();
    let module_result = Module::new(&engine, &bytes);
    caller.data_mut().jit_compile_time_ns += compile_start.elapsed().as_nanos();
    let module = match module_result {
        Ok(m) => m,
        Err(e) => {
            if std::env::var_os("PYRE_WASM_DUMP_BAD_TRACE").is_some() {
                let path = "/tmp/pyre_bad_trace.wasm";
                let _ = std::fs::write(path, &bytes);
                eprintln!("[jit_compile_wasm] dumped {} bytes to {path}", bytes.len());
                match wasmprinter::print_bytes(&bytes) {
                    Ok(wat) => eprintln!("--- WAT ---\n{wat}\n--- /WAT ---"),
                    Err(pe) => eprintln!("[jit_compile_wasm] wat print failed: {pe}"),
                }
            }
            return Err(e).context("compile trace module");
        }
    };

    // A fresh trampoline per trace; it reads all state from `caller.data()`.
    let jit_call = Func::wrap(
        &mut *caller,
        |mut inner: Caller<'_, Host>, frame_ptr: i32| {
            if let Err(e) =
                jit_call_trampoline(&mut inner, frame_ptr as u32, CALL_RESULT_OFS as u32)
            {
                eprintln!("[jit_call] {e:?}");
            }
        },
    );
    let jit_call_compact = Func::wrap(
        &mut *caller,
        |mut inner: Caller<'_, Host>, frame_ptr: i32, call_area_ofs: i32| {
            if let Err(e) = jit_call_trampoline(&mut inner, frame_ptr as u32, call_area_ofs as u32)
            {
                eprintln!("[jit_call_compact] {e:?}");
            }
        },
    );

    // Supply imports in the module's declared order.
    let mut externs: Vec<Extern> = Vec::new();
    for import in module.imports() {
        match (import.module(), import.name()) {
            ("env", "memory") => externs.push(Extern::Memory(memory)),
            ("env", "jit_call") => externs.push(Extern::Func(jit_call)),
            ("env", "jit_call_compact") => externs.push(Extern::Func(jit_call_compact)),
            ("env", "__indirect_function_table") => externs.push(Extern::Table(table)),
            (m, n) => {
                return Err(Error::msg(format!(
                    "trace module has unexpected import {m}.{n}"
                )));
            }
        }
    }

    let instance =
        Instance::new(&mut *caller, &module, &externs).context("instantiate trace module")?;
    if std::env::var_os("PYRE_WASM_DUMP_ALL_TRACES").is_some() {
        let trace_id = instance
            .get_global(&mut *caller, "trace_entry_census_id")
            .and_then(|global| global.get(&mut *caller).i64())
            .map(|id| format!(" trace_id={id}"))
            .unwrap_or_default();
        match wasmprinter::print_bytes(&bytes) {
            Ok(wat) => eprintln!(
                "=== trace module{trace_id} ({} bytes) ===\n{wat}",
                bytes.len()
            ),
            Err(pe) => eprintln!("[jit_compile_wasm] wat print failed: {pe}"),
        }
    }
    let trace = instance
        .get_func(&mut *caller, "trace")
        .context("trace module is missing its `trace` export")?;

    Ok((table, trace))
}

/// Compile and instantiate a trace, then append its export to the trace table.
fn jit_compile(caller: &mut Caller<'_, Host>, bytes_ptr: u32, bytes_len: u32) -> Result<u32> {
    let (table, trace) = jit_compile_trace(caller, bytes_ptr, bytes_len)?;
    // Register the trace into the shared indirect function table so it is
    // reachable by table index. `grow` returns the newly appended slot.
    let slot = table
        .grow(&mut *caller, 1, Ref::Func(Some(trace)))
        .context("register trace into shared table")? as u32;
    Ok(slot)
}

/// Compile and instantiate a trace, then replace an existing trace-table slot.
fn jit_replace(
    caller: &mut Caller<'_, Host>,
    func_id: u32,
    bytes_ptr: u32,
    bytes_len: u32,
) -> Result<u32> {
    if (func_id as u64) < caller.data().trace_base {
        return Err(Error::msg(format!(
            "jit_replace_wasm: id {func_id} is not a trace slot"
        )));
    }
    let (table, trace) = jit_compile_trace(caller, bytes_ptr, bytes_len)?;
    if !matches!(
        table.get(&mut *caller, func_id as u64),
        Some(Ref::Func(Some(_)))
    ) {
        return Err(Error::msg(format!(
            "jit_replace_wasm: id {func_id} is not a live trace"
        )));
    }
    // The Store retains every instantiated module. Replacing this table entry
    // therefore leaves an old trace live when a non-tail indirect call still
    // has one of its frames on the guest stack.
    table
        .set(&mut *caller, func_id as u64, Ref::Func(Some(trace)))
        .context("replace trace in shared table")?;
    Ok(func_id)
}

/// Run a previously compiled trace, returning its guard-exit index.
fn jit_execute(caller: &mut Caller<'_, Host>, func_id: u32, frame_ptr: u32) -> Result<u32> {
    caller.data_mut().jit_execute_count += 1;
    if (func_id as u64) < caller.data().trace_base {
        return Err(Error::msg(format!(
            "jit_execute_wasm: id {func_id} is not a trace slot"
        )));
    }
    let table = caller.data().table.context("main table not initialized")?;
    // The id IS the table slot; dispatch through the shared table by index —
    // the same lookup an in-module `call_indirect` would perform. A freed trace
    // (slot reset to `Func(None)`) or out-of-range id misses here.
    let trace = match table.get(&mut *caller, func_id as u64) {
        Some(Ref::Func(Some(f))) => f,
        _ => {
            return Err(Error::msg(format!(
                "jit_execute_wasm: id {func_id} is not a live trace (unknown or freed)"
            )));
        }
    };
    let mut results = [Val::I32(0)];
    trace.call(&mut *caller, &[Val::I32(frame_ptr as i32)], &mut results)?;
    let ret = match results[0] {
        Val::I32(x) => x as u32,
        _ => 0,
    };
    // Diagnostic: record which (trace, guard-exit fail_index) round-tripped.
    if std::env::var_os("PYRE_WASM_EXEC_TRACE").is_some()
        && let Some(mem) = caller.data().memory
    {
        let fail_index = read_u32(&mem, &*caller, frame_ptr as usize);
        *caller
            .data_mut()
            .exec_hist
            .entry((func_id, fail_index))
            .or_insert(0) += 1;
    }
    Ok(ret)
}

/// Dispatch a residual call requested by a running trace.
// PROBE(PYRE_WASM_CALL_HIST): temporary per-callee crossing histogram.
static PROBE_CALL_HIST: std::sync::Mutex<Option<std::collections::BTreeMap<u32, u64>>> =
    std::sync::Mutex::new(None);

/// Whether `PYRE_WASM_CALL_HIST` was set, read once.
///
/// `jit_call_trampoline` runs per residual call -- 15168 of them on
/// fib_recursive -- and `std::env::var_os` takes the process-wide environment
/// lock and scans it, so reading the variable there put that cost on every
/// crossing whether or not the probe was wanted.
fn probe_call_hist_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_WASM_CALL_HIST").is_some())
}

pub(crate) fn probe_call_hist_dump() {
    if !probe_call_hist_enabled() {
        return;
    }
    let g = PROBE_CALL_HIST.lock().unwrap();
    let Some(m) = g.as_ref() else { return };
    let total: u64 = m.values().sum();
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    eprintln!("[probe] call_hist total={total} distinct={}", v.len());
    for (slot, n) in v.into_iter().take(15) {
        let pct = if total > 0 {
            *n as f64 * 100.0 / total as f64
        } else {
            0.0
        };
        eprintln!("[probe] call_hist slot={slot} n={n} pct={pct:.1}");
    }
}

/// Time every residual crossing, including the paths that return early, so the
/// share of a run spent leaving the guest is measured rather than inferred from
/// `jit_call_count` alone.
fn jit_call_trampoline(
    caller: &mut Caller<'_, Host>,
    frame_ptr: u32,
    call_area_ofs: u32,
) -> Result<()> {
    let entered = std::time::Instant::now();
    let r = jit_call_trampoline_inner(caller, frame_ptr, call_area_ofs);
    caller.data_mut().jit_call_time_ns += entered.elapsed().as_nanos();
    r
}

fn jit_call_trampoline_inner(
    caller: &mut Caller<'_, Host>,
    frame_ptr: u32,
    call_area_ofs: u32,
) -> Result<()> {
    caller.data_mut().jit_call_count += 1;
    let memory = caller.data().memory.context("memory")?;
    let table = caller.data().table.context("table")?;
    let call_area = frame_ptr as usize + call_area_ofs as usize;

    let func_ptr = read_u32(&memory, &*caller, call_area + 8);
    if probe_call_hist_enabled() {
        let mut g = PROBE_CALL_HIST.lock().unwrap();
        *g.get_or_insert_with(Default::default)
            .entry(func_ptr)
            .or_insert(0) += 1;
    }

    // `func_ptr == 0` is the "newstr" sentinel; without a host string
    // allocator (matching the browser glue's null table slot) it yields 0.
    if func_ptr == 0 {
        write_i64(&memory, &mut *caller, call_area, 0)?;
        return Ok(());
    }

    let func = match table.get(&mut *caller, func_ptr as u64) {
        Some(Ref::Func(Some(f))) => f,
        _ => {
            write_i64(&memory, &mut *caller, call_area, 0)?;
            return Ok(());
        }
    };

    let ty = func.ty(&*caller);
    let params: Vec<ValType> = ty.params().collect();
    let mut args: Vec<Val> = Vec::with_capacity(params.len());
    for (i, pty) in params.iter().enumerate() {
        let raw = read_i64(&memory, &*caller, call_area + 24 + i * 8);
        args.push(match pty {
            ValType::I32 => Val::I32(raw as i32),
            ValType::I64 => Val::I64(raw),
            // Floats cross the call area as their raw bit pattern in an i64 slot.
            ValType::F32 => Val::F32(raw as u32),
            ValType::F64 => Val::F64(raw as u64),
            other => {
                return Err(Error::msg(format!(
                    "unsupported residual-call param type {other:?}"
                )));
            }
        });
    }

    let mut results: Vec<Val> = ty
        .results()
        .map(|t| match t {
            ValType::I64 => Val::I64(0),
            ValType::F32 => Val::F32(0),
            ValType::F64 => Val::F64(0),
            _ => Val::I32(0),
        })
        .collect();

    // Mirror the browser glue's try/catch: a trapping residual target is
    // reported as a zero result rather than aborting the whole run.
    if let Err(e) = func.call(&mut *caller, &args, &mut results) {
        eprintln!("[jit_call] residual target trapped: {e:?}");
        write_i64(&memory, &mut *caller, call_area, 0)?;
        return Ok(());
    }

    let result = match results.first() {
        Some(Val::I32(x)) => (*x as u32) as i64, // zero-extend; high word stays 0
        Some(Val::I64(x)) => *x,
        Some(Val::F64(x)) => *x as i64,
        Some(Val::F32(x)) => (*x as u64) as i64,
        _ => 0,
    };
    write_i64(&memory, &mut *caller, call_area, result)?;
    Ok(())
}

/// Drain the guest's fd-2 buffer through `pyre_take_stderr`.
///
/// The guest has no descriptors, so everything it writes to fd 2 accumulates in
/// linear memory until this export hands it over. Returns empty on a module
/// predating the export, which keeps the old stdout-only behaviour.
fn take_guest_stderr(
    store: &mut Store<Host>,
    instance: &Instance,
    memory: &Memory,
) -> Result<Vec<u8>> {
    let Ok(take_stderr) = instance.get_typed_func::<(), u64>(&mut *store, "pyre_take_stderr")
    else {
        return Ok(Vec::new());
    };
    let packed = take_stderr.call(&mut *store, ())?;
    let (ptr, len) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut bytes = vec![0u8; len as usize];
    memory.read(&*store, ptr as usize, &mut bytes)?;
    let dealloc = instance.get_typed_func::<(u32, u32), ()>(&mut *store, "pyre_dealloc")?;
    dealloc.call(&mut *store, (ptr, len))?;
    Ok(bytes)
}

/// Scan wasm linear memory for panic messages pyre-wasm's hook wrote there.
///
/// The hook prepends a literal `[pyre panic] ` to the panic info, and that
/// concatenation only exists at runtime (never as a static format template), so
/// matching the combined prefix recovers the real formatted message — including
/// the asserted values — rather than a `{…}`-placeholder template.
pub(crate) fn recover_panic_messages(data: &[u8]) -> Vec<String> {
    let needle = b"[pyre panic] panicked at";
    let mut out: Vec<String> = Vec::new();
    let mut from = 0;
    while let Some(rel) = data[from..].windows(needle.len()).position(|w| w == needle) {
        let pos = from + rel;
        let end = data[pos..]
            .iter()
            .position(|&b| b == 0)
            .map(|n| pos + n)
            .unwrap_or((pos + 400).min(data.len()));
        // Decode and cut at the first non-text byte (`U+FFFD` from lossy
        // decoding of trailing String capacity garbage).
        let text = String::from_utf8_lossy(&data[pos..end]);
        let text = text
            .split('\u{FFFD}')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if !text.is_empty() && !out.contains(&text) {
            out.push(text);
            if out.len() >= 4 {
                break;
            }
        }
        from = end + 1;
    }
    out
}

fn read_u32(mem: &Memory, store: impl AsContext, off: usize) -> u32 {
    let mut b = [0u8; 4];
    let _ = mem.read(store, off, &mut b);
    u32::from_le_bytes(b)
}

fn read_i64(mem: &Memory, store: impl AsContext, off: usize) -> i64 {
    let mut b = [0u8; 8];
    let _ = mem.read(store, off, &mut b);
    i64::from_le_bytes(b)
}

fn write_i64(mem: &Memory, mut store: impl AsContextMut, off: usize, v: i64) -> Result<()> {
    mem.write(&mut store, off, &v.to_le_bytes())
        .context("write call-area result")
}

/// Dump the module's imports and exports, for debugging the host contract.
fn inspect_module(module_path: &PathBuf) -> Result<()> {
    // Match `run`'s engine configuration.
    let config = Config::new();
    let engine = Engine::new(&config)?;
    let module = Module::from_file(&engine, module_path)
        .with_context(|| format!("load wasm module {}", module_path.display()))?;
    println!("imports:");
    for import in module.imports() {
        println!(
            "  {}.{} : {:?}",
            import.module(),
            import.name(),
            import.ty()
        );
    }
    println!("exports:");
    for export in module.exports() {
        println!("  {} : {:?}", export.name(), export.ty());
    }
    Ok(())
}
