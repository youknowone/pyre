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
    jit_execute_count: u64,
    /// Diagnostic: per-op residual-call host crossings (`env.jit_call`
    /// trampoline invocations). Compare against `jit_execute_count` to test
    /// whether per-op crossings or per-guard-exit crossings dominate.
    jit_call_count: u64,
    /// Diagnostic (PYRE_WASM_EXEC_TRACE=1): histogram of (trace func_id,
    /// guard-exit fail_index) over every host round-trip, so we can see which
    /// guard keeps returning to the host instead of chaining in-module.
    exec_hist: std::collections::BTreeMap<(u32, u32), u64>,
    /// `PYRE_WASM_GUEST_PROFILE` sampling profiler; taken/restored around each
    /// epoch tick so `sample` can borrow the store it lives in.
    guest_profiler: Option<wasmtime::GuestProfiler>,
}

fn main() {
    let mut module_path: Option<PathBuf> = None;
    let mut script: Option<PathBuf> = None;
    let mut inspect = false;
    let mut engine: Option<WasmEngine> = None;

    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--inspect" => inspect = true,
            "--module" => {
                module_path = Some(PathBuf::from(
                    argv.next()
                        .unwrap_or_else(|| fatal("--module needs a path")),
                ))
            }
            "--engine" => {
                let v = argv
                    .next()
                    .unwrap_or_else(|| fatal("--engine needs a value"));
                engine = Some(WasmEngine::parse(&v).unwrap_or_else(|e| fatal(&e)));
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: pyre-wasm-runner [--module <pyre_wasm.wasm>] \
                     [--engine wasmtime|wasmi] [--inspect] <script.py>"
                );
                std::process::exit(2);
            }
            other if other.starts_with('-') => fatal(&format!("unknown flag {other}")),
            other => script = Some(PathBuf::from(other)),
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
                WasmEngine::Wasmtime => run(&module_path, &source).map_err(fmt_err),
                WasmEngine::Wasmi => wasmi_host::run(&module_path, &source),
            }
        })
        .expect("spawn worker thread");

    match worker.join() {
        Ok(Ok(code)) => std::process::exit(code),
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

fn run(module_path: &PathBuf, source: &str) -> Result<i32> {
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
    if fuel_limit.is_some() {
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
    let engine = Engine::new(&config)?;

    let module = load_main_module(&engine, module_path)?;

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
    if let Some(n) = fuel_limit {
        store.set_fuel(n)?;
    }

    let linker = build_linker(&engine)?;
    let instance = linker
        .instantiate(&mut store, &module)
        .context("instantiate main module")?;

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
    // After a fuel-exhaustion trap the store has no fuel, so the diagnostic
    // export calls below would themselves immediately trap and read as 0.
    // Refill so the readout reflects the real (compile-time) counter values.
    if fuel_limit.is_some() {
        let _ = store.set_fuel(u64::MAX);
    }
    if let Some(path) = &guest_profile_out {
        if let Some(p) = store.data_mut().guest_profiler.take() {
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
            ];
            let mut parts = Vec::new();
            for (i, lbl) in labels.iter().enumerate() {
                let n = diag.call(&mut store, i as u32).unwrap_or(0);
                parts.push(format!("{lbl}={n}"));
            }
            eprintln!("[jit-stats] bridge_diag {}", parts.join(" "));
        }
        // must_compile / start_retrace gate tallies (diagnostic).
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
        let guest_jit_execute_count = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_jit_execute_count")
            .and_then(|f| f.call(&mut store, ()))
            .ok();
        let host = store.data();
        eprintln!(
            "[jit-stats] compiles={} executes={} jit_calls={} linear_mem={} gc_oldgen={} gc_nursery={} \
             gc_minors={} gc_majors={} heap_live_bytes={} heap_live_count={}",
            host.jit_compile_count,
            guest_jit_execute_count.unwrap_or(host.jit_execute_count),
            host.jit_call_count,
            lin_mem,
            gc_oldgen,
            gc_nursery,
            gc_minors,
            gc_majors,
            heap_live_bytes,
            heap_live_count,
        );
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
    // Emit the sign-stable badness counters under MAJIT_STATS (the same env
    // check.py sets for every backend), so the regression floor gates wasm on
    // loops_aborted / internal_compile_panics like the native backends. Kept
    // separate from the verbose PYRE_WASM_JIT_STATS block above and emitted
    // last, so under check.py (MAJIT_STATS only) this is the single [jit-stats]
    // line the snapshot picks up. The exports read 0 on an old module that
    // predates them, which is the healthy value, so a stale runner degrades
    // safely rather than reporting a false regression.
    if std::env::var_os("MAJIT_STATS").is_some() {
        let loops_aborted = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_jit_loops_aborted")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(0);
        let internal_compile_panics = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_jit_internal_compile_panics")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(0);
        eprintln!(
            "[jit-stats] loops_aborted={loops_aborted} \
             internal_compile_panics={internal_compile_panics}"
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
    if len != 0 {
        dealloc.call(&mut store, (in_ptr, len))?;
    }

    use std::io::Write;
    std::io::stdout().write_all(&out)?;
    std::io::stdout().flush()?;
    Ok(0)
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
    if !cache_disabled {
        if let Ok(bytes) = module.serialize() {
            // Best-effort: a failed cache write only forgoes the speedup. Write
            // the artifact before the key so an interrupted write never leaves a
            // key pointing at a half-written `.cwasm`.
            if std::fs::write(&cache_path, bytes).is_ok() {
                let _ = std::fs::write(&key_path, &hash);
            }
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

fn jit_compile(caller: &mut Caller<'_, Host>, bytes_ptr: u32, bytes_len: u32) -> Result<u32> {
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
    if std::env::var_os("PYRE_WASM_DUMP_ALL_TRACES").is_some() {
        match wasmprinter::print_bytes(&bytes) {
            Ok(wat) => eprintln!("=== trace module ({} bytes) ===\n{wat}", bytes.len()),
            Err(pe) => eprintln!("[jit_compile_wasm] wat print failed: {pe}"),
        }
    }
    let module = match Module::new(&engine, &bytes) {
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
    let trace = instance
        .get_func(&mut *caller, "trace")
        .context("trace module is missing its `trace` export")?;

    // Register the trace into the shared indirect function table so it is
    // reachable by table index. `grow` returns the previous size, i.e. the
    // index of the newly appended entry, which becomes this trace's id.
    let slot = table
        .grow(&mut *caller, 1, Ref::Func(Some(trace)))
        .context("register trace into shared table")? as u32;

    Ok(slot)
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
    if std::env::var_os("PYRE_WASM_EXEC_TRACE").is_some() {
        if let Some(mem) = caller.data().memory {
            let fail_index = read_u32(&mem, &*caller, frame_ptr as usize);
            *caller
                .data_mut()
                .exec_hist
                .entry((func_id, fail_index))
                .or_insert(0) += 1;
        }
    }
    Ok(ret)
}

/// Dispatch a residual call requested by a running trace.
fn jit_call_trampoline(
    caller: &mut Caller<'_, Host>,
    frame_ptr: u32,
    call_area_ofs: u32,
) -> Result<()> {
    caller.data_mut().jit_call_count += 1;
    let memory = caller.data().memory.context("memory")?;
    let table = caller.data().table.context("table")?;
    let call_area = frame_ptr as usize + call_area_ofs as usize;

    let func_ptr = read_u32(&memory, &*caller, call_area + 8);

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
