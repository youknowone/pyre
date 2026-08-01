// `web` (wasm-bindgen) and `wasm-host` (C-ABI) export conflicting `run_python`
// surfaces; exactly one host binding may be active at a time.
#[cfg(all(feature = "web", feature = "wasm-host"))]
compile_error!("features `web` and `wasm-host` are mutually exclusive");

// The wasm-host C-ABI packs a result pointer and length into the high/low
// halves of a u64, which only round-trips with 32-bit pointers.
#[cfg(all(feature = "wasm-host", not(target_arch = "wasm32")))]
compile_error!("feature `wasm-host` requires target_arch = \"wasm32\"");

#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

// Native-host (`wasm-host`) builds target wasm32-unknown-unknown, which has no OS
// entropy. To avoid the wasm-bindgen-based `wasm_js` backend (whose imports a
// non-JS embedder cannot satisfy), getrandom is wired to its `custom` backend
// via `--cfg getrandom_backend="custom"`, which calls this hook. pyre seeds only
// non-cryptographic uses (string hash key, the `random` module) from it, and the
// values never affect check.py's oracle comparison, so a deterministic
// SplitMix64 stream is sufficient.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
mod custom_getrandom {
    use core::sync::atomic::{AtomicU64, Ordering};

    static STATE: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);

    #[unsafe(no_mangle)]
    unsafe extern "Rust" fn __getrandom_v03_custom(
        dest: *mut u8,
        len: usize,
    ) -> Result<(), getrandom::Error> {
        let mut i = 0;
        while i < len {
            let mut z = STATE
                .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
                .wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            let bytes = z.to_le_bytes();
            let n = core::cmp::min(8, len - i);
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), dest.add(i), n) };
            i += n;
        }
        Ok(())
    }
}

#[cfg(any(feature = "web", feature = "wasm-host"))]
use pyre_interpreter::*;

#[cfg(any(feature = "web", feature = "wasm-host"))]
use std::cell::RefCell;
#[cfg(any(feature = "web", feature = "wasm-host"))]
use std::sync::Once;

// Diagnostic counting allocator (feature `heap-prof`): wraps the platform
// allocator and tracks net-live bytes/count so the host runner can tell a true
// not-freed leak (live grows linearly) from fragmentation (live flat,
// linear-memory still grows). realloc delegates to keep dlmalloc's in-place
// behaviour and accounts only the size delta. Not compiled into shipping builds.
#[cfg(feature = "heap-prof")]
mod heap_prof {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicI64, Ordering};

    pub(super) static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
    pub(super) static LIVE_COUNT: AtomicI64 = AtomicI64::new(0);

    // Net-live allocation count bucketed by size class: bucket i = sizes in
    // `(8*(i-1), 8*i]`, i.e. `(size+7)/8` clamped to 63. Bucket 63 aggregates
    // everything ≥ 504 bytes. Lets the host see exactly which size leaks.
    pub(super) const NBUCKETS: usize = 64;
    pub(super) static BUCKETS: [AtomicI64; NBUCKETS] = {
        const Z: AtomicI64 = AtomicI64::new(0);
        [Z; NBUCKETS]
    };

    #[inline]
    fn bucket(size: usize) -> usize {
        (size.div_ceil(8)).min(NBUCKETS - 1)
    }

    pub(super) struct CountingAlloc;

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(layout) };
            if !p.is_null() {
                LIVE_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed);
                LIVE_COUNT.fetch_add(1, Ordering::Relaxed);
                BUCKETS[bucket(layout.size())].fetch_add(1, Ordering::Relaxed);
            }
            p
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let p = unsafe { System.alloc_zeroed(layout) };
            if !p.is_null() {
                LIVE_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed);
                LIVE_COUNT.fetch_add(1, Ordering::Relaxed);
                BUCKETS[bucket(layout.size())].fetch_add(1, Ordering::Relaxed);
            }
            p
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) };
            LIVE_BYTES.fetch_sub(layout.size() as i64, Ordering::Relaxed);
            LIVE_COUNT.fetch_sub(1, Ordering::Relaxed);
            BUCKETS[bucket(layout.size())].fetch_sub(1, Ordering::Relaxed);
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let p = unsafe { System.realloc(ptr, layout, new_size) };
            if !p.is_null() {
                LIVE_BYTES.fetch_add(new_size as i64 - layout.size() as i64, Ordering::Relaxed);
                BUCKETS[bucket(layout.size())].fetch_sub(1, Ordering::Relaxed);
                BUCKETS[bucket(new_size)].fetch_add(1, Ordering::Relaxed);
            }
            p
        }
    }
}

#[cfg(feature = "heap-prof")]
#[global_allocator]
static HEAP_PROF_ALLOC: heap_prof::CountingAlloc = heap_prof::CountingAlloc;

// Residual-call host trampoline for the native-host (`wasm-host`) build.
//
// wasm32 `call_indirect` type-checks every call, so the in-module metainterp
// cannot transmute a raw funcptr to a statically-guessed `extern "C" fn` and
// call it — a residual target whose real signature is not the uniform
// `(i64…) -> i64` traps. The compiled trace already round-trips such calls
// through the host (`env.jit_call`); this routes the recording / blackhole
// path through the symmetric `pyre_jit.jit_call_host` import, which reflects
// the callee's wasm signature and coerces each positional argument.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
mod residual_host {
    use core::cell::UnsafeCell;

    // Call-area layout shared with `majit-backend-wasm` codegen and the host
    // runner's `jit_call_trampoline`; offsets are relative to the frame-pointer
    // base passed to the import.
    const CALL_RESULT_OFS: usize = 2000;
    const CALL_FUNC_OFS: usize = 2008;
    const CALL_NARGS_OFS: usize = 2016;
    const CALL_ARGS_OFS: usize = 2024;
    const MAX_ARGS: usize = 16;
    const SCRATCH_LEN: usize = CALL_ARGS_OFS + MAX_ARGS * 8;

    #[link(wasm_import_module = "pyre_jit")]
    unsafe extern "C" {
        fn jit_call_host(frame_ptr: u32);
    }

    // A wasm32 module instance is single-threaded, so a shared scratch buffer
    // needs no synchronization. Residual calls nest synchronously: each level
    // writes its arguments, the host reads them before invoking the callee, and
    // each level reads its result immediately after the host returns — so an
    // inner call that reuses the buffer cannot clobber an outer call's
    // already-consumed arguments or not-yet-written result.
    struct Scratch(UnsafeCell<[u8; SCRATCH_LEN]>);
    unsafe impl Sync for Scratch {}
    static SCRATCH: Scratch = Scratch(UnsafeCell::new([0u8; SCRATCH_LEN]));

    fn residual_host_call(func_ptr: usize, args: &[i64]) -> i64 {
        assert!(
            args.len() <= MAX_ARGS,
            "residual_host_call: arity {} exceeds {MAX_ARGS}",
            args.len()
        );
        let base = SCRATCH.0.get() as *mut u8;
        unsafe {
            (base.add(CALL_FUNC_OFS) as *mut i64).write_unaligned(func_ptr as i64);
            (base.add(CALL_NARGS_OFS) as *mut i64).write_unaligned(args.len() as i64);
            for (i, &a) in args.iter().enumerate() {
                (base.add(CALL_ARGS_OFS + i * 8) as *mut i64).write_unaligned(a);
            }
            jit_call_host(base as u32);
            (base.add(CALL_RESULT_OFS) as *const i64).read_unaligned()
        }
    }

    /// Install the trampoline on the current thread. Idempotent.
    pub fn install() {
        majit_backend::call_stub::set_residual_host_call(Some(residual_host_call));
    }
}

// Host-filesystem source provider for the native-host (`wasm-host`) build.
//
// wasm32 has no filesystem, but the wasmtime runner does, so module source is
// read through `pyre_host.*` host imports the runner satisfies. The runner
// reports the real stdlib root (the `$PYRE_STDLIB` directory `pyre/check.py`
// forwards); seeding it on `sys.path` lets the SAME import machinery that runs
// on native resolve `import re` against genuine host paths. The browser/web
// build has no such host, so it embeds the stdlib instead (`wasm_vfs`).
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
mod host_fs_provider {
    use pyre_interpreter::importing::SourceProvider;
    use std::path::Path;

    #[link(wasm_import_module = "pyre_host")]
    unsafe extern "C" {
        /// Write the real stdlib root path into `[buf, buf+cap)`; return its
        /// byte length (without writing if it exceeds `cap`), or -1 if unset.
        fn host_stdlib_root(buf_ptr: *mut u8, buf_cap: u32) -> i64;
        /// 1 if `[path, path+len)` names a directory, else 0.
        fn host_is_dir(path_ptr: *const u8, path_len: u32) -> u32;
        /// Byte length of the regular file at `path`, or -1 if not a file.
        fn host_file_size(path_ptr: *const u8, path_len: u32) -> i64;
        /// Read the file at `path` into `[buf, buf+cap)`; return bytes written
        /// (clamped to `cap`), or -1 on error.
        fn host_read(path_ptr: *const u8, path_len: u32, buf_ptr: *mut u8, buf_cap: u32) -> i64;
    }

    struct HostFsProvider;

    impl SourceProvider for HostFsProvider {
        fn is_file(&self, path: &Path) -> bool {
            let p = path.to_string_lossy();
            unsafe { host_file_size(p.as_ptr(), p.len() as u32) >= 0 }
        }
        fn is_dir(&self, path: &Path) -> bool {
            let p = path.to_string_lossy();
            unsafe { host_is_dir(p.as_ptr(), p.len() as u32) != 0 }
        }
        fn read_to_bytes(&self, path: &Path) -> std::io::Result<Vec<u8>> {
            let p = path.to_string_lossy();
            let size = unsafe { host_file_size(p.as_ptr(), p.len() as u32) };
            if size < 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("{}", path.display()),
                ));
            }
            let mut buf = vec![0u8; size as usize];
            let n = unsafe {
                host_read(
                    p.as_ptr(),
                    p.len() as u32,
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                )
            };
            if n < 0 {
                return Err(std::io::Error::other(format!(
                    "host_read failed: {}",
                    path.display()
                )));
            }
            buf.truncate(n as usize);
            Ok(buf)
        }
    }

    /// Query the host for the stdlib root, seed it on `sys.path`, and install
    /// the host-FS source provider.  Called once before the first import.
    pub fn install() {
        let mut buf = vec![0u8; 4096];
        let n = unsafe { host_stdlib_root(buf.as_mut_ptr(), buf.len() as u32) };
        if n > 0 && (n as usize) <= buf.len() {
            if let Ok(root) = std::str::from_utf8(&buf[..n as usize]) {
                pyre_interpreter::importing::add_sys_path(Path::new(root));
            }
        }
        pyre_interpreter::importing::install_source_provider(std::rc::Rc::new(HostFsProvider));
    }
}

/// Diagnostic (`heap-prof`): net-live guest-heap bytes (alloc − dealloc).
/// Linear growth here under steady re-entry is a true not-freed leak.
#[cfg(feature = "heap-prof")]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_heap_live_bytes() -> i64 {
    heap_prof::LIVE_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Diagnostic (`heap-prof`): net-live guest-heap allocation count.
/// `live_bytes / live_count` is the average leaked-object size.
#[cfg(feature = "heap-prof")]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_heap_live_count() -> i64 {
    heap_prof::LIVE_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Diagnostic (`heap-prof`): net-live allocation count in size-class bucket
/// `i` (sizes in `(8*(i-1), 8*i]`; bucket 63 = ≥504 B). Out-of-range → 0.
#[cfg(feature = "heap-prof")]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_heap_bucket(i: u32) -> i64 {
    let i = i as usize;
    if i < heap_prof::NBUCKETS {
        heap_prof::BUCKETS[i].load(std::sync::atomic::Ordering::Relaxed)
    } else {
        0
    }
}

/// Diagnostic-only: read a `compile_bridge` outcome tally from the wasm JIT
/// backend (index legend in `majit_backend_wasm::BRIDGE_DIAG`). Exported (not
/// an import) so it does not shift the module's function-index space, which
/// would break the JIT's baked `fn as usize` table indices. The host runner
/// prints these at `PYRE_WASM_JIT_STATS` time.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_bridge_diag(i: u32) -> u64 {
    majit_backend_wasm::bridge_diag(i as usize)
}

/// Test-only control plane for the terminal-declined CALL_ASSEMBLER regression.
/// Exported rather than imported so it cannot perturb table/function indices
/// used by JIT-emitted modules. Zero disables it; see
/// `majit_backend_wasm::set_force_ca_terminal_decline` for selector semantics.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_set_force_ca_terminal_decline(selector: u64) {
    majit_backend_wasm::set_force_ca_terminal_decline(selector);
}

/// Diagnostic-only: number of JIT trace entries made from the guest.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_execute_count() -> u64 {
    majit_backend_wasm::jit_execute_count()
}

/// Diagnostic-only: modules that crossed the lazy materialization gate.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_compile_count() -> u64 {
    majit_backend_wasm::jit_compile_count()
}

/// Diagnostic-only: byte-identical module cache hits.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_compile_cache_hits() -> u64 {
    majit_backend_wasm::jit_compile_cache_hits()
}

/// Diagnostic-only: read a guard-failure → bridge-trace gate tally from the
/// metainterp (`majit_metainterp::MC_DIAG`). Same export-not-import rationale
/// as `pyre_jit_bridge_diag`.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_mc_diag(i: u32) -> u64 {
    majit_metainterp::mc_diag(i as usize)
}

/// Diagnostic-only: the full-body-walk decline census
/// (`pyre_jit_trace::jitcode_dispatch::census_entries`), which names the
/// `DispatchError` variant behind every aborted walk. The map is always
/// populated; only its `[fbw-census]` print is gated on
/// `PYRE_FBW_DEBUG_ABORT`, which a wasm guest can neither read nor print, so
/// these three exports are the only channel that carries the attribution out.
///
/// The names are `&'static str` in guest memory: `..._name` returns
/// `(ptr << 32) | len` for the host to slice out of the exported memory, and
/// `..._count` returns that entry's tally. Both index the same sorted order as
/// `census_entries`, which is stable for a given call — read the length first
/// and do not interleave with anything that could record a new decline.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_fbw_census_len() -> u32 {
    pyre_jit_trace::jitcode_dispatch::census_entries().len() as u32
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_fbw_census_name(i: u32) -> u64 {
    match pyre_jit_trace::jitcode_dispatch::census_entries().get(i as usize) {
        Some((name, _)) => ((name.as_ptr() as u64) << 32) | name.len() as u64,
        None => 0,
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_fbw_census_count(i: u32) -> u64 {
    pyre_jit_trace::jitcode_dispatch::census_entries()
        .get(i as usize)
        .map_or(0, |&(_, count)| count as u64)
}

/// Sign-stable JIT counters that read 0 in a healthy run, the same badness
/// fields the native `[jit-stats]` line reports (`pyrex` maybe_print_jit_stats).
/// A nonzero value means a trace aborted or an internal compile bug degraded
/// the trace, never a tuning choice — so `check.py`'s regression floor can gate
/// wasm on them exactly as it gates the native backends. Exported (not imported)
/// so reading them cannot perturb the JIT's table/function indices.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_loops_aborted() -> u64 {
    pyre_jit::eval::driver_pair().0.get_stats().loops_aborted as u64
}

/// The `Counters.ABORT_*` split behind `pyre_jit_loops_aborted`, indexed as
/// `majit_metainterp::jitprof::ABORT_COUNTER_KINDS`. On the native backends
/// that breakdown comes from `MAJIT_LOG`'s summary, but the wasm32 guest has
/// no environment, so a `loops_aborted` regression that only reproduces here
/// otherwise carries no reason at all. Exported rather than imported for the
/// same reason as `pyre_jit_bridge_diag`. The host prints it at
/// `PYRE_WASM_JIT_STATS` time.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_abort_diag(i: u32) -> u64 {
    pyre_jit::eval::driver_pair().0.abort_diag(i as usize)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_internal_compile_panics() -> u64 {
    pyre_jit::eval::driver_pair()
        .0
        .get_stats()
        .internal_compile_panics as u64
}

/// The rest of the native `[jit-stats]` line. These are not sign-stable, so
/// each is gated on its own terms rather than against zero: `loops_compiled`
/// fails on any FALL, `guard_failures` on a rise past a band, and
/// `bridges_compiled` on neither direction, since it moves both ways under
/// ordinary tuning.
///
/// They come out of the same `JitStats` the fields above read, and were simply
/// never exported. Their absence was not neutral: a count-valued field missing
/// from a recorded baseline is skipped silently, so a change that stopped
/// compiling a loop altogether aborted nothing and *lowered* `guard_failures` —
/// both remaining gates read the loss as an improvement.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_loops_compiled() -> u64 {
    pyre_jit::eval::driver_pair().0.get_stats().loops_compiled as u64
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_bridges_compiled() -> u64 {
    pyre_jit::eval::driver_pair().0.get_stats().bridges_compiled as u64
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_guard_failures() -> u64 {
    pyre_jit::eval::driver_pair().0.get_stats().guard_failures as u64
}

/// The descr-universe invariants, the remaining `JITSTATS_BADNESS_FIELDS`. The
/// native backends print these from `descr_set_jit_stats`; the guest has no
/// stderr, so it exports the counts and the runner prints the line. Without
/// these the regression floor read every one as 0 on wasm — a missing field is
/// read as absent-and-therefore-zero — so a wasm-only rise passed the gate.
///
/// `stale_absent` re-asks the `AbsentContainer` question against the universe as
/// it now stands, so it is only meaningful once the run has published
/// everything; the runner calls these after the guest's entry point returns.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_descr_set_absent() -> u64 {
    pyre_jit::descr_set_counts().absent
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_descr_set_ambiguous() -> u64 {
    pyre_jit::descr_set_counts().ambiguous
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_descr_set_stale_absent() -> u64 {
    pyre_jit::descr_set_counts().stale_absent
}

/// The denominator, so a wasm run that resolves nothing cannot read the same as
/// one that resolves everything. Host-dependent, hence excluded from the
/// committed snapshot (`JITSTATS_SNAPSHOT_FIELDS`) — it is reported for
/// diagnosis, not gated.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_descr_set_resolved() -> u64 {
    pyre_jit::descr_set_counts().resolved
}

#[cfg(any(feature = "web", feature = "wasm-host"))]
static PANIC_HOOK: Once = Once::new();

#[cfg(any(feature = "web", feature = "wasm-host"))]
fn install_panic_hook() {
    PANIC_HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            // The module is built with the default `panic=abort`, so a panic
            // ends the run. Record the formatted message in linear memory
            // before the abort; the runner's `recover_panic_messages` scans for
            // it (the browser glue surfaces it the same way) so the real cause
            // is visible rather than a bare trap.
            let msg = format!("[pyre panic] {info}");
            OUTPUT_BUF.with(|buf| buf.borrow_mut().push_str(&msg));
        }));
    });
}

#[cfg(any(feature = "web", feature = "wasm-host"))]
thread_local! {
    static OUTPUT_BUF: RefCell<String> = RefCell::new(String::new());
    /// fd-2 capture. wasm32 has no descriptors, so everything the interpreter
    /// writes to stderr — tracebacks, warnings, `sys.stderr.write` — is
    /// collected here and handed to the embedder separately from stdout.
    static ERR_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// Process status the run ends with: 0, or `SystemExit`'s code, or 1 for
    /// an uncaught exception. `targetpypystandalone.py:37 entry_point` returns
    /// the same value; the embedder exits with it.
    static EXIT_CODE: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    /// Path the source came from, if the embedder named one. The guest is
    /// handed source text, not a file, so without it every code object
    /// compiles as `<string>` and a traceback can name neither the file nor
    /// the offending line.
    static SCRIPT_PATH: RefCell<Option<String>> = const { RefCell::new(None) };
}

#[cfg(any(feature = "web", feature = "wasm-host"))]
fn install_wasm_print_hook() {
    pyre_interpreter::set_print_hook(|b| {
        // The hook is handed raw fd-1 bytes; this embedder returns a `String`
        // to its host, so the decode happens here rather than inside the
        // interpreter, where it would corrupt every embedder's output.
        OUTPUT_BUF.with(|buf| buf.borrow_mut().push_str(&String::from_utf8_lossy(b)));
    });
    pyre_interpreter::set_stderr_hook(|b| {
        ERR_BUF.with(|buf| buf.borrow_mut().extend_from_slice(b));
    });
}

/// Run a Python source string and return the output as a string.
///
/// Host-agnostic core shared by the `web` (wasm-bindgen) and `wasm-host`
/// (C-ABI) entry points below.
#[cfg(any(feature = "web", feature = "wasm-host"))]
fn run_python_impl(source: &str) -> String {
    install_panic_hook();
    #[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
    residual_host::install();
    // Eagerly install pyre-jit's hooks (pyrex real_main does the same at
    // boot): the dict `eq_w` / `hash_w` / `hash_str` /
    // `compares_by_identity` trampolines must be live before
    // `install_builtin_modules` / `import` builds the first str- or
    // object-keyed dict, not only after the first JIT-traced bytecode
    // (`dict_eq_hook::missing_hash_hook` fails fast otherwise).
    pyre_jit::eval::init_jit_hooks();
    pyre_interpreter::importing::install_builtin_modules();
    // Give the import machinery a source of module bytes. The browser has no
    // filesystem, so the web build serves the embedded stdlib closure from an
    // in-memory VFS; the native-host (`wasm-host`) build reads the host filesystem
    // through `pyre_host.*` imports the runner satisfies.
    #[cfg(feature = "web")]
    pyre_interpreter::importing::mount_embedded_stdlib(std::path::Path::new("/lib-python/3"));
    #[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
    {
        // `pymain_sys_path_add_path0`: the script's directory heads
        // `sys.path`, ahead of the stdlib root `install` appends, so a module
        // beside the script shadows one of the same name in the stdlib.
        if let Some(dir) = SCRIPT_PATH
            .with(|p| p.borrow().clone())
            .and_then(|p| std::path::Path::new(&p).parent().map(|d| d.to_path_buf()))
        {
            pyre_interpreter::importing::add_sys_path(&dir);
        }
        host_fs_provider::install();
    }
    install_wasm_print_hook();
    OUTPUT_BUF.with(|buf| buf.borrow_mut().clear());
    ERR_BUF.with(|buf| buf.borrow_mut().clear());
    EXIT_CODE.with(|c| c.set(0));

    let filename = SCRIPT_PATH.with(|p| p.borrow().clone());
    let filename = filename.as_deref().unwrap_or("<string>");
    let code = match compile_source_with_filename(source, Mode::Exec, filename) {
        Ok(code) => code,
        Err(e) => {
            // `pyrex::run_source` renders the same `File "…", line N` + caret
            // banner on stderr and exits 1.
            pyre_interpreter::eprint_syntax_error(&pyre_interpreter::compile_err_to_syntax_error(
                e, source,
            ));
            EXIT_CODE.with(|c| c.set(1));
            return String::new();
        }
    };

    let execution_context = std::rc::Rc::new(PyExecutionContext::default());
    // Seed the TLS execution-context slot (pyrex real_main does the same at
    // boot). `getexecutioncontext().gettopframe()` must be live so a residual
    // `bh_call_fn_impl` from a blackhole resume — e.g. a `print(...)` after a
    // JIT-compiled loop — can resolve its parent frame instead of tripping the
    // fail-fast topframe assert.
    pyre_interpreter::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&execution_context));
    // objspace.py `space.user_del_action`: the queue every registered
    // finalizer's trigger fires into (`finalizer_queue_trigger` reads
    // `ec.user_del_action`).  Without it the pointer stays null and the trigger
    // returns silently, so NOTHING is ever finalized on this entry point — no
    // `__del__`, no generator `finally`, not even under an explicit
    // `gc.collect()`.  `pyrex::setup_exec_context` installs it at boot; this
    // entry point is the other launcher and needs the same step.
    unsafe {
        let ec_ptr = std::rc::Rc::as_ptr(&execution_context) as *mut PyExecutionContext;
        (*ec_ptr).install_user_del_action();
    }
    // Register the __build_class__ callback. Class construction resolves the
    // live frame from the execution-context slot seeded above.
    pyre_interpreter::call::register_build_class();
    let mut frame =
        match pyre_interpreter::pyframe::PyFrame::new_with_context(code, execution_context) {
            Ok(frame) => frame,
            Err(e) => return format!("Error: {e}"),
        };
    // Nothing reaches a GC frame but the `CURRENT_FRAME` / `f_backref` chain,
    // which this one joins only when `eval_with_jit` installs it; the module
    // registration below allocates and can collect in between. Publish it for
    // that span (pyrex `run_source` does the same).
    let _main_frame_root = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(
        &*frame as *const pyre_interpreter::pyframe::PyFrame as pyre_object::PyObjectRef,
    );

    // Register the `__main__` module in sys.modules (pyrex real_main does the
    // same), reusing the canonical globals dict so `__main__.__dict__`,
    // `globals()`, and `function.__globals__` share one identity. Without this,
    // `sys.modules['__main__']` / `import __main__` raise KeyError.
    let canonical = frame.get_w_globals();
    let main_module = pyre_object::module::w_module_new_aliasing_dict("__main__", canonical);
    pyre_interpreter::importing::set_sys_module("__main__", main_module);

    // catch_unwind to capture panics from JIT as error messages
    let eval_result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pyre_jit::eval::eval_with_jit(&mut frame)
    })) {
        Ok(r) => r,
        Err(_) => {
            EXIT_CODE.with(|c| c.set(1));
            let panic_msg = OUTPUT_BUF.with(|buf| buf.borrow().clone());
            return if panic_msg.is_empty() {
                "[pyre] unknown panic".to_string()
            } else {
                panic_msg
            };
        }
    };

    let mut output = OUTPUT_BUF.with(|buf| buf.borrow().clone());

    match eval_result {
        Ok(result) => {
            if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                if !output.is_empty() && !output.ends_with('\n') {
                    output.push('\n');
                }
                output.push_str(&format!("{}", PyDisplay(result)));
            }
        }
        Err(e) => {
            // `pyrex::real_main`: a `SystemExit` sets the status and prints
            // nothing; anything else prints its traceback and exits 1.  Both
            // go to stderr, so the run's stdout stays byte-comparable with the
            // native binaries instead of gaining an `Error: …` tail.
            if e.kind == pyre_interpreter::PyErrorKind::SystemExit {
                EXIT_CODE.with(|c| c.set(pyre_interpreter::system_exit_code(&e)));
            } else {
                pyre_interpreter::eprint_exception(&e, true);
                EXIT_CODE.with(|c| c.set(1));
            }
        }
    }

    output
}

/// Bytes the run wrote to fd 2, draining the buffer.
#[cfg(any(feature = "web", feature = "wasm-host"))]
fn take_stderr() -> Vec<u8> {
    ERR_BUF.with(|buf| std::mem::take(&mut *buf.borrow_mut()))
}

/// Browser / JS entry point: marshalled by wasm-bindgen.
///
/// The browser has one output channel, so the fd-2 text (traceback, warnings)
/// is appended to the returned string rather than handed over separately the
/// way the C-ABI surface does it.
#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn run_python(source: &str) -> String {
    let mut output = run_python_impl(source);
    let err = take_stderr();
    if !err.is_empty() {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&String::from_utf8_lossy(&err));
    }
    output
}

/// Native-host (`wasm-host` feature) C-ABI surface.
///
/// wasm-bindgen is unavailable without a JS runtime, so the embedder talks
/// to the module through plain exports over linear memory:
///   1. `pyre_alloc(len)` → reserve `len` bytes, write the UTF-8 source there;
///   2. `pyre_run_python(ptr, len)` → run it, returns a packed `u64`
///      (`hi32` = result pointer, `lo32` = result byte length);
///   3. read the UTF-8 result, then `pyre_dealloc(ptr, len)` both buffers;
///   4. `pyre_take_stderr()` → the same packed pair for the run's fd-2 bytes,
///      and `pyre_exit_code()` → the status to exit with.
///
/// `pyre_set_script_path(ptr, len)` may precede step 2 to name the file the
/// source came from.
#[cfg(feature = "wasm-host")]
mod host_abi {
    use super::run_python_impl;
    use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};

    // Buffers crossing the boundary are allocated and freed through the
    // global allocator with a `Layout::array::<u8>(len)` derived purely
    // from `len`, so the host only ever needs to remember the length to
    // free a buffer soundly.

    /// Reserve `len` bytes in linear memory and return a pointer the host
    /// can write into. Pair every call with `pyre_dealloc`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_alloc(len: usize) -> *mut u8 {
        if len == 0 {
            return std::ptr::NonNull::<u8>::dangling().as_ptr();
        }
        // Layout::array can only fail on overflow, impossible for a real
        // wasm linear-memory size.
        let layout = Layout::array::<u8>(len).expect("pyre_alloc: size overflow");
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        ptr
    }

    /// Release a buffer previously handed out by `pyre_alloc` or returned
    /// by `pyre_run_python`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_dealloc(ptr: *mut u8, len: usize) {
        if ptr.is_null() || len == 0 {
            return;
        }
        let layout = Layout::array::<u8>(len).expect("pyre_dealloc: size overflow");
        unsafe { dealloc(ptr, layout) }
    }

    /// Diagnostic: total bytes the GC holds in the old generation (promoted +
    /// raw/large old-gen objects), or 0 if no GC is installed. Read by the host
    /// runner after a run to split GC-retained memory from host-heap growth.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_gc_oldgen_bytes() -> u64 {
        pyre_jit::wasm_gc_heap_stats().0 as u64
    }

    /// Diagnostic: bytes currently filled in the GC nursery, or 0 if no GC is
    /// installed. Companion to [`pyre_gc_oldgen_bytes`].
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_gc_nursery_bytes() -> u64 {
        pyre_jit::wasm_gc_heap_stats().1 as u64
    }

    /// Diagnostic: minor collections run so far, or 0 if no GC is installed.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_gc_minor_collections() -> u64 {
        pyre_jit::wasm_gc_collection_counts().0 as u64
    }

    /// Diagnostic: major collections run so far, or 0 if no GC is installed.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_gc_major_collections() -> u64 {
        pyre_jit::wasm_gc_collection_counts().1 as u64
    }

    /// Run the UTF-8 Python source at `ptr[..len]`. Returns a packed
    /// `(result_ptr << 32) | result_len`; the result is a UTF-8 byte buffer
    /// the host must free with `pyre_dealloc`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_run_python(ptr: *const u8, len: usize) -> u64 {
        let result = match guest_str(ptr, len) {
            Some(source) => run_python_impl(&source),
            None => "Error: input buffer out of wasm memory bounds".to_string(),
        };

        pack_into_guest(result.into_bytes())
    }

    /// Name the file the next `pyre_run_python` source came from. It becomes
    /// the compiled code's filename — hence what a traceback prints and what
    /// its source-line lookup reads — and its directory heads `sys.path`.
    /// Without it the guest only has source text, so both fall back to
    /// `<string>`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_set_script_path(ptr: *const u8, len: usize) {
        let path = guest_str(ptr, len).filter(|s| !s.is_empty());
        super::SCRIPT_PATH.with(|p| *p.borrow_mut() = path);
    }

    /// Copy `[ptr, ptr+len)` out of linear memory as UTF-8. The embedder
    /// supplies the pair raw, so a range escaping linear memory is rejected
    /// (`None`) before a slice is formed rather than being undefined
    /// behaviour; an empty buffer reads as the empty string.
    fn guest_str(ptr: *const u8, len: usize) -> Option<String> {
        if ptr.is_null() || len == 0 {
            return Some(String::new());
        }
        let mem_bytes = core::arch::wasm32::memory_size(0).saturating_mul(65536);
        match (ptr as usize).checked_add(len) {
            Some(end) if end <= mem_bytes => {
                let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
                Some(String::from_utf8_lossy(bytes).into_owned())
            }
            _ => None,
        }
    }

    /// Take the bytes the run wrote to fd 2 (traceback, warnings,
    /// `sys.stderr.write`) and hand them over the same way as the stdout
    /// result: a packed `(ptr << 32) | len` the host frees with
    /// `pyre_dealloc`. Draining, so a second call returns an empty buffer.
    ///
    /// wasm32 has no descriptors, so without this export everything the
    /// interpreter writes to stderr is discarded.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_take_stderr() -> u64 {
        pack_into_guest(super::take_stderr())
    }

    /// The walker's decline census (`[fbw-census] <name>: <count>` lines) as
    /// the same packed pair, for the host to print under
    /// `PYRE_WASM_FBW_CENSUS`.
    ///
    /// `PYRE_FBW_DEBUG_ABORT` selects that dump on a native build, but the
    /// guest reads no environment and writes to no descriptor, so neither the
    /// switch nor the `eprintln!` exists here. The counters are accumulated
    /// either way; this is the only way to read them off a wasm run. Reading,
    /// not draining, unlike `pyre_take_stderr`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_fbw_census() -> u64 {
        pack_into_guest(pyre_jit_trace::jitcode_dispatch::census_report().into_bytes())
    }

    /// Status the last `pyre_run_python` ended with: `SystemExit`'s code, 1 for
    /// an uncaught exception or a `SyntaxError`, else 0. The host exits with
    /// it, as `pyrex` does with `targetpypystandalone.py:37 entry_point`'s
    /// return value.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_exit_code() -> i32 {
        super::EXIT_CODE.with(|c| c.get())
    }

    /// Copy `bytes` into a fresh guest buffer and pack its pointer and length
    /// into the `(hi32, lo32)` pair every result-returning export uses.
    fn pack_into_guest(bytes: Vec<u8>) -> u64 {
        let len = bytes.len();
        let ptr = pyre_alloc(len);
        if len != 0 {
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len) };
        }
        ((ptr as u64) << 32) | (len as u64)
    }
}

/// Diagnostic-only: read one slot of the pyre walker's per-walk census
/// (slot layout in `pyre_jit_trace::trace::fbw_diag`). The guest cannot read
/// `PYRE_FBW_CENSUS`, so this is the only walk-level observability the wasm
/// target has. Exported (not an import) for the same function-index reason as
/// `pyre_jit_bridge_diag`. The host runner prints these at
/// `PYRE_WASM_JIT_STATS` time.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_fbw_diag(i: u32) -> u64 {
    pyre_jit_trace::trace::fbw_diag::get(i as usize)
}
