/// MiniMarkGC — the core collector implementing the GcAllocator trait.
///
/// A generational copying collector with:
/// - Bump-pointer nursery for young objects
/// - ArenaCollection old gen plus rawmalloc fallback, with incremental major sweep
/// - Write barrier with remembered set for old-to-young pointers
///
/// Modeled after incminimark's minor/major collection.
use indexmap::IndexSet;
use majit_ir::GcRef;
use std::collections::VecDeque;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};
/// Monotonic source for the durations the collector reports to its hooks.
///
/// `wasm32-unknown-unknown` has no clock: `Instant::now` panics with "time not
/// implemented on this platform", which is also why the interpreter does not
/// install its `time` module on that target. Report a zero duration there
/// rather than taking the panic — `duration`, `duration_min`, and
/// `duration_max` only reach `hook.py`'s stats objects and the `total_gc_time`
/// counter, and no collection decision reads any of them.
struct GcClock {
    #[cfg(not(target_arch = "wasm32"))]
    start: std::time::Instant,
}

impl GcClock {
    fn start() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            start: std::time::Instant::now(),
        }
    }

    fn elapsed_secs(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start.elapsed().as_secs_f64()
        }
        #[cfg(target_arch = "wasm32")]
        {
            0.0
        }
    }
}

use crate::address_dict::AddressMap;
use crate::flags;
use crate::header::{GcHeader, header_of};
use crate::hook::GcHooks;
use crate::nursery::{DEFAULT_NURSERY_SIZE, Nursery};
use crate::oldgen::OldGen;
use crate::rawrefcount::{self, RrcList};
use crate::trace::{ClassTypeLayout, TypeEntry, TypeInfo, TypeInfoLayout, TypeRegistry};
use crate::{FinalizerTriggerFn, GcAllocator};

/// `inspector.py:209 HeapDumper.BUFSIZE` raw signed-word writer. Keeping the
/// fixed-size buffer here avoids materializing the full heap dump in memory.
struct HeapDumpWriter {
    fd: i32,
    buffer: Vec<isize>,
}

// Some targets and host-seam failures provide no OS errno. Use the POSIX EIO
// value for those cases so every heap-dump failure still carries an error code.
pub const HEAP_DUMP_EIO: i32 = 5;

impl HeapDumpWriter {
    const BUFSIZE: usize = 8192;

    fn new(fd: i32) -> Self {
        Self {
            fd,
            buffer: Vec::with_capacity(Self::BUFSIZE),
        }
    }

    fn write(&mut self, value: isize) -> Result<(), i32> {
        self.buffer.push(value);
        if self.buffer.len() == Self::BUFSIZE {
            self.flush()?;
        }
        Ok(())
    }

    fn write_marker(&mut self) -> Result<(), i32> {
        for value in [0, 0, 0, -1] {
            self.write(value)?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), i32> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let byte_len = self.buffer.len() * std::mem::size_of::<isize>();
        // SAFETY: the initialized `isize` elements occupy exactly `byte_len`
        // bytes and remain borrowed for the duration of the write.
        let bytes =
            unsafe { std::slice::from_raw_parts(self.buffer.as_ptr().cast::<u8>(), byte_len) };
        let write_result = if let Some(result) = crate::try_heap_dump_write(self.fd, bytes) {
            result
        } else {
            // Neither `write` nor `_write` exists here, so no call was made and
            // `errno` still names some unrelated earlier one. Report the dump's own
            // failure code instead of reading a stale `errno`.
            #[cfg(not(any(unix, windows)))]
            {
                Err(HEAP_DUMP_EIO)
            }
            #[cfg(any(unix, windows))]
            {
                #[cfg(unix)]
                let written: isize = unsafe {
                    libc::write(
                        self.fd,
                        self.buffer.as_ptr().cast::<libc::c_void>(),
                        byte_len,
                    )
                };
                // The CRT entry point is `_write`, but `libc` exports it under the
                // POSIX name with a `#[link_name = "_write"]` alias, so the Rust
                // path is `libc::write` on this target as well. It takes a
                // `c_uint` count and returns `c_int`, unlike the `size_t`/`ssize_t`
                // unix signature above.
                #[cfg(windows)]
                let written: isize = unsafe {
                    libc::write(
                        self.fd,
                        self.buffer.as_ptr().cast::<libc::c_void>(),
                        byte_len as libc::c_uint,
                    ) as isize
                };
                if written < 0 {
                    Err(std::io::Error::last_os_error()
                        .raw_os_error()
                        .unwrap_or(HEAP_DUMP_EIO))
                } else {
                    Ok(written)
                }
            }
        };
        let written = write_result?;
        if written as usize != byte_len {
            return Err(HEAP_DUMP_EIO);
        }
        self.buffer.clear();
        Ok(())
    }
}

/// Host predicate answering whether the consumer of a deferred major request
/// would act on one right now. Unset = never arm.
///
/// A standing switch is not enough, because the consumer — the interpreter
/// dispatch-loop safepoint — refuses in two different ways. It refuses for the
/// whole process when the interpreter GC is off, and it refuses per-thread,
/// moment to moment, whenever the eval loop is nested too deep for the frame
/// walker to see the full root set. Arming through the second kind of refusal
/// is not merely wasted: `threshold_reached` stays true until a major
/// completes, so every subsequent born-old allocation re-arms, the compiled
/// back-edge poll fails continuously, and the guard accumulates bridges. Once
/// a bridge is attached at each level the chain closes on itself and compiled
/// code stops returning to the dispatch loop at all — which starves the rest of
/// the word, `EB_ASYNC` included, so a signal goes undelivered for as long as
/// the loop runs. Measured on a depth-3 loop: SIGINT delivery went from 0.11 s
/// to over 30 s.
///
/// So ask the consumer. The probe is called only after `threshold_reached`, and
/// only on the born-old path, so it costs nothing on the ordinary nursery
/// allocation. Left unset the threshold is answered the way it always was — by
/// the major progress every minor collection drives.
pub type DeferredMajorRequestProbeFn = fn() -> bool;

crate::global_hook!(static DEFERRED_MAJOR_REQUEST_PROBE: DeferredMajorRequestProbeFn);

/// Install the predicate gating born-old major requests, or `None` to stop
/// arming them. See [`DEFERRED_MAJOR_REQUEST_PROBE`].
pub fn set_deferred_major_request_probe(probe: Option<DeferredMajorRequestProbeFn>) {
    DEFERRED_MAJOR_REQUEST_PROBE.set(probe);
}

/// Whether a request armed now would be acted on. False with no probe installed.
fn deferred_major_request_wanted() -> bool {
    DEFERRED_MAJOR_REQUEST_PROBE
        .get()
        .is_some_and(|probe| probe())
}

/// Consume a pending request, reporting whether one was armed.
///
/// Clear it whether or not the caller goes on to collect: the bit is the
/// compiled back edge's deopt trigger, so one the caller keeps fires again on
/// the next back edge, and the next.
pub fn take_deferred_major_request() -> bool {
    majit_ir::eval_breaker_word::take_gc()
}

/// Arm a major collection request for the next root-complete interpreter
/// dispatch.  Hosts use this when collection is required for semantics rather
/// than because an allocation crossed the collector's threshold.
pub fn request_deferred_major_collection() {
    majit_ir::eval_breaker_word::set_gc();
}

/// Configuration for the MiniMarkGC.
pub struct GcConfig {
    /// Nursery size in bytes.
    pub nursery_size: usize,
    /// incminimark.py:326/472 `debug_tiny_nursery` — the bytes
    /// `collect_and_reserve` leaves free after each reservation, so a minor
    /// collection runs roughly every `debug_tiny_nursery` bytes of allocation.
    /// `None` is upstream's `-1` (off).  Set only by the `PYPY_GC_NURSERY`
    /// clamp; an explicitly configured `nursery_size` is left alone.
    pub debug_tiny_nursery: Option<usize>,
    /// Maximum object size that can be allocated in the nursery.
    /// Larger objects go directly to old gen.
    pub large_object_threshold: usize,
    /// incminimark.py:275: card_page_indices (0 disables card marking).
    /// Must be a power of two.
    pub card_page_indices: u32,
    /// translationoption.py:185 `taggedpointers` (default off). When set,
    /// a small `int` may be stored as an unboxed immediate with an odd
    /// low bit; the collector must then skip such fields rather than read
    /// them as object headers (`is_valid_gc_object`, gc/base.py:380-383).
    pub taggedpointers: bool,
}

/// The variables [`GcConfig`] and [`MiniMarkGC::with_config`] resolve against,
/// for an embedder that has to hand its environment over rather than share it.
///
/// Published here so such a host does not keep its own copy of the list in step
/// with the collector; `PYPY_GC_DEBUG` and the tracing knobs are absent because
/// nothing in this file reads them.
pub const GC_ENV_NAMES: &[&str] = &[
    "PYPY_GC_NURSERY",
    "PYPY_GC_MAX_PINNED",
    "PYPY_GC_INCREMENT_STEP",
    "PYPY_GC_MAJOR_COLLECT",
    "PYPY_GC_GROWTH",
    "PYPY_GC_MIN",
    "PYPY_GC_MAX",
    "PYPY_GC_MAX_DELTA",
];

/// Environment an embedder supplies because the platform gives the process
/// none. Read only where `std::env` misses, so a host that has a real
/// environment resolves against it exactly as before.
///
/// `wasm32-unknown-unknown` is the case that needs it: `std::env::var` there
/// always fails, so every name in [`GC_ENV_NAMES`] reads as unset and a guest
/// runs the built-in defaults no matter what its host was configured with. The
/// interpreter's launcher options have the same problem and the same answer
/// (`pyre-wasm`'s `LAUNCH_ENV`).
static SUPPLIED_ENV: RwLock<Vec<(String, String)>> = RwLock::new(Vec::new());

/// Install the environment [`GC_ENV_NAMES`] resolves against when the process
/// has none. Call before the first allocation: the values are read once, when
/// the collector is built.
pub fn set_supplied_env(entries: Vec<(String, String)>) {
    *SUPPLIED_ENV.write().unwrap() = entries;
}

/// `std::env::var`, falling back to what the embedder supplied.
fn env_var(varname: &str) -> Option<String> {
    if let Ok(value) = std::env::var(varname) {
        return Some(value);
    }
    let supplied = SUPPLIED_ENV.read().unwrap();
    supplied
        .iter()
        .find(|(name, _)| name == varname)
        .map(|(_, value)| value.clone())
}

/// env.py:17-36 `_read_float_and_factor_from_env`. Parse `varname` as a float
/// with an optional `k`/`m`/`g` size suffix (optionally followed by `b`/`B`),
/// returning `(value, factor)`. `None` mirrors PyPy's `(0.0, 0)` absent /
/// unparseable result, which callers treat as "unset". `float(realvalue)`
/// accepts `inf`/`nan`, so this parser passes them through unchanged;
/// non-finite handling happens at the `int`/`r_uint` conversion sites, as
/// upstream where `int(inf)`/`r_uint(inf)` raise.
fn read_float_and_factor_from_env(varname: &str) -> Option<(f64, f64)> {
    let raw = env_var(varname)?;
    let mut value = raw.trim();
    if value.is_empty() {
        return None;
    }
    // env.py:20-21 — a trailing b/B is dropped before the k/m/g check.
    if value.len() > 1 && matches!(value.as_bytes().last(), Some(b'b' | b'B')) {
        value = &value[..value.len() - 1];
    }
    if value.is_empty() {
        return None;
    }
    // env.py:22-31 — a k/m/g suffix sets the factor; otherwise factor 1.
    let (number, factor) = match value.as_bytes().last().copied() {
        Some(b'k') | Some(b'K') => (&value[..value.len() - 1], 1024.0),
        Some(b'm') | Some(b'M') => (&value[..value.len() - 1], 1024.0 * 1024.0),
        Some(b'g') | Some(b'G') => (&value[..value.len() - 1], 1024.0 * 1024.0 * 1024.0),
        _ => (value, 1.0),
    };
    let parsed = number.parse::<f64>().ok()?;
    Some((parsed, factor))
}

/// env.py:38-44 `read_from_env` / `read_uint_from_env`: `value * factor` as an
/// integer byte count. `None` (unset / unparseable / non-positive) lets callers
/// fall back to the default, mirroring PyPy's `if x > 0` guards. PyPy's
/// `read_uint_from_env` r_uint-wraps a negative product to a huge positive; pyre
/// treats non-positive as unset, differing only on nonsensical negative input.
/// A non-finite product is treated as unset too: `int(inf)`/`r_uint(inf)` raise
/// upstream, so we fall back to the default instead of a `usize::MAX` byte count.
fn read_uint_from_env(varname: &str) -> Option<usize> {
    let (value, factor) = read_float_and_factor_from_env(varname)?;
    let bytes = value * factor;
    (bytes.is_finite() && bytes > 0.0).then_some(bytes as usize)
}

/// incminimark.py:516-528 `PYPY_GC_MAX_PINNED`.  Unlike the byte-sized GC
/// options this is a plain integer count: an absent or empty value selects the
/// nursery-derived default, while an invalid or negative value leaves the
/// constructor's initial zero in place and therefore disables pinning.
///
/// A value too large for `usize` saturates. `int()` is unbounded upstream, so
/// the limit it installs there is simply larger than any reachable pin count —
/// which is what `usize::MAX` is here. The field is a `usize`, so saturating is
/// the only representable reading of "a cap this program can never hit".
fn read_max_number_of_pinned_objects() -> Option<usize> {
    let raw = env_var("PYPY_GC_MAX_PINNED")?;
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    if value.starts_with('-') {
        return Some(0);
    }
    Some(
        value
            .parse::<u128>()
            .map(|count| count.min(usize::MAX as u128) as usize)
            .unwrap_or(0),
    )
}

/// env.py:46-50 `read_float_from_env`: the plain float, but only when no size
/// factor was given (`factor != 1` → unset). Callers apply their own `> 1.0`
/// threshold gate.
fn read_float_from_env(varname: &str) -> Option<f64> {
    let (value, factor) = read_float_and_factor_from_env(varname)?;
    if factor != 1.0 {
        return None;
    }
    Some(value)
}

/// env.py:387-411 `get_darwin_sysctl_signed`: read a signed integer sysctl by
/// name; 0 on any error.  Upstream reuses this helper for FreeBSD's
/// `hw.usermem` probe as well as the Darwin cache/memory probes.
#[cfg(any(target_os = "macos", target_os = "freebsd"))]
fn get_sysctl_signed(name: &[u8]) -> i64 {
    let mut val: i64 = 0;
    let mut len = std::mem::size_of::<i64>();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            &mut val as *mut i64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && len == std::mem::size_of::<i64>() {
        val
    } else {
        0
    }
}

/// env.py:413-455 `get_L2cache_darwin`. Returns the performance-cluster
/// L2 plus the legacy L3 cache size via `sysctl`, or -1 when it cannot be
/// determined.  Apple documents lower performance-level indices as faster
/// cores, so `hw.perflevel0.l2cachesize` is the cache relevant to the cores
/// running the mutator.  Intel Macs do not expose that key and retain the
/// legacy `hw.l2cachesize` fallback.
#[cfg(target_os = "macos")]
fn get_l2cache() -> i64 {
    let mut l2cache = get_sysctl_signed(b"hw.perflevel0.l2cachesize\0");
    if l2cache <= 0 {
        l2cache = get_sysctl_signed(b"hw.l2cachesize\0");
    }
    let mangled = l2cache + get_sysctl_signed(b"hw.l3cachesize\0");
    if mangled > 0 { mangled } else { -1 }
}

/// env.py:149-210 `get_L2cache_linux2_cpuinfo`: find the smallest cache-size
/// entry across CPUs.  The label must begin immediately after a newline, and
/// the value must be expressed in K/kilobytes, exactly like upstream.
fn l2cache_from_cpuinfo(data: &[u8], label: &[u8]) -> i64 {
    let mut smallest = i64::MAX;
    for (line_index, line) in data.split(|byte| *byte == b'\n').enumerate() {
        // `_findend(data, '\n' + label, ...)` does not inspect the first line.
        if line_index == 0 || !line.starts_with(label) {
            continue;
        }
        let mut pos = label.len();
        while pos < line.len() && matches!(line[pos], b' ' | b'\t') {
            pos += 1;
        }
        if line.get(pos) != Some(&b':') {
            continue;
        }
        pos += 1;
        while pos < line.len() && matches!(line[pos], b' ' | b'\t') {
            pos += 1;
        }
        let start = pos;
        while pos < line.len() && line[pos].is_ascii_digit() {
            pos += 1;
        }
        if start == pos {
            continue;
        }
        let Ok(number) = std::str::from_utf8(&line[start..pos])
            .unwrap_or("")
            .parse::<i64>()
        else {
            continue;
        };
        while pos < line.len() && matches!(line[pos], b' ' | b'\t') {
            pos += 1;
        }
        if !matches!(line.get(pos), Some(b'K' | b'k')) {
            continue;
        }
        if let Some(bytes) = number.checked_mul(1024) {
            smallest = smallest.min(bytes);
        }
    }
    if smallest < i64::MAX { smallest } else { -1 }
}

/// File half of env.py:149-210 `get_L2cache_linux2_cpuinfo`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn get_l2cache_linux_cpuinfo(filename: &str, label: &[u8]) -> i64 {
    std::fs::read(filename)
        .ok()
        .map_or(-1, |data| l2cache_from_cpuinfo(&data, label))
}

/// Parse the leading decimal plus K/k form used by Linux sysfs cache sizes.
fn parse_sysfs_cache_size(data: &[u8]) -> Option<i64> {
    let end = data.iter().position(|byte| !byte.is_ascii_digit())?;
    if end == 0 || !matches!(data.get(end), Some(b'K' | b'k')) {
        return None;
    }
    std::str::from_utf8(&data[..end])
        .ok()?
        .parse::<i64>()
        .ok()?
        .checked_mul(1024)
}

/// env.py:287-356 `get_L2cache_linux2_system_cpu_index`: walk every
/// `cpuN/cache/indexM`, take the smallest L2 and L3, then sum them.  Missing
/// either level deliberately overflows the signed sentinel to a non-positive
/// value, matching translated RPython's `sys.maxint` arithmetic and fallback.
fn get_l2cache_linux_system_cpu_index(sys_cpu_root: &str) -> i64 {
    let mut cpu = 0usize;
    let mut l2cache = i64::MAX;
    let mut l3cache = i64::MAX;
    loop {
        let mut index = 0usize;
        loop {
            let cachedir = format!("{sys_cpu_root}/cpu{cpu}/cache/index{index}");
            let Ok(level_data) = std::fs::read(format!("{cachedir}/level")) else {
                break;
            };
            let Ok(level) = std::str::from_utf8(&level_data)
                .unwrap_or("")
                .trim()
                .parse::<i64>()
            else {
                break;
            };
            if level != 2 && level != 3 {
                index += 1;
                continue;
            }
            let Ok(size_data) = std::fs::read(format!("{cachedir}/size")) else {
                break;
            };
            if let Some(number) = parse_sysfs_cache_size(&size_data) {
                if level == 2 {
                    l2cache = l2cache.min(number);
                } else {
                    l3cache = l3cache.min(number);
                }
            }
            index += 1;
        }
        if index == 0 {
            break;
        }
        cpu += 1;
    }
    let mangled = l2cache.wrapping_add(l3cache);
    if mangled > 0 { mangled } else { -1 }
}

/// env.py:251-285 `get_L2cache_linux2_sparc`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn get_l2cache_linux_sparc(sys_cpu_root: &str) -> i64 {
    let mut cpu = 0usize;
    let mut smallest = i64::MAX;
    loop {
        let filename = format!("{sys_cpu_root}/cpu{cpu}/l2_cache_size");
        let Ok(data) = std::fs::read(filename) else {
            break;
        };
        let Ok(number) = std::str::from_utf8(&data)
            .unwrap_or("")
            .trim()
            .parse::<i64>()
        else {
            break;
        };
        smallest = smallest.min(number);
        cpu += 1;
    }
    if smallest < i64::MAX { smallest } else { -1 }
}

/// env.py:126-146 `get_L2cache_linux2`: select the literal upstream probe by
/// machine architecture.  Rust's architecture spellings differ slightly from
/// `os.uname()[4]`, so include both forms where the target exists.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn get_l2cache_linux_for(arch: &str, cpuinfo: &str, sys_cpu_root: &str) -> i64 {
    if arch.ends_with("86") || arch == "x86_64" {
        return get_l2cache_linux_cpuinfo(cpuinfo, b"cache size");
    }
    if matches!(arch, "alpha" | "ppc" | "powerpc") {
        return get_l2cache_linux_cpuinfo(cpuinfo, b"L2 cache");
    }
    if matches!(arch, "ia64" | "aarch64") {
        return get_l2cache_linux_system_cpu_index(sys_cpu_root);
    }
    if matches!(arch, "parisc" | "parisc64") {
        return get_l2cache_linux_cpuinfo(cpuinfo, b"D-cache");
    }
    if matches!(arch, "sparc" | "sparc64") {
        return get_l2cache_linux_sparc(sys_cpu_root);
    }
    -1
}

#[cfg(target_os = "linux")]
fn get_l2cache() -> i64 {
    get_l2cache_linux_for(
        std::env::consts::ARCH,
        "/proc/cpuinfo",
        "/sys/devices/system/cpu",
    )
}

/// env.py:437-438 `get_L2cache = globals().get('get_L2cache_' + sys.platform,
/// lambda: -1)`.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_l2cache() -> i64 {
    -1
}

/// env.py:443-450 `best_nursery_size_for_L2cache`. About half the L2 cache,
/// but only when it exceeds 8MB (so an L3-inclusive value does not size the
/// nursery off L3); otherwise the 4MB unknown-cache fallback
/// (`NURSERY_SIZE_UNKNOWN_CACHE`, env.py:440 = `DEFAULT_NURSERY_SIZE`).
fn best_nursery_size_for_l2cache(l2cache: i64) -> usize {
    if l2cache > 8 * 1024 * 1024 {
        (l2cache / 2) as usize
    } else {
        DEFAULT_NURSERY_SIZE
    }
}

/// env.py:452-456 `estimate_best_nursery_size`.
fn estimate_best_nursery_size() -> usize {
    best_nursery_size_for_l2cache(get_l2cache())
}

/// incminimark.py:453 `minsize = 2 * (self.nonlarge_max + 1)` — pyre stores
/// `nonlarge_max + 1` directly as the large-object cutoff, so the doubling is
/// of the cutoff itself.  A nursery below this cannot hold one non-large
/// object, which is why upstream never lets the environment configure one.
const LARGE_OBJECT_THRESHOLD: usize = (16384 + 512) * 8;

/// incminimark.py:459-473 — the environment-configured nursery size, and the
/// `debug_tiny_nursery` budget a request below `minsize` turns into.
///
/// ```python
/// newsize = env.read_from_env('PYPY_GC_NURSERY')
/// if newsize <= 0:
///     newsize = env.estimate_best_nursery_size()
///     if newsize <= 0:
///         newsize = defaultsize
/// if newsize < minsize:
///     self.debug_tiny_nursery = newsize & ~(WORD-1)
///     newsize = minsize
/// ```
///
/// `PYPY_GC_NURSERY=1` means "collect on every malloc", not "a one-byte
/// nursery": the arena stays at `minsize` so every non-large object still
/// fits, and the requested size becomes the budget
/// `collect_and_reserve` leaves free after each reservation.  Dropping the
/// clamp and keeping the raw value instead hands out an arena smaller than a
/// single object.
fn default_nursery_size() -> (usize, Option<usize>) {
    // estimate_best_nursery_size never returns <= 0 (its floor is the 4MB
    // unknown-cache fallback), so the final `defaultsize` arm is unreachable.
    let newsize = read_uint_from_env("PYPY_GC_NURSERY").unwrap_or_else(estimate_best_nursery_size);
    let minsize = 2 * LARGE_OBJECT_THRESHOLD;
    if newsize < minsize {
        (minsize, Some(newsize & !(std::mem::size_of::<usize>() - 1)))
    } else {
        (newsize, None)
    }
}

/// env.py:67 `addressable_size = float(2**63)` for a 64-bit host: the most
/// memory the process could address, used as the fallback / upper clamp when
/// the real total-memory probe is unavailable or larger.
const ADDRESSABLE_SIZE: f64 = 9_223_372_036_854_775_808.0; // 2**63

/// env.py:100-110 `get_total_memory_darwin`. Clamp a sysctl-probed total:
/// fall back to the addressable size when the probe failed (`<= 0`) and cap it
/// at the addressable size otherwise.
#[cfg(any(target_os = "macos", target_os = "freebsd"))]
fn get_total_memory_sysctl(result: i64) -> f64 {
    if result <= 0 {
        ADDRESSABLE_SIZE
    } else {
        (result as f64).min(ADDRESSABLE_SIZE)
    }
}

/// env.py:70-98 `get_total_memory_linux`. Read `/proc/meminfo`, parse the
/// `MemTotal:` line (kB) into a byte count, then clamp: fall back to the
/// addressable size on read/parse failure (`result < 0.0`) and cap it at the
/// addressable size otherwise. The `< 0.0` failure sentinel — NOT the darwin
/// `<= 0` — must be kept (a probed `MemTotal: 0` is degenerate but not the
/// "probe failed" marker).
#[cfg(target_os = "linux")]
fn get_total_memory_linux(filename: &str) -> f64 {
    let mut result = -1.0_f64;
    // env.py:74-80 `os.read(fd, 4096)`: `MemTotal:` is always the first line
    // of `/proc/meminfo`, so the first 4 KiB always contain it.
    if let Ok(buf) = std::fs::read(filename) {
        let buf = &buf[..buf.len().min(4096)];
        let prefix = b"MemTotal:";
        if buf.starts_with(prefix) {
            // env.py:83 `_skipspace`: advance past ' ' / '\t' after the prefix.
            let mut start = prefix.len();
            while start < buf.len() && (buf[start] == b' ' || buf[start] == b'\t') {
                start += 1;
            }
            // env.py:85-86: take the leading ASCII-digit run.
            let mut stop = start;
            while stop < buf.len() && buf[stop].is_ascii_digit() {
                stop += 1;
            }
            if start < stop {
                let digits = std::str::from_utf8(&buf[start..stop]).unwrap_or("");
                if let Ok(kb) = digits.parse::<f64>() {
                    result = kb * 1024.0; // env.py:88 assume kB
                }
            }
        }
    }
    if result < 0.0 {
        ADDRESSABLE_SIZE
    } else {
        result.min(ADDRESSABLE_SIZE)
    }
}

/// env.py:113-127 `get_total_memory`. Total physical memory in bytes.
/// Linux reads `/proc/meminfo`; macOS reads `hw.memsize`; FreeBSD reads
/// `hw.usermem`; every other platform returns the addressable size
/// (env.py:113-127).
#[cfg(target_os = "linux")]
fn get_total_memory() -> f64 {
    get_total_memory_linux("/proc/meminfo")
}

#[cfg(target_os = "macos")]
fn get_total_memory() -> f64 {
    get_total_memory_sysctl(get_sysctl_signed(b"hw.memsize\0"))
}

#[cfg(target_os = "freebsd")]
fn get_total_memory() -> f64 {
    get_total_memory_sysctl(get_sysctl_signed(b"hw.usermem\0"))
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "freebsd")))]
fn get_total_memory() -> f64 {
    ADDRESSABLE_SIZE
}

impl Default for GcConfig {
    fn default() -> Self {
        // large_object = (16384+512)*8 from incminimark for 64-bit
        let (nursery_size, debug_tiny_nursery) = default_nursery_size();
        GcConfig {
            nursery_size,
            debug_tiny_nursery,
            large_object_threshold: LARGE_OBJECT_THRESHOLD,
            card_page_indices: 128,
            taggedpointers: false,
        }
    }
}

/// Root set: a list of locations that hold GcRef values the GC must trace.
///
/// Each root is a pointer to a GcRef-sized slot. During collection,
/// the GC reads the GcRef from this slot, traces it, and writes back
/// the (possibly updated) value.
pub struct RootSet {
    /// Stack roots: mutable pointers to GcRef slots on the stack or in frames.
    roots: Vec<*mut GcRef>,
}

unsafe impl Send for RootSet {}

impl RootSet {
    pub fn new() -> Self {
        RootSet { roots: Vec::new() }
    }

    /// Add a root. The pointer must remain valid until removed.
    ///
    /// # Safety
    /// The caller must ensure the pointer remains valid for the lifetime of the root.
    pub unsafe fn add(&mut self, root: *mut GcRef) {
        self.roots.push(root);
    }

    /// Remove a root.
    pub fn remove(&mut self, root: *mut GcRef) {
        // Host root brackets are stack-shaped, matching RPython's
        // shadowstack push/pop discipline.  Keep the general out-of-order
        // fallback for callers that hold overlapping guards, but make the
        // overwhelmingly common matching-pop path O(1).
        if self.roots.last().copied() == Some(root) {
            self.roots.pop();
            return;
        }
        if let Some(pos) = self.roots.iter().position(|r| *r == root) {
            self.roots.swap_remove(pos);
        }
    }

    /// Clear all roots.
    pub fn clear(&mut self) {
        self.roots.clear();
    }

    /// Number of roots.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
}

impl Default for RootSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Default card page shift: each card covers 2^7 = 128 array elements.
pub const DEFAULT_CARD_PAGE_SHIFT: u32 = 7;

/// incminimark.py:2390-2634 major-collection state machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GcState {
    Scanning,
    Marking,
    Sweeping,
    Finalizing,
}

impl GcState {
    fn encoded(self) -> u8 {
        match self {
            Self::Scanning => crate::GcStepTransition::SCANNING,
            Self::Marking => crate::GcStepTransition::MARKING,
            Self::Sweeping => crate::GcStepTransition::SWEEPING,
            Self::Finalizing => crate::GcStepTransition::FINALIZING,
        }
    }
}

/// State for incremental major marking.
///
/// Instead of doing a full mark-sweep in one pause, the marking work
/// is spread across multiple minor collections. Each minor collection
/// piggybacks an incremental marking step that processes a bounded
/// number of objects from the gray stack.
struct IncrementalMarkState {
    /// incminimark.py `objects_to_trace`: objects discovered by the ordinary
    /// marking walk.
    gray_stack: Vec<usize>,
    /// incminimark.py `more_objects_to_trace`: objects exposed or modified by
    /// the mutator while MARKING.  Keeping this separate is the termination
    /// mechanism at incminimark.py:2453-2470: only when the ordinary worklist
    /// consumed less than half its step does the collector swap this list in
    /// and drain it without a budget.
    more_gray_stack: Vec<usize>,
    /// Number of objects marked so far in this cycle.
    objects_marked: usize,
    /// Target number of bytes to trace per increment.
    ///
    /// Mirrors incminimark.py:448,500-504 `gc_increment_step`, whose runtime
    /// default is `nursery_size * 4` when `PYPY_GC_INCREMENT_STEP` is unset.
    mark_budget_per_step: usize,
    /// Reusable buffer for `mark_object`: the FIXED-part GC pointer offsets of
    /// the object currently being traced (bounded by the struct's field count),
    /// copied out so the immutable `self.types` borrow is released before
    /// greying (which mutates `self.incr_state.gray_stack`). RPython's
    /// `_collect_obj` visitor pushes straight onto the gray stack; pyre copies
    /// only the small fixed offsets and streams variable-part items one at a
    /// time, so a large varsize GC-pointer array is never retained here.
    mark_offsets: Vec<usize>,
}

impl IncrementalMarkState {
    fn new(nursery_size: usize) -> Self {
        // incminimark.py:500-504: an explicit positive byte budget wins;
        // otherwise mark four nursery sizes per incremental step.
        let mark_budget_per_step = read_uint_from_env("PYPY_GC_INCREMENT_STEP")
            .unwrap_or_else(|| nursery_size.saturating_mul(4))
            .max(1);
        IncrementalMarkState {
            gray_stack: Vec::new(),
            more_gray_stack: Vec::new(),
            objects_marked: 0,
            mark_budget_per_step,
            mark_offsets: Vec::new(),
        }
    }
}

struct FinalizerHandler {
    deque: VecDeque<usize>,
    trigger: FinalizerTriggerFn,
}

/// The MiniMark generational GC.
#[allow(non_snake_case)] // _T_IS_RPYTHON_INSTANCE_BYTE keeps the RPython spelling
pub struct MiniMarkGC {
    /// The nursery (young generation).
    nursery: Nursery,
    /// gc.py:525-531 published nursery-top slot read by generated inline
    /// allocation paths. The atomic publishes limit changes to generated code
    /// and Rust-side readers. The Box keeps its baked address stable even if
    /// the containing GC value moves.
    published_nursery_top: Box<AtomicUsize>,
    /// The old generation.
    oldgen: OldGen,
    /// Type registry for tracing objects.
    pub types: TypeRegistry,
    /// Root set.
    pub roots: RootSet,
    /// Remembered set: old objects that may point to young objects.
    /// These are old-gen object payload addresses whose TRACK_YOUNG_PTRS
    /// flag has been cleared by the write barrier.
    remembered_set: Vec<usize>,
    /// incminimark.py:355 `prebuilt_root_objects = AddressStack()`.
    /// Immortal objects enter exactly once, when their first pointer write
    /// clears NO_HEAP_PTRS in the write barrier.
    prebuilt_root_objects: Vec<usize>,
    /// incminimark.py:346-352: objects with GCFLAG_CARDS_SET bit.
    /// Card bits are stored inline before each object's GcHeader.
    /// This list tracks which objects have at least one card bit set.
    old_objects_with_cards_set: Vec<usize>,
    /// incminimark.py:414 `self.young_objects_with_weakrefs =
    /// self.AddressStack()`. Nursery-resident WEAKREF objects pending
    /// minor-cycle target invalidation. Populated by `alloc_with_type`
    /// when the requested type's `TypeInfo.is_weakref` is set
    /// (incminimark.py:692 `if contains_weakptr:`). Drained at end of
    /// minor cycle by `invalidate_young_weakrefs`
    /// (incminimark.py:1866-1867, :3058-3105).
    young_objects_with_weakrefs: Vec<usize>,
    /// incminimark.py:415 `self.old_objects_with_weakrefs =
    /// self.AddressStack()`. Old-gen WEAKREF objects whose target is
    /// also in old-gen. Populated by `invalidate_young_weakrefs` for
    /// survivors and by direct old-gen weakref allocation. Drained by
    /// `invalidate_old_weakrefs` during the sweep phase of a major
    /// collection (incminimark.py:3107-3133); the major-side
    /// consumer lands in the major-collection sweep phase.
    old_objects_with_weakrefs: Vec<usize>,
    /// incminimark.py:407 `self.young_objects_with_destructors =
    /// self.AddressStack()`. Nursery-resident objects whose type carries
    /// a lightweight `TypeInfo.destructor`. Populated by
    /// `register_young_object_if_needed` at allocation (incminimark.py:689
    /// `if needs_finalizer:`). Drained at end of minor cycle by
    /// `deal_with_young_objects_with_destructors` (incminimark.py:1868,
    /// :2884-2895): a dead object's destructor runs, a survivor moves to
    /// `old_objects_with_destructors`.
    young_objects_with_destructors: Vec<usize>,
    /// incminimark.py:408 `self.old_objects_with_destructors =
    /// self.AddressStack()`. Old-gen objects with a destructor, populated
    /// by promotion from the young list and by direct old-gen allocation.
    /// Drained by `deal_with_old_objects_with_destructors`
    /// (incminimark.py:2510-2511, :2897-2912) before the major sweep:
    /// a VISITED object survives, a dying one's destructor runs.
    old_objects_with_destructors: Vec<usize>,
    /// incminimark.py:388-390.  Registrations first enter the probably-young
    /// deque as `(object, finalizer-handler-index)` pairs and are promoted by
    /// the next minor collection.
    probably_young_objects_with_finalizers: VecDeque<(usize, usize)>,
    /// incminimark.py:392-393.  Old objects still waiting to become
    /// unreachable, paired with their translated FinalizerQueue handler.
    old_objects_with_finalizers: VecDeque<(usize, usize)>,
    /// gc/base.py finalizer handlers: one death deque and trigger per queue.
    finalizer_handlers: Vec<FinalizerHandler>,
    finalizer_lock: bool,
    /// incminimark.py:394 `self.enabled = True`.
    enabled: bool,
    /// True while [`do_collect_oldgen_nonmoving`](MiniMarkGC::do_collect_oldgen_nonmoving)
    /// is running. A non-moving major skips the leading minor, so unlike
    /// `do_collect_full` it marks through a *populated* nursery: `mark_object`
    /// / `seed_major_root` set `flags::VISITED` on reachable nursery objects
    /// that no sweep then clears (the sweep only walks old-gen). While this is
    /// set, every nursery object greyed this cycle is recorded in
    /// `oldgen_nonmoving_nursery_marks` so VISITED can be cleared as the
    /// strictly-last step — otherwise `copy_nursery_object` would memcpy a
    /// stale VISITED bit into the next minor's promoted copy.
    oldgen_nonmoving_active: bool,
    /// Nursery payload addresses greyed during the current non-moving major
    /// (only populated while `oldgen_nonmoving_active`). Drained by the final
    /// VISITED-clear pass.
    oldgen_nonmoving_nursery_marks: Vec<usize>,
    /// incminimark.py:3160,3175-3182 — the raw-refcount lists, identity tables
    /// and dead queue, empty until the embedder calls
    /// [`rawrefcount_init`](MiniMarkGC::rawrefcount_init).
    rrc: rawrefcount::RawRefCount,
    /// Configuration.
    config: GcConfig,
    /// Count of minor collections performed.
    pub minor_collections: usize,
    /// Count of major collections performed.
    pub major_collections: usize,
    /// `incminimark.py:self.hooks`, supplied by the translated standalone
    /// target and restricted to its allocation-free low-level surface.
    hooks: GcHooks,
    /// Hook accounting captured at the MARKING -> SWEEPING seam.
    stat_ac_arenas_count: usize,
    stat_rawmalloced_total_size: usize,
    /// incminimark.py:387 `total_gc_time`, in seconds.
    total_gc_time: f64,
    /// incminimark.py:2390-2634 `gc_state`.
    gc_state: GcState,
    /// State for incremental major collection.
    incr_state: IncrementalMarkState,
    /// incminimark.py:304 `self.major_collection_threshold`. After a major
    /// collection, the next one is triggered once the total old-gen size
    /// grows to this many times the surviving size (TRANSLATION_PARAMS
    /// default 1.82; `PYPY_GC_MAJOR_COLLECT` override).
    major_collection_threshold: f64,
    /// incminimark.py:305 `self.growth_rate_max`. Caps how fast the
    /// next-major threshold may grow from one collection to the next
    /// (TRANSLATION_PARAMS default 1.4; `PYPY_GC_GROWTH` override).
    growth_rate_max: f64,
    /// incminimark.py:307,488,562 `self.min_heap_size`. Floor below which the
    /// next-major threshold is never set: `max(PYPY_GC_MIN or nursery*8,
    /// nursery*major_collection_threshold)`. This floor is what stops
    /// allocation-heavy, tiny-live-set workloads (e.g. recursive fib) from
    /// thrashing major cycles off a near-zero surviving baseline.
    min_heap_size: f64,
    /// incminimark.py:308 `self.max_heap_size` (`PYPY_GC_MAX`; 0.0 = unbounded).
    max_heap_size: f64,
    /// incminimark.py:310,498 `self.max_delta` (`PYPY_GC_MAX_DELTA`, else
    /// `0.125 * get_total_memory()`). Caps the absolute next-major threshold
    /// growth per cycle.
    max_delta: f64,
    /// incminimark.py:309 `self.max_heap_size_already_raised`. Set true the
    /// first time a bounded major collection finds the heap already over
    /// `max_heap_size`; a second such event aborts the process
    /// (`out_of_memory`, minimarkpage.py:611 -> fatalerror) instead of raising
    /// another MemoryError. Only reachable when `max_heap_size > 0`
    /// (`PYPY_GC_MAX` set), so it stays false in the default unbounded config.
    max_heap_size_already_raised: bool,
    /// Set by `finish_incremental_cycle` when a bounded major collection leaves
    /// the heap over `max_heap_size`. The allocation that triggered the
    /// collection (`alloc_with_type`) reads and clears this and returns NULL,
    /// so the JIT `CHECK_MEMORY_ERROR` path / interpreter allocation chokepoint
    /// raises `MemoryError` (incminimark.py:2603-2615 `raise MemoryError`,
    /// lowered to the NULL-return the backend already understands).
    oom_pending: bool,
    /// Byte size of the allocation that triggered the current nursery-full
    /// collection, carried across `do_collect_nursery` so
    /// `finish_incremental_cycle` can pass it to `threshold_reached`
    /// (incminimark's `major_collection_step(reserving_size)` argument). Carried
    /// on the collector rather than threaded through the public
    /// `do_collect_nursery` signature and its many call sites; reset to 0 by the
    /// triggering allocation once the collection returns.
    pending_reserving_size: usize,
    /// incminimark.py:568 `self.next_major_collection_initial`. The
    /// pre-reservation threshold; `set_major_threshold_from` grows the next
    /// threshold relative to this value, bounded by `growth_rate_max`.
    next_major_collection_initial: f64,
    /// incminimark.py:569 `self.next_major_collection_threshold`. A new
    /// incremental major cycle starts once `get_total_memory_used()` reaches
    /// this value (`threshold_reached`).
    next_major_collection_threshold: f64,
    /// incminimark.py:2492,3011-3020 `self.kept_alive_by_finalizer`.
    ///
    /// During finalization ordering, unreachable object graphs are marked so
    /// their app-level finalizers can still run.  They remain physically live
    /// after this sweep, but upstream excludes exactly these bytes from the
    /// survivor baseline used to size the next major threshold; otherwise a
    /// workload consisting mostly of finalizable garbage grows that threshold
    /// forever (PyPy issue #2590).
    kept_alive_by_finalizer: usize,
    /// Bytes promoted to old gen since the current incremental cycle started.
    ///
    /// Mirrors incminimark's `size_objects_made_old`.
    bytes_made_old_since_cycle: usize,
    /// Promotion credit granted by completed major-GC steps within the current
    /// incremental cycle.
    ///
    /// Mirrors incminimark's `threshold_objects_made_old`.
    threshold_bytes_made_old: usize,
    /// incminimark.py:1816,2174,2232 `nursery_surviving_size`: allocator-size
    /// bytes promoted by the most recent minor collection.  The MARKING step
    /// traces at least twice this many bytes so a high-survival nursery cannot
    /// make the gray frontier grow faster than it is consumed.
    nursery_surviving_size: usize,
    /// Pinned nursery objects that must not be moved during minor collection.
    pinned_objects: IndexSet<usize>,
    /// incminimark.py:426-440 pin-liveness state. Each minor collection
    /// rebuilds `surviving_pinned_objects` from actual traced edges; old
    /// parents found along those edges are revisited by the next minor.
    surviving_pinned_objects: IndexSet<usize>,
    old_objects_pointing_to_pinned: Vec<usize>,
    /// incminimark.py:311,430 `max_number_of_pinned_objects` and
    /// `pinned_objects_in_nursery`.
    max_number_of_pinned_objects: usize,
    pinned_objects_in_nursery: usize,
    /// minimark.py:338 `nursery_objects_shadows = AddressDict()`.
    /// Maps nursery object payload address → pre-allocated old-gen
    /// shadow payload address.  When `id()` or `identityhash()` is
    /// called on a nursery object, a shadow is allocated in old-gen
    /// and registered here.  `copy_nursery_object` copies to the
    /// shadow instead of a fresh allocation.  Cleared after each
    /// minor collection.
    nursery_objects_shadows: AddressMap<usize>,
    /// Registry of compiled code regions for GC root scanning.
    pub compiled_code_registry: CompiledCodeRegistry,
    /// llsupport/gc.py:563 vtable→typeid mapping. RPython derives this
    /// arithmetically from the GC `type_info_group` base; pyre's GC
    /// keeps an explicit table because frontends register vtables
    /// independently of any translator pipeline. This has RPython
    /// AddressDict hashing: keys are addresses and insertion order is not
    /// observable at any call site.
    vtable_to_type_id: AddressMap<u32>,
    /// `gc.py:603-622 _setup_guard_is_object` instance state.
    /// `_infobits_offset` is the byte offset of `infobits` inside
    /// `TYPE_INFO`; `_infobits_offset_plus` is the additional offset
    /// past leading zero bytes of the little-endian flag word.
    /// `_T_IS_RPYTHON_INSTANCE_BYTE` is the single non-zero byte
    /// pulled out of the flag word.
    ///
    /// Mirrors RPython's GcLLDescr_framework instance fields exactly
    /// — they're computed once at construction by `_setup_guard_is_object`
    /// and read by `get_translated_info_for_guard_is_object`. The
    /// `_T_IS_RPYTHON_INSTANCE_BYTE` field name keeps the RPython
    /// upper-case spelling so a `git grep` across the two trees
    /// finds the same name on both sides.
    _infobits_offset: usize,
    _infobits_offset_plus: usize,
    _T_IS_RPYTHON_INSTANCE_BYTE: u8,
    /// incminimark.py:315-317: log2(card_page_indices).
    /// 0 if card marking is disabled.
    card_page_shift: u32,
    /// When true, `alloc_with_type` forces a full collection before every
    /// allocation (see the `gc_stress` cargo feature). Off by default; opted in
    /// per instance via [`MiniMarkGC::set_stress_collect`]. Exists only under
    /// the feature so the default build carries no extra field or branch.
    #[cfg(feature = "gc_stress")]
    stress_collect: bool,
}

impl MiniMarkGC {
    /// Create a new GC with default configuration.
    pub fn new() -> Self {
        Self::with_config(GcConfig::default())
    }

    /// Create a new GC with custom configuration.
    pub fn with_config(config: GcConfig) -> Self {
        let nursery_size = config.nursery_size;

        // incminimark.py:261,268,475-481 — major_collection_threshold /
        // growth_rate_max from TRANSLATION_PARAMS, overridable by env.
        let major_collection_threshold = read_float_from_env("PYPY_GC_MAJOR_COLLECT")
            .filter(|v| *v > 1.0)
            .unwrap_or(1.82);
        let growth_rate_max = read_float_from_env("PYPY_GC_GROWTH")
            .filter(|v| *v > 1.0)
            .unwrap_or(1.4);
        // incminimark.py:483-488 — min_heap_size: PYPY_GC_MIN, else nursery*8.
        let mut min_heap_size = read_uint_from_env("PYPY_GC_MIN")
            .map(|v| v as f64)
            .unwrap_or(nursery_size as f64 * 8.0);
        // incminimark.py:490-492 — max_heap_size: PYPY_GC_MAX, else 0 (unbounded).
        let max_heap_size = read_uint_from_env("PYPY_GC_MAX")
            .map(|v| v as f64)
            .unwrap_or(0.0);
        // incminimark.py:494-498 — max_delta: PYPY_GC_MAX_DELTA, else
        // 0.125 * env.get_total_memory().
        let max_delta = read_uint_from_env("PYPY_GC_MAX_DELTA")
            .map(|v| v as f64)
            .unwrap_or_else(|| 0.125 * get_total_memory());
        // incminimark.py:562-563 — allocate_nursery floors min_heap_size by
        // nursery_size * major_collection_threshold.
        min_heap_size = min_heap_size.max(nursery_size as f64 * major_collection_threshold);

        let nursery = Nursery::new(config.nursery_size);
        // incminimark.py:516-528. `nonlarge_max + 1` is the large-object
        // cutoff; pyre stores that cutoff directly in the configuration.
        let max_number_of_pinned_objects =
            read_max_number_of_pinned_objects().unwrap_or_else(|| {
                config
                    .nursery_size
                    .checked_div(config.large_object_threshold.saturating_mul(2))
                    .unwrap_or(0)
            });
        let published_nursery_top = Box::new(AtomicUsize::new(nursery.top_ptr() as usize));
        let mut gc = MiniMarkGC {
            nursery,
            published_nursery_top,
            oldgen: OldGen::new(),
            types: TypeRegistry::new(),
            roots: RootSet::new(),
            remembered_set: Vec::new(),
            prebuilt_root_objects: Vec::new(),
            old_objects_with_cards_set: Vec::new(),
            young_objects_with_weakrefs: Vec::new(),
            old_objects_with_weakrefs: Vec::new(),
            young_objects_with_destructors: Vec::new(),
            old_objects_with_destructors: Vec::new(),
            probably_young_objects_with_finalizers: VecDeque::new(),
            old_objects_with_finalizers: VecDeque::new(),
            finalizer_handlers: Vec::new(),
            finalizer_lock: false,
            enabled: true,
            oldgen_nonmoving_active: false,
            oldgen_nonmoving_nursery_marks: Vec::new(),
            rrc: rawrefcount::RawRefCount::default(),
            config,
            minor_collections: 0,
            major_collections: 0,
            hooks: GcHooks,
            stat_ac_arenas_count: 0,
            stat_rawmalloced_total_size: 0,
            total_gc_time: 0.0,
            gc_state: GcState::Scanning,
            incr_state: IncrementalMarkState::new(nursery_size),
            major_collection_threshold,
            growth_rate_max,
            min_heap_size,
            max_heap_size,
            max_delta,
            max_heap_size_already_raised: false,
            oom_pending: false,
            pending_reserving_size: 0,
            // incminimark.py:568-569 — both initialized to min_heap_size,
            // then refined by set_major_threshold_from(0.0) below.
            next_major_collection_initial: min_heap_size,
            next_major_collection_threshold: min_heap_size,
            kept_alive_by_finalizer: 0,
            bytes_made_old_since_cycle: 0,
            threshold_bytes_made_old: 0,
            nursery_surviving_size: 0,
            pinned_objects: IndexSet::new(),
            surviving_pinned_objects: IndexSet::new(),
            old_objects_pointing_to_pinned: Vec::new(),
            max_number_of_pinned_objects,
            pinned_objects_in_nursery: 0,
            nursery_objects_shadows: AddressMap::default(),
            compiled_code_registry: CompiledCodeRegistry::new(),
            vtable_to_type_id: AddressMap::default(),
            _infobits_offset: 0,
            _infobits_offset_plus: 0,
            _T_IS_RPYTHON_INSTANCE_BYTE: 0,
            card_page_shift: 0,
            // gc.py:603-617 has no analogue; seeded from the `MAJIT_GC_STRESS`
            // env var so a whole binary can be stressed without code edits,
            // while individual tests use `set_stress_collect`.
            #[cfg(feature = "gc_stress")]
            stress_collect: std::env::var_os("MAJIT_GC_STRESS").is_some(),
        };
        // incminimark.py:314-317
        if gc.config.card_page_indices > 0 {
            gc.card_page_shift = 0;
            while (1u32 << gc.card_page_shift) < gc.config.card_page_indices {
                gc.card_page_shift += 1;
            }
        }
        // incminimark.py:570 — allocate_nursery finalizes the first threshold.
        gc.set_major_threshold_from(0.0, 0.0);
        gc._setup_guard_is_object();
        gc
    }

    /// Refresh the gc.py:525-531 published nursery-top slot from the real
    /// allocator bound.
    fn refresh_published_nursery_top(&mut self) {
        self.published_nursery_top
            .store(self.nursery.top_ptr() as usize, Ordering::Release);
    }

    // ── incminimark.py:1292-1308 card marking geometry ──

    /// incminimark.py:1292-1299: number of machine words needed for card bits.
    fn card_marking_words_for_length(&self, length: usize) -> usize {
        const LONG_BIT: usize = 64;
        const LONG_BIT_SHIFT: usize = 6;
        (length + (LONG_BIT << self.card_page_shift) - 1)
            >> (self.card_page_shift as usize + LONG_BIT_SHIFT)
    }

    /// incminimark.py:1301-1308: number of bytes needed for card bits.
    fn card_marking_bytes_for_length(&self, length: usize) -> usize {
        (length + (8 << self.card_page_shift) - 1) >> (self.card_page_shift as usize + 3)
    }

    /// incminimark.py:1622-1625: address of card byte at `byteindex`.
    /// Card bytes are stored in reverse order before the GcHeader.
    /// `obj` is the payload address (after GcHeader).
    #[inline]
    fn get_card_ptr(obj: usize, byteindex: usize) -> *mut u8 {
        // addr_byte = obj - size_gc_header  (= GcHeader address)
        // return addr_byte + (~byteindex)   (= addr_byte - 1 - byteindex)
        (obj - GcHeader::SIZE - 1 - byteindex) as *mut u8
    }

    /// incminimark.py:955-1088: allocate a large object with optional card
    /// marker bits prepended before the GcHeader.
    ///
    /// `has_gc_ptrs_in_var` should be true for arrays containing GC pointers
    /// (i.e. `has_gcptr_in_varsize(typeid)` in RPython).
    pub fn alloc_in_oldgen_with_cards(
        &mut self,
        type_id: u32,
        total_size: usize,
        length: usize,
        has_gc_ptrs_in_var: bool,
    ) -> GcRef {
        // incminimark.py:1017-1030
        let (card_header_bytes, extra_flags) =
            if self.card_page_shift > 0 && has_gc_ptrs_in_var && length > 0 {
                let extra_words = self.card_marking_words_for_length(length);
                let chs = 8 * extra_words; // WORD * extra_words
                (chs, flags::HAS_CARDS | flags::TRACK_YOUNG_PTRS)
            } else {
                (0, flags::TRACK_YOUNG_PTRS)
            };

        let ptr = self
            .oldgen
            .alloc_with_card_header(total_size, card_header_bytes);
        let hdr = unsafe { &mut *(ptr as *mut GcHeader) };
        *hdr = GcHeader::with_flags(type_id, self.oldgen_birth_flags(extra_flags));
        self.bytes_made_old_since_cycle = self
            .bytes_made_old_since_cycle
            .saturating_add(card_header_bytes + total_size);
        GcRef((ptr as usize) + GcHeader::SIZE)
    }

    /// `gc.py:603-617 _setup_guard_is_object` parity. Computes
    /// `(_infobits_offset, _infobits_offset_plus, _T_IS_RPYTHON_INSTANCE_BYTE)`
    /// once at construction time and stores them as instance state.
    /// Mirrors line-by-line:
    ///
    /// ```python
    /// def _setup_guard_is_object(self):
    ///     from rpython.memory.gctypelayout import GCData, T_IS_RPYTHON_INSTANCE
    ///     import struct
    ///     infobits_offset, _ = symbolic.get_field_token(
    ///         GCData.TYPE_INFO, 'infobits', True)
    ///     mask = struct.pack("l", T_IS_RPYTHON_INSTANCE)
    ///     assert mask.count('\x00') == len(mask) - 1
    ///     infobits_offset_plus = 0
    ///     while mask.startswith('\x00'):
    ///         infobits_offset_plus += 1
    ///         mask = mask[1:]
    ///     self._infobits_offset = infobits_offset
    ///     self._infobits_offset_plus = infobits_offset_plus
    ///     self._T_IS_RPYTHON_INSTANCE_BYTE = ord(mask[0])
    /// ```
    fn _setup_guard_is_object(&mut self) {
        // `symbolic.get_field_token(TYPE_INFO, 'infobits', True)`.
        let infobits_offset = TypeInfoLayout::INFOBITS_OFFSET;
        // `mask = struct.pack("l", T_IS_RPYTHON_INSTANCE)`.
        let mask = TypeInfoLayout::T_IS_RPYTHON_INSTANCE.to_le_bytes();
        // `assert mask.count('\x00') == len(mask) - 1`.
        let nonzero = mask.iter().filter(|&&b| b != 0).count();
        assert_eq!(
            nonzero, 1,
            "T_IS_RPYTHON_INSTANCE must occupy exactly one byte of \
             the packed Signed word (gc.py:610)"
        );
        // `while mask.startswith('\x00'): infobits_offset_plus += 1; mask = mask[1:]`.
        let mut infobits_offset_plus = 0usize;
        while infobits_offset_plus < mask.len() && mask[infobits_offset_plus] == 0 {
            infobits_offset_plus += 1;
        }
        self._infobits_offset = infobits_offset;
        self._infobits_offset_plus = infobits_offset_plus;
        self._T_IS_RPYTHON_INSTANCE_BYTE = mask[infobits_offset_plus];
    }

    /// Register a type and return its ID.
    pub fn register_type(&mut self, info: TypeInfo) -> u32 {
        self.types.register(info)
    }

    /// Check if an address is in the nursery.
    #[inline]
    pub fn is_in_nursery(&self, addr: usize) -> bool {
        debug_assert!(
            !self.is_tagged_immediate(addr),
            "odd-valued (i.e. tagged) pointer unexpected here"
        );
        self.nursery.contains(addr)
    }

    /// The tagged-immediate test used by `is_valid_gc_object`
    /// (gc/base.py:380-383). With `taggedpointers` set (enablement only;
    /// default off per `translationoption.py:185`), a small `int` may be
    /// stored as an unboxed immediate with an odd low bit; such a value is
    /// not a heap object and must not be read as an object header.
    #[inline]
    fn is_tagged_immediate(&self, addr: usize) -> bool {
        self.config.taggedpointers && (addr & 1 == 1)
    }

    /// gc/base.py:380-383 `is_valid_gc_object`.
    #[inline]
    fn is_valid_gc_object(&self, addr: usize) -> bool {
        addr != 0 && !self.is_tagged_immediate(addr)
    }

    /// incminimark.py range-check parity: nursery membership is a pure
    /// range check; JIT inline nursery bump-alloc must produce GcRefs
    /// indistinguishable from the slow path's, so a side table would
    /// not stay in sync.
    #[inline]
    pub fn is_managed_heap_object(&self, addr: usize) -> bool {
        self.is_valid_gc_object(addr) && (self.nursery.contains(addr) || self.oldgen.contains(addr))
    }

    /// Resolve a PyObject header without asking the nursery/oldgen allocators.
    /// A registered vtable plus the exact matching header tid is the translated
    /// type-layout witness. The tid equality rejects headerless bootstrap
    /// objects whose preceding allocator word happens to look like flags.
    ///
    /// This is the ONLY admissible probe for an address outside both managed
    /// generations, and its caller (`do_write_barrier`) must go through it.
    /// Recording every `alloc_with_gc_header` result in an
    /// ownership table instead would make the answer exact, but that is a probe
    /// on the allocation fast path — the same shape measured at −15% in the GC
    /// box-probe experiment — so the tid witness stands.
    #[inline]
    fn registered_pyobject_header(&self, addr: usize) -> Option<*mut GcHeader> {
        if addr < GcHeader::SIZE || !addr.is_multiple_of(GcHeader::ALIGN) {
            return None;
        }
        let vtable = unsafe { *(addr as *const usize) };
        let expected_type_id = *self.vtable_to_type_id.get(&vtable)?;
        let hdr = unsafe { header_of(addr) };
        if unsafe { (*hdr).type_id() } == expected_type_id {
            Some(hdr)
        } else {
            None
        }
    }

    /// Valid-object nursery check for arbitrary GC fields/roots.
    #[inline]
    fn is_nursery_object_start(&self, addr: usize) -> bool {
        self.is_valid_gc_object(addr) && self.is_in_nursery(addr)
    }

    /// Allocate a fixed-size object with the given type ID and size (excluding header).
    /// Returns a GcRef pointing to the object payload (after the header).
    ///
    /// Split into an inline bump and an out-of-line tail like
    /// [`alloc_with_type_rooted`](Self::alloc_with_type_rooted).
    #[inline]
    pub fn alloc_with_type(&mut self, type_id: u32, payload_size: usize) -> GcRef {
        // gc_stress: force a full collection *before* allocating so that any
        // object live at this allocation point but unreachable from a
        // registered root or custom-trace path is moved/swept deterministically
        // — turning a latent dangling pointer into an immediate failure. The
        // collection runs before reserving the new object so the returned
        // pointer is never itself moved. Gated on both the `gc_stress` feature
        // (so default/release builds compile no branch) and the per-instance
        // `stress_collect` flag (so an opted-in GC stresses only the allocations
        // it owns, leaving unrelated suites — and their `minor_collections == 0`
        // assertions — untouched).
        #[cfg(feature = "gc_stress")]
        if self.stress_collect {
            self.do_collect_full();
        }

        let Some(total_size) = GcHeader::SIZE.checked_add(payload_size) else {
            return GcRef(0);
        };

        if total_size <= self.config.large_object_threshold {
            let ptr = self.nursery.alloc(total_size);
            if !ptr.is_null() {
                return self.finish_nursery_object(ptr, type_id);
            }
        }
        self.alloc_with_type_slow(type_id, total_size)
    }

    /// The outcomes [`alloc_with_type`](Self::alloc_with_type) leaves out of
    /// line: a large object, and a nursery too full to serve the bump.
    #[cold]
    #[inline(never)]
    fn alloc_with_type_slow(&mut self, type_id: u32, total_size: usize) -> GcRef {
        // Large objects go directly to old gen.
        if total_size > self.config.large_object_threshold {
            return self.alloc_in_oldgen(type_id, total_size);
        }

        // Nursery full: trigger minor collection and retry.
        // minimark.py:1282 collect_and_reserve parity. Carry the triggering
        // allocation size so a bounded major collection applies the
        // PYPY_GC_MAX out-of-memory policy against it (threshold_reached).
        self.pending_reserving_size = total_size;
        self.do_collect_nursery();
        self.pending_reserving_size = 0;
        // incminimark.py:2603-2615 — a bounded major collection that leaves
        // the heap over `max_heap_size` asks this allocation to fail so the
        // caller raises MemoryError: NULL propagates to the compiled-code
        // `CHECK_MEMORY_ERROR` path and to the interpreter allocation
        // chokepoint. Never taken in the unbounded default (`PYPY_GC_MAX`
        // unset), so the fallback below is unchanged there.
        if std::mem::take(&mut self.oom_pending) {
            return GcRef(0);
        }
        let ptr = self.nursery.alloc(total_size);
        let ptr = if ptr.is_null() {
            self.reserve_nursery_gap(total_size)
        } else {
            ptr
        };
        if ptr.is_null() && Self::nursery_allocation_size(total_size) > self.nursery.size() {
            // Production incminimark keeps `nonlarge_max` below the
            // nursery size. Tiny backend tests can configure the two
            // independently; retain their external-allocation fallback
            // only for an object that cannot physically fit anywhere.
            return self.alloc_in_oldgen(type_id, total_size);
        }
        assert!(
            !ptr.is_null(),
            "collect_and_reserve could not find nursery space for a non-large object"
        );
        self.apply_debug_tiny_nursery();
        self.finish_nursery_object(ptr, type_id)
    }

    /// `malloc_fixedsize` / `collect_and_reserve` with one native-stack root.
    ///
    /// RPython's GC transform puts live GC locals in a shadow-stack/stack-map,
    /// but the allocation fast path is still a plain nursery bump. Rust locals
    /// are not scanned, so callers pass the exceptional local slot explicitly.
    /// Register it only when the bump fails and a collection is actually
    /// required; this preserves the upstream fast path instead of doing a
    /// root-set add/remove for every allocation.
    ///
    /// Split into a `malloc_fast` body and its exceptional tail the way
    /// `framework.py:361-382` does: the copy it declares `inline=True` carries
    /// only the bump, so a caller pays a bump and a branch, while a large
    /// object or a full nursery costs a call. Without the split the whole
    /// function is a frame that saves ten callee-saved registers before it
    /// can reach a ten-instruction bump.
    ///
    /// # Safety
    /// `root` must point to a valid mutable `GcRef` slot until this call
    /// returns. `needs_write_barrier` must point to a valid mutable `bool`
    /// slot.
    #[inline]
    pub unsafe fn alloc_with_type_rooted(
        &mut self,
        type_id: u32,
        payload_size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.alloc_with_type_rooted_body::<false>(
                type_id,
                payload_size,
                root,
                needs_write_barrier,
            )
        }
    }

    /// [`alloc_with_type_rooted`](Self::alloc_with_type_rooted) for a type that
    /// carries neither a finalizer nor the weakref flag: `malloc_fast`
    /// (`framework.py:361-382`), the copy `gct_fv_gc_malloc` selects at
    /// `framework.py:830-838` for exactly that case.
    ///
    /// # Safety
    /// Same contract as [`alloc_with_type_rooted`](Self::alloc_with_type_rooted),
    /// plus `type_id` must name a type with no destructor and no weakref flag.
    #[inline]
    pub unsafe fn alloc_fast_with_type_rooted(
        &mut self,
        type_id: u32,
        payload_size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.alloc_with_type_rooted_body::<true>(
                type_id,
                payload_size,
                root,
                needs_write_barrier,
            )
        }
    }

    /// # Safety
    /// See [`alloc_with_type_rooted`](Self::alloc_with_type_rooted); `FAST`
    /// additionally carries `malloc_fast`'s obligation on `type_id`.
    #[inline]
    unsafe fn alloc_with_type_rooted_body<const FAST: bool>(
        &mut self,
        type_id: u32,
        payload_size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe { *needs_write_barrier = false };

        #[cfg(feature = "gc_stress")]
        if self.stress_collect {
            unsafe { self.roots.add(root) };
            self.do_collect_full();
            self.roots.remove(root);
        }

        let Some(total_size) = GcHeader::SIZE.checked_add(payload_size) else {
            return GcRef(0);
        };

        if total_size <= self.config.large_object_threshold {
            let ptr = self.nursery.alloc(total_size);
            if !ptr.is_null() {
                return self.finish_bumped_nursery_object::<FAST>(ptr, type_id);
            }
        }
        unsafe { self.alloc_with_type_rooted_slow(type_id, total_size, root, needs_write_barrier) }
    }

    /// The outcomes [`alloc_with_type_rooted`] leaves out of line: a large
    /// object, and a nursery too full to serve the bump.
    ///
    /// # Safety
    /// Same contract as [`alloc_with_type_rooted`].
    #[cold]
    #[inline(never)]
    unsafe fn alloc_with_type_rooted_slow(
        &mut self,
        type_id: u32,
        total_size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        // Large objects never trigger a nursery collection, so the native
        // slot needs no temporary registration.
        if total_size > self.config.large_object_threshold {
            unsafe { *needs_write_barrier = true };
            return self.alloc_in_oldgen(type_id, total_size);
        }

        self.pending_reserving_size = total_size;
        unsafe { self.roots.add(root) };
        self.do_collect_nursery();
        self.roots.remove(root);
        self.pending_reserving_size = 0;
        if std::mem::take(&mut self.oom_pending) {
            return GcRef(0);
        }
        let ptr = self.nursery.alloc(total_size);
        let ptr = if ptr.is_null() {
            self.reserve_nursery_gap(total_size)
        } else {
            ptr
        };
        if ptr.is_null() && Self::nursery_allocation_size(total_size) > self.nursery.size() {
            unsafe { *needs_write_barrier = true };
            return self.alloc_in_oldgen(type_id, total_size);
        }
        assert!(
            !ptr.is_null(),
            "collect_and_reserve could not find nursery space for a non-large object"
        );
        self.apply_debug_tiny_nursery();
        self.finish_nursery_object(ptr, type_id)
    }

    /// incminimark.py:946-948, the tail of `collect_and_reserve`:
    ///
    /// ```python
    /// if self.debug_tiny_nursery >= 0:   # for debugging
    ///     if self.nursery_top - self.nursery_free > self.debug_tiny_nursery:
    ///         self.nursery_free = self.nursery_top - self.debug_tiny_nursery
    /// ```
    ///
    /// Runs after the reservation, so the object just handed out kept the
    /// whole arena and only what follows it is rationed.
    #[inline]
    fn apply_debug_tiny_nursery(&mut self) {
        let Some(budget) = self.config.debug_tiny_nursery else {
            return;
        };
        let free = self.nursery.free_ptr() as usize;
        let top = self.nursery.top_ptr() as usize;
        if top.saturating_sub(free) > budget {
            // SAFETY: `top - budget` is above `free`, which the reservation
            // just established is inside the nursery.
            unsafe { self.nursery.set_free_ptr((top - budget) as *mut u8) };
        }
    }

    /// The tail a successful nursery bump runs: `init_gc_object`, then
    /// incminimark.py:687-693's two registrations.
    ///
    /// `FAST` is `malloc_fast` (`framework.py:361-382`), the copy of
    /// `malloc_fixedsize` annotated `s_False, s_False, s_False` — under that
    /// annotation both `if`s constant-fold away, so the common allocation never
    /// reaches the type's flags at all. `gct_fv_gc_malloc`
    /// (`framework.py:830-838`) selects the copy only for a type with no
    /// finalizer, and never passes `contains_weakptr=True` for a fixed-size
    /// malloc: a WEAKREF is built by `gct_weakref_create` instead. The
    /// obligation rides with the caller here too, so the `ll_assert` the
    /// general body spells out becomes a debug assertion.
    #[inline]
    fn finish_bumped_nursery_object<const FAST: bool>(
        &mut self,
        ptr: *mut u8,
        type_id: u32,
    ) -> GcRef {
        if FAST {
            debug_assert!(
                !self.type_needs_young_registration(type_id),
                "malloc_fast served a type that needs a young weakref/destructor registration"
            );
            Self::init_nursery_object(ptr, type_id);
            return GcRef((ptr as usize) + GcHeader::SIZE);
        }
        self.finish_nursery_object(ptr, type_id)
    }

    /// The `malloc_fast` precondition, as the debug build checks it.
    fn type_needs_young_registration(&self, type_id: u32) -> bool {
        if (type_id as usize) >= self.types.len() {
            return false;
        }
        let info = self.types.get(type_id);
        info.is_weakref || info.destructor.is_some()
    }

    /// The tail every non-`malloc_fast` nursery bump shares: `init_gc_object`
    /// plus the weakref and destructor registration of incminimark.py:687-693.
    #[inline]
    fn finish_nursery_object(&mut self, ptr: *mut u8, type_id: u32) -> GcRef {
        Self::init_nursery_object(ptr, type_id);
        let obj = GcRef((ptr as usize) + GcHeader::SIZE);
        self.register_young_object_if_needed(type_id, obj.0);
        obj
    }

    /// incminimark.py:687-693's pair of `if`s, reached the way upstream reaches
    /// them: with both flags already in hand.
    ///
    /// A WEAKREF (`T_IS_WEAKREF` in its TYPE_INFO) joins the young-weakref list
    /// so the next minor collection can invalidate the single `weakptr` slot at
    /// `weakref::WEAKPTR_OFFSET` inside the payload (gctypelayout.py:592) if its
    /// target dies. A type carrying a lightweight `TypeInfo.destructor` joins
    /// the young-destructor list, so that collection either runs the destructor
    /// or promotes the entry to `old_objects_with_destructors`.
    /// `obj_addr` is the payload base (post-header) in both cases.
    ///
    /// `malloc_fixedsize` takes `needs_finalizer` and `contains_weakptr` as
    /// arguments, so its tail costs two tests on values the caller supplied —
    /// and in the `malloc_fast` copy (`framework.py:371-382`, annotated
    /// `s_False, s_False, s_False`) both fold away entirely. pyre resolves them
    /// from the type table instead, which is one row lookup, not two: probing
    /// separately repeats the bound test, the table base load and the stride
    /// multiply that the first probe already performed.
    #[inline]
    fn register_young_object_if_needed(&mut self, type_id: u32, obj_addr: usize) {
        if (type_id as usize) >= self.types.len() {
            return;
        }
        let info = self.types.get(type_id);
        let (is_weakref, has_destructor) = (info.is_weakref, info.destructor.is_some());
        if is_weakref {
            self.young_objects_with_weakrefs.push(obj_addr);
        }
        if has_destructor {
            self.young_objects_with_destructors.push(obj_addr);
        }
    }

    /// incminimark.py:865-930 `collect_and_reserve` pinned-barrier walk.
    ///
    /// A minor collection can leave `nursery_free` after the highest pinned
    /// object, where the tail is too short for the triggering allocation even
    /// though an earlier gap is large enough. Upstream walks the ordered
    /// `nursery_barriers` and reserves from the first fitting gap; it never
    /// changes a non-large malloc into an old-generation allocation. Preserve
    /// that young-result contract because the GC rewrite elides write barriers
    /// while initializing a fresh nursery object (`rewrite.py:911`).
    fn nursery_allocation_size(total_size: usize) -> usize {
        total_size
            .max(GcHeader::MIN_NURSERY_OBJ_SIZE)
            .checked_add(7)
            .map(|size| size & !7)
            .unwrap_or(usize::MAX)
    }

    /// `incminimark.py:978-983 external_malloc` parity: both the variable
    /// portion and the final payload sum are overflow-checked.  The caller
    /// turns `None` into NULL so compiled code raises `MemoryError`.
    fn checked_varsize_payload_size(
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> Option<usize> {
        item_size
            .checked_mul(length)
            .and_then(|var_size| base_size.checked_add(var_size))
    }

    fn reserve_nursery_gap(&mut self, total_size: usize) -> *mut u8 {
        if self.pinned_objects.is_empty() {
            return std::ptr::null_mut();
        }

        let aligned_size = Self::nursery_allocation_size(total_size);
        let nursery_start = self.nursery.start_ptr() as usize;
        let nursery_end = nursery_start + self.nursery.size();
        let mut barriers = Vec::with_capacity(self.pinned_objects.len());
        for &obj_addr in &self.pinned_objects {
            let type_id = unsafe { (*header_of(obj_addr)).type_id() };
            let payload_size = self.size_for_typeid(obj_addr, type_id, "pinned_barriers");
            let object_size = Self::nursery_allocation_size(GcHeader::SIZE + payload_size);
            barriers.push((obj_addr - GcHeader::SIZE, object_size));
        }
        barriers.sort_unstable_by_key(|&(header_start, _)| header_start);

        let mut gap_start = nursery_start;
        for (header_start, object_size) in barriers {
            if header_start.saturating_sub(gap_start) >= aligned_size {
                unsafe {
                    self.nursery.set_free_ptr(gap_start as *mut u8);
                    self.nursery.set_top_ptr(header_start as *const u8);
                }
                self.refresh_published_nursery_top();
                return self.nursery.alloc(total_size);
            }
            gap_start = gap_start.max(header_start.saturating_add(object_size));
        }
        if nursery_end.saturating_sub(gap_start) >= aligned_size {
            unsafe {
                self.nursery.set_free_ptr(gap_start as *mut u8);
                self.nursery.set_top_ptr(nursery_end as *const u8);
            }
            self.refresh_published_nursery_top();
            return self.nursery.alloc(total_size);
        }
        std::ptr::null_mut()
    }

    /// Allocate without triggering collection.
    ///
    /// If the nursery cannot satisfy the request, this falls back directly to
    /// old-gen allocation so compiled code can keep running without needing
    /// stack-map-mediated collection.
    pub fn alloc_with_type_no_collect(&mut self, type_id: u32, payload_size: usize) -> GcRef {
        let Some(total_size) = GcHeader::SIZE.checked_add(payload_size) else {
            return GcRef(0);
        };

        if total_size > self.config.large_object_threshold {
            return self.alloc_in_oldgen(type_id, total_size);
        }

        let ptr = self.nursery.alloc(total_size);
        if ptr.is_null() {
            return self.alloc_in_oldgen(type_id, total_size);
        }

        self.finish_nursery_object(ptr, type_id)
    }

    /// Fallible host-side counterpart of `alloc_with_type_no_collect`.
    ///
    /// The ordinary nursery bump remains allocation-free and therefore cannot
    /// fail. Oversized objects and nursery-full spill use rawmalloc and return
    /// NULL on failure, matching RPython's `MemoryError` path.
    pub fn try_alloc_with_type_no_collect(&mut self, type_id: u32, payload_size: usize) -> GcRef {
        let mut needs_write_barrier = true;
        unsafe {
            self.try_alloc_with_type_no_collect_with_placement(
                type_id,
                payload_size,
                &mut needs_write_barrier,
            )
        }
    }

    /// [`try_alloc_with_type_no_collect`](Self::try_alloc_with_type_no_collect)
    /// for a type that carries neither a finalizer nor the weakref flag:
    /// `malloc_fast` (`framework.py:361-382`).
    ///
    /// # Safety
    /// `type_id` must name a type with no destructor and no weakref flag.
    pub unsafe fn try_alloc_fast_with_type_no_collect(
        &mut self,
        type_id: u32,
        payload_size: usize,
    ) -> GcRef {
        let mut needs_write_barrier = true;
        unsafe {
            self.try_alloc_fast_with_type_no_collect_with_placement(
                type_id,
                payload_size,
                &mut needs_write_barrier,
            )
        }
    }

    /// Placement-reporting counterpart of
    /// [`try_alloc_with_type_no_collect`](Self::try_alloc_with_type_no_collect).
    ///
    /// `framework.py:28-61` propagates `no_write_barrier_needed` through a
    /// fresh nursery allocation. The no-collect path can instead spill to
    /// old-gen, so report that exceptional placement to the initializer.
    ///
    /// Split like [`alloc_with_type_rooted`](Self::alloc_with_type_rooted):
    /// the bump inlines into the caller, the old-gen spill is a call.
    ///
    /// # Safety
    /// `needs_write_barrier` must point to a valid mutable `bool` slot for the
    /// duration of this call.
    #[inline]
    pub unsafe fn try_alloc_with_type_no_collect_with_placement(
        &mut self,
        type_id: u32,
        payload_size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.try_alloc_with_type_no_collect_body::<false>(
                type_id,
                payload_size,
                needs_write_barrier,
            )
        }
    }

    /// [`try_alloc_with_type_no_collect_with_placement`](Self::try_alloc_with_type_no_collect_with_placement)
    /// for a type that carries neither a finalizer nor the weakref flag:
    /// `malloc_fast` (`framework.py:361-382`).
    ///
    /// # Safety
    /// Same contract as
    /// [`try_alloc_with_type_no_collect_with_placement`](Self::try_alloc_with_type_no_collect_with_placement),
    /// plus `type_id` must name a type with no destructor and no weakref flag.
    #[inline]
    pub unsafe fn try_alloc_fast_with_type_no_collect_with_placement(
        &mut self,
        type_id: u32,
        payload_size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.try_alloc_with_type_no_collect_body::<true>(
                type_id,
                payload_size,
                needs_write_barrier,
            )
        }
    }

    /// # Safety
    /// See
    /// [`try_alloc_with_type_no_collect_with_placement`](Self::try_alloc_with_type_no_collect_with_placement);
    /// `FAST` additionally carries `malloc_fast`'s obligation on `type_id`.
    #[inline]
    unsafe fn try_alloc_with_type_no_collect_body<const FAST: bool>(
        &mut self,
        type_id: u32,
        payload_size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe { *needs_write_barrier = true };
        let Some(total_size) = GcHeader::SIZE.checked_add(payload_size) else {
            return GcRef(0);
        };

        if total_size <= self.config.large_object_threshold {
            let ptr = self.nursery.alloc(total_size);
            if !ptr.is_null() {
                let obj = self.finish_bumped_nursery_object::<FAST>(ptr, type_id);
                unsafe { *needs_write_barrier = false };
                return obj;
            }
        }
        self.spill_to_oldgen_or_null(type_id, total_size)
    }

    /// The old-gen spill
    /// [`try_alloc_with_type_no_collect_with_placement`](Self::try_alloc_with_type_no_collect_with_placement)
    /// leaves out of line, for a large object or a nursery with no room.
    #[cold]
    #[inline(never)]
    fn spill_to_oldgen_or_null(&mut self, type_id: u32, total_size: usize) -> GcRef {
        self.try_alloc_in_oldgen(type_id, total_size)
            .unwrap_or(GcRef(0))
    }

    /// Run the lightweight destructor registered for `obj_addr`'s type,
    /// if any. incminimark.py `call_destructor` analog: looks up the
    /// type's `TypeInfo.destructor` and calls it on the payload base.
    ///
    /// # Safety
    /// `obj_addr` must point at a live (not-yet-freed) object payload
    /// whose header still names a registered type id — true for a dead
    /// nursery object before `nursery.reset()` and a dying old-gen object
    /// before the first incremental old-gen sweep step.
    fn run_destructor(&self, obj_addr: usize) {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        if (type_id as usize) >= self.types.len() {
            return;
        }
        if let Some(destructor) = self.types.get(type_id).destructor {
            unsafe { destructor(obj_addr) };
        }
    }

    /// Initialize a nursery object's header.
    fn init_nursery_object(header_ptr: *mut u8, type_id: u32) {
        let hdr = unsafe { &mut *(header_ptr as *mut GcHeader) };
        // Young objects do NOT have TRACK_YOUNG_PTRS (we assume any
        // young object can point to any other young object).
        *hdr = GcHeader::new(type_id);
    }

    /// Flags for an object born directly into the old generation, adding
    /// `flags::VISITED` while a major marking cycle is in progress.
    ///
    /// An object allocated straight into the old generation during MARKING (a
    /// large object or nursery-full fallback) enters the still-active page/raw
    /// lists that will be frozen at the MARKING->SWEEPING seam. The root scan has
    /// already happened, so allocate it black; allocations during SWEEPING need
    /// no VISITED workaround because incminimark.py:2513-2514 and :2688-2694's
    /// list swaps keep them outside this cycle's candidates. Any young pointer
    /// later stored into a born-black object re-enters marking through the write
    /// barrier's remembered-set rescan.
    #[inline]
    fn oldgen_birth_flags(&self, base: u64) -> u64 {
        if self.gc_state == GcState::Marking {
            base | flags::VISITED
        } else {
            base
        }
    }

    /// Walk everything reachable from the roots at the end of a minor and
    /// report each traced slot that still names an unpinned nursery address.
    ///
    /// Reachability is what makes the answer usable: a block the mutator
    /// abandoned — a grown-out-of mapdict storage, a replaced items block —
    /// keeps whatever it last held until the sweep frees it, and those stale
    /// words are not violations. Only declared slots are read, so a word of
    /// padding or a non-reference field that happens to fall inside the
    /// nursery range cannot be mistaken for one either.
    ///
    /// Returns `(holder, slot offset, value, parent)` per bad slot, where the
    /// parent is the object the holder was first reached through: a block with
    /// no header of its own is traced only via its owner's custom trace, so
    /// the owner is what the remembered set has to carry.
    fn bh_probe_stale_young_slots(&self) -> Vec<(usize, usize, usize, Option<usize>)> {
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut stale: Vec<(usize, usize, usize, Option<usize>)> = Vec::new();
        let mut pending: Vec<(GcRef, Option<usize>)> = self
            .enumerate_all_root_values()
            .into_iter()
            .map(|gcref| (gcref, None))
            .collect();
        while let Some((gcref, parent)) = pending.pop() {
            if gcref.is_null()
                || !self.is_managed_heap_object(gcref.0)
                || self.nursery.contains(gcref.0)
                || !seen.insert(gcref.0)
            {
                continue;
            }
            let here = gcref.0;
            self.visit_referent_slots(here, &mut |slot| {
                let value = unsafe { *slot }.0;
                if value == 0 {
                    return;
                }
                if self.nursery.contains(value) && !self.pinned_objects.contains(&value) {
                    stale.push((here, slot as usize - here, value, parent));
                    return;
                }
                pending.push((GcRef(value), Some(here)));
            });
        }
        stale
    }

    /// TEMPORARY DIAGNOSTIC (`MAJIT_GC_BH_PROBE`), run before a minor starts.
    ///
    /// A minor traces the roots and the remembered set and nothing else, so an
    /// old-generation object holding a young reference while off that set has
    /// its slot left behind: the value moves and the slot keeps the dead
    /// address. That is a missing write barrier, and the moment before the
    /// collection is the only one at which the evidence still exists — after
    /// it the slot is indistinguishable from one that was never written.
    ///
    /// Panics naming the holder, the slot, and the object the holder was first
    /// reached through, since a block with no header of its own is traced only
    /// through its owner and it is the owner that has to carry the barrier.
    fn bh_probe_check_barriers_before_minor(&mut self) {
        if !crate::bh_probe_enabled()
            || self.minor_collections < read_uint_from_env("MAJIT_GC_BH_PROBE_FROM").unwrap_or(0)
        {
            return;
        }
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut pending: Vec<(GcRef, Option<usize>)> = self
            .enumerate_all_root_values()
            .into_iter()
            .map(|gcref| (gcref, None))
            .collect();
        let mut bad: Vec<String> = Vec::new();
        while let Some((gcref, parent)) = pending.pop() {
            if gcref.is_null() || !self.is_managed_heap_object(gcref.0) || !seen.insert(gcref.0) {
                continue;
            }
            let here = gcref.0;
            // A young holder needs no barrier: the minor traces the nursery in
            // full, so only an old-generation slot can be left behind.
            let holder_is_old = self.oldgen.contains(here);
            let remembered = self.remembered_set.contains(&here);
            let parent_remembered = parent.is_some_and(|p| self.remembered_set.contains(&p));
            self.visit_referent_slots(here, &mut |slot| {
                let value = unsafe { *slot }.0;
                if value == 0 {
                    return;
                }
                // Two ways a traced slot can already be wrong here: it names a
                // managed address whose header is not a type at all (the value
                // died and its memory was handed out again), or it names a
                // young object from an old holder the barrier never remembered.
                let managed = self.nursery.contains(value) || self.oldgen.contains(value);
                let bad_target = managed
                    && (unsafe { (*header_of(value)).type_id() } as usize) >= self.types.len();
                let unbarriered = holder_is_old
                    && !remembered
                    && !parent_remembered
                    && self.nursery.contains(value)
                    && !self.pinned_objects.contains(&value);
                if bad_target || unbarriered {
                    let tid = unsafe { (*header_of(here)).type_id() };
                    let parent_tid = parent.map(|p| unsafe { (*header_of(p)).type_id() });
                    bad.push(format!(
                        "{} holder={here:#x} tid={tid} old={holder_is_old} slot={:#x} \
                         offset={} value={value:#x} young={} track_young={} remembered={} \
                         barriered_ever={} | parent={:#x} tid={parent_tid:?} remembered={} \
                         barriered_ever={}",
                        if bad_target {
                            "BAD-TARGET"
                        } else {
                            "UNBARRIERED"
                        },
                        slot as usize,
                        slot as usize - here,
                        self.nursery.contains(value),
                        unsafe { (*header_of(here)).has_flag(flags::TRACK_YOUNG_PTRS) },
                        remembered,
                        crate::bh_probe_was_barriered(here),
                        parent.unwrap_or(0),
                        parent_remembered,
                        parent.is_some_and(crate::bh_probe_was_barriered),
                    ));
                }
                if !bad_target {
                    pending.push((GcRef(value), Some(here)));
                }
            });
        }
        // The root walk only covers what `enumerate_all_root_values` names, and
        // an object the mutator is writing to is live whether or not that
        // enumeration reaches it. Sweep every recorded old-generation block of
        // the type under investigation instead, since the population that
        // matters is "old holders with a young reference and no barrier".
        // The remembered set is traced whether or not anything still points at
        // its entries, so it is its own population: an entry the root walk
        // never reached is a dead object the minor will still read.
        for index in 0..self.remembered_set.len() {
            let here = self.remembered_set[index];
            let root_reachable = seen.contains(&here);
            self.visit_referent_slots(here, &mut |slot| {
                let value = unsafe { *slot }.0;
                if value == 0 || !(self.nursery.contains(value) || self.oldgen.contains(value)) {
                    return;
                }
                if (unsafe { (*header_of(value)).type_id() } as usize) < self.types.len() {
                    return;
                }
                let tid = unsafe { (*header_of(here)).type_id() };
                bad.push(format!(
                    "REMEMBERED-BAD-TARGET holder={here:#x} tid={tid} \
                     root_reachable={root_reachable} slot={:#x} offset={} value={value:#x} \
                     young={} barriered_ever={}",
                    slot as usize,
                    slot as usize - here,
                    self.nursery.contains(value),
                    crate::bh_probe_was_barriered(here),
                ));
            });
        }
        if !bad.is_empty() {
            panic!(
                "BH PROBE: {} unbarriered old->young slot(s) before minor #{}\n  {}",
                bad.len(),
                self.minor_collections,
                bad.join("\n  ")
            );
        }
    }

    /// TEMPORARY DIAGNOSTIC (`MAJIT_GC_BH_PROBE`).
    ///
    /// At the end of a minor every live nursery object has been forwarded and
    /// every traced slot rewritten, so no surviving object may still name a
    /// nursery address. Accumulate the distinct `(type, offset)` classes of
    /// those that do, then panic once with the whole table — the wasm runner
    /// recovers a guest panic message out of linear memory, so this is the one
    /// diagnostic channel that reaches the console from inside the module.
    fn bh_probe_check_no_young_refs(&mut self) {
        if !crate::bh_probe_enabled() {
            return;
        }
        // How many distinct classes to gather before reporting, and how many
        // minors to keep scanning if fewer than that ever appear. Both are
        // overridable, because the interpreter fills the old generation long
        // before the JIT compiles anything: reporting at minor #1 can only ever
        // show pre-JIT allocations.
        let class_budget = read_uint_from_env("MAJIT_GC_BH_PROBE_CLASSES").unwrap_or(10);
        let report_at_minor = read_uint_from_env("MAJIT_GC_BH_PROBE_MINOR").unwrap_or(120);
        // The reachability walk below is a whole-heap traversal per minor, far
        // too slow to leave on for a whole run; skip the minors before the one
        // being investigated.
        if self.minor_collections < read_uint_from_env("MAJIT_GC_BH_PROBE_FROM").unwrap_or(0) {
            return;
        }
        const WORD: usize = std::mem::size_of::<usize>();
        // The recorded allocation size and origin of each old-generation block,
        // which the walk below does not carry.
        let blocks: std::collections::HashMap<usize, (usize, u8, &'static str)> =
            crate::with_bh_objects(|objects| {
                objects
                    .iter()
                    .map(|&(addr, payload_size, origin, phase)| {
                        (addr, (payload_size, origin, phase))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut fresh: Vec<crate::BhProbeViolation> = Vec::new();
        {
            for (addr, offset, word, parent) in self.bh_probe_stale_young_slots() {
                let (payload_size, origin, phase) =
                    blocks
                        .get(&addr)
                        .copied()
                        .unwrap_or((0, crate::BH_PROBE_ORIGIN_BORN_OLD, "?"));
                let tid = unsafe { (*header_of(addr)).type_id() };
                if (tid as usize) >= self.types.len() || crate::bh_probe_tid_ignored(tid) {
                    continue;
                }
                {
                    if crate::bh_probe_violation_seen(tid, offset, origin) {
                        continue;
                    }
                    let info = self.types.get(tid);
                    fresh.push(crate::BhProbeViolation {
                        minor: self.minor_collections,
                        origin,
                        phase,
                        holder: addr,
                        tid,
                        payload_size,
                        offset,
                        value: word,
                        forwarded: unsafe { (*header_of(word)).is_forwarded() },
                        type_size: info.size,
                        item_size: info.item_size,
                        length_offset: info.length_offset,
                        gc_ptr_offsets: info.gc_ptr_offsets.to_vec(),
                        items_have_gc_ptrs: info.items_have_gc_ptrs,
                        custom_trace: info.custom_trace.is_some(),
                        is_object: info.is_object,
                        // Only an `rclass.OBJECT` layout has a type pointer in
                        // its first word, and only a *registered* vtable is
                        // safe to dereference — a born-old block the allocator
                        // has handed out but its caller has not filled yet
                        // carries whatever the zero-fill left there.
                        holder_name: if info.is_object
                            && self
                                .vtable_to_type_id
                                .contains_key(&unsafe { *(addr as *const usize) })
                        {
                            crate::bh_probe_type_name(addr)
                        } else {
                            None
                        },
                        // The value moved, so its copy is a live object whose
                        // type names the field far better than an offset does.
                        value_name: {
                            let hdr = unsafe { header_of(word) };
                            let moved = unsafe { (*hdr).is_forwarded() }
                                .then(|| unsafe { GcHeader::forwarding_address(hdr) });
                            moved
                                .filter(|&m| {
                                    self.vtable_to_type_id
                                        .contains_key(&unsafe { *(m as *const usize) })
                                })
                                .and_then(crate::bh_probe_type_name)
                        },
                        neighbourhood: {
                            let lo = offset.saturating_sub(2 * WORD);
                            let hi = (offset + 3 * WORD).min(payload_size.max(offset + WORD));
                            (lo..hi)
                                .step_by(WORD)
                                .map(|o| (o, unsafe { *((addr + o) as *const usize) }))
                                .collect()
                        },
                        track_young_ptrs: unsafe {
                            (*header_of(addr)).has_flag(flags::TRACK_YOUNG_PTRS)
                        },
                        remembered: self.remembered_set.contains(&addr),
                        barriered_ever: crate::bh_probe_was_barriered(addr),
                        traced_this_minor: crate::bh_probe_was_traced(addr),
                        store_sites: crate::bh_probe_store_sites(addr, offset),
                        parent,
                        parent_tid: parent.map(|p| unsafe { (*header_of(p)).type_id() }),
                        parent_name: parent
                            .filter(|&p| {
                                self.vtable_to_type_id
                                    .contains_key(&unsafe { *(p as *const usize) })
                            })
                            .and_then(crate::bh_probe_type_name),
                        parent_remembered: parent.is_some_and(|p| self.remembered_set.contains(&p)),
                        parent_barriered_ever: parent.is_some_and(crate::bh_probe_was_barriered),
                        parent_traced_this_minor: parent.is_some_and(crate::bh_probe_was_traced),
                    });
                }
            }
        }
        let total = crate::bh_probe_record_violations(fresh);
        if total >= class_budget || (total > 0 && self.minor_collections >= report_at_minor) {
            panic!(
                "{}",
                crate::bh_probe_violation_report(self.minor_collections)
            );
        }
    }

    /// Allocate directly in old gen (for large objects or post-collection fallback).
    fn alloc_in_oldgen(&mut self, type_id: u32, total_size: usize) -> GcRef {
        let ptr = self.oldgen.alloc(total_size);
        self.finish_alloc_in_oldgen(type_id, total_size, ptr)
    }

    /// Fallible old-gen allocation used by host allocation hooks. Upstream
    /// rawmalloc failure returns NULL so the caller can raise `MemoryError`.
    fn try_alloc_in_oldgen(&mut self, type_id: u32, total_size: usize) -> Option<GcRef> {
        let ptr = self.oldgen.try_alloc(total_size)?;
        Some(self.finish_alloc_in_oldgen(type_id, total_size, ptr))
    }

    fn finish_alloc_in_oldgen(&mut self, type_id: u32, total_size: usize, ptr: *mut u8) -> GcRef {
        if crate::bh_probe_enabled() {
            let lo = self.nursery.start_ptr() as usize;
            crate::BH_PROBE_NURSERY_LO.store(lo, std::sync::atomic::Ordering::Relaxed);
            crate::BH_PROBE_NURSERY_HI.store(
                lo + self.nursery.size(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        // `do_malloc_fixedsize_clear` and the resume.py direct reader both
        // require a zero-filled payload.  In particular, resume
        // materialization writes only fields present in resumedata; omitted
        // PyFrame owned-content fields must remain null for its destructor.
        // Keep this rare born-old contract here: ArenaCollection.malloc and
        // nursery promotion's alloc-and-copy path stay uninitialized.
        unsafe {
            std::ptr::write_bytes(
                ptr.add(GcHeader::SIZE),
                0,
                total_size.saturating_sub(GcHeader::SIZE),
            );
        }
        let hdr = unsafe { &mut *(ptr as *mut GcHeader) };
        // Old objects start with TRACK_YOUNG_PTRS set (they need write barrier).
        // An object born into the old generation while a major marking cycle is
        // in progress must also be allocated black (see `oldgen_birth_flags`).
        *hdr = GcHeader::with_flags(type_id, self.oldgen_birth_flags(flags::TRACK_YOUNG_PTRS));
        self.bytes_made_old_since_cycle =
            self.bytes_made_old_since_cycle.saturating_add(total_size);
        let obj_addr = (ptr as usize) + GcHeader::SIZE;
        if crate::gc_lifetime_log_enabled() {
            // Pairs with `[gc][free]`: whether a dangling reference names an
            // object freed after the referrer was born, or one already dead
            // when the referrer took it, decides between a missed marking edge
            // and a stale cached pointer, and the free line alone cannot say.
            eprintln!(
                "[gc][alloc] addr={obj_addr:#x} type_id={type_id} kind=oldgen state={:?}",
                self.gc_state
            );
        }
        if (type_id as usize) < self.types.len() {
            let info = self.types.get(type_id);
            // A destructor-bearing object that never passes through the
            // nursery (large object, or nursery-full fallback) is recorded
            // straight onto the old-destructor list so a later major
            // collection runs its destructor when it dies.
            if info.destructor.is_some() {
                self.old_objects_with_destructors.push(obj_addr);
            }
            // A weakref born directly in the old generation (large object or
            // nursery-full fallback) skips the young registration path, so
            // record it straight onto the old-weakref list. Without this its
            // `weakptr` is never invalidated and `weakref()` returns a
            // dangling pointer once the target dies.
            if info.is_weakref {
                self.old_objects_with_weakrefs.push(obj_addr);
            }
        }
        crate::note_bh_object(
            obj_addr,
            total_size - GcHeader::SIZE,
            crate::BH_PROBE_ORIGIN_BORN_OLD,
        );
        // external_malloc (incminimark.py:987-994) tests the same threshold
        // here and drives `minor_collection_with_major_progress` before
        // handing the block back. Collecting at this point is what pyre cannot
        // do: the caller is holding the raw pointer on the Rust stack, which
        // is not a root, and so is whatever else it had live. Ask the question
        // where upstream asks it and defer only the answer — the request rides
        // the eval-breaker word to the interpreter dispatch loop, where the
        // frame walker sees the whole root set.
        //
        // Only where that walk will actually happen: the threshold stays
        // reached until a major completes, so arming past a consumer that
        // refuses re-arms on every following allocation.
        // See [`DEFERRED_MAJOR_REQUEST_PROBE`].
        if self.threshold_reached(total_size) && deferred_major_request_wanted() {
            majit_ir::eval_breaker_word::set_gc();
        }
        GcRef(obj_addr)
    }

    /// Perform a minor (nursery) collection.
    ///
    /// 1. Scan roots: copy referenced nursery objects to old gen.
    /// 2. Process remembered set: copy nursery objects referenced by old-gen objects.
    /// 3. Iteratively process newly discovered references until stable.
    /// 4. Reset nursery.
    pub fn do_collect_nursery(&mut self) {
        let start = GcClock::start();
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let walk_all_mutators = crate::gc_sync::mutators_quiesced();
        if crate::majit_log_enabled() {
            eprintln!(
                "[gc][minor] start count={} remembered={} cards_set={}",
                self.minor_collections + 1,
                self.remembered_set.len(),
                self.old_objects_with_cards_set.len(),
            );
        }
        self.minor_collections += 1;
        self.bh_probe_check_barriers_before_minor();
        // `bytes_made_old_since_cycle` is the running sum of every
        // `copy_nursery_object` payload, so its delta across this collection is
        // exactly what was promoted out of the nursery. The major-collection
        // paths that reset it all run after the sample is taken below.
        let drain_sample = crate::drain_census_enabled()
            .then(|| (self.nursery.used(), self.bytes_made_old_since_cycle));
        // incminimark.py:1816: this is the survivor count for this minor only;
        // every promotion below adds its allocator-sized object extent.
        self.nursery_surviving_size = 0;
        // incminimark.py:1779-1785: pinning does not keep an object alive.
        // Rebuild both the survivor set and count from traced edges below.
        self.surviving_pinned_objects.clear();
        self.pinned_objects_in_nursery = 0;
        // incminimark.py:1800-1807: a black old parent may expose an unpinned
        // child that will move during this minor, so make the parent gray
        // again before the active major marking cycle can sweep that child.
        if self.gc_state == GcState::Marking {
            for &obj_addr in &self.old_objects_pointing_to_pinned {
                let hdr = unsafe { header_of(obj_addr) };
                if unsafe { (*hdr).has_flag(flags::VISITED) } {
                    self.incr_state.more_gray_stack.push(obj_addr);
                }
            }
        }
        // incminimark.py:1826-1832: replace the list before anything can append
        // to it, so parents discovered during this minor accumulate in the
        // fresh one instead of in the copy being drained. Upstream performs the
        // swap after `collect_roots_in_nursery` because its root callback
        // passes a NULL parent and therefore records nothing; the
        // old-generation jitframe arm of Phase 1c below traces with a real
        // parent, so here the swap has to come first. A parent recorded after
        // the swap keeps `GCFLAG_PINNED_OBJECT_PARENT_KNOWN` for the rest of
        // the minor and would not be re-recorded when the drained copy is
        // visited, which would drop it from the list permanently and leave the
        // flag set for good.
        let old_parents_pointing_to_pinned =
            std::mem::take(&mut self.old_objects_pointing_to_pinned);
        crate::bh_probe_clear_traced();
        // Phase 1: Process roots — copy nursery objects they point to.
        // We use raw pointers to avoid borrow checker issues since
        // copy_nursery_object mutates oldgen/nursery.
        // Pinned objects are left in place (not copied to old gen).
        let roots: Vec<*mut GcRef> = self.roots.roots.to_vec();
        for root_ptr in roots {
            let gcref = unsafe { &mut *root_ptr };
            self.drag_out_root(gcref);
        }

        // Phase 1b: Process shadow stack roots.
        // RPython gc.py: GcRootMap_shadowstack — walk the thread-local
        // shadow stack to find GC refs pushed by compiled JIT code.
        let mut visit_shadow_root = |gcref: &mut GcRef| {
            self.drag_out_root(gcref);
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_roots(&mut visit_shadow_root);
        } else {
            crate::shadow_stack::walk_roots(&mut visit_shadow_root);
        }

        // Phase 1c: Process jitframe shadow stack roots.
        // RPython root_walker.walk_roots with jitframe entries —
        // reads jf_gcmap from each jitframe and traces ref slots.
        // assembler.py:1122 (_call_header_shadowstack) pushes jf_ptr;
        // callbuilder.py:93 (push_gcmap) writes per-call gcmap to jf_gcmap.
        // Collect libc-jitframe slots so we can traverse/update them
        // after the walk finishes without reborrowing `self` inside the
        // tracer callback.
        let mut libc_jf_slots: Vec<*mut majit_ir::GcRef> = Vec::new();
        let mut visit_jf_root = |gcref: &mut GcRef| {
            if self.is_nursery_object_start(gcref.0) {
                self.drag_out_root(gcref);
            } else if !gcref.is_null() && self.oldgen.contains(gcref.0) {
                // RPython parity: old-gen jitframes need their interior
                // nursery refs traced directly. The custom_trace hook
                // walks gcmap bits to find Ref slots.
                self.trace_and_update_object(gcref.0, "minor_jitframe_root");
            } else if !gcref.is_null() && crate::shadow_stack::is_libc_jitframe(gcref.0) {
                // pyre dynasm extension: jitframes allocated via
                // `libc::calloc` in execute_token are neither in the
                // nursery nor the old gen. The registered libc-jitframe
                // tracer walks `jf_gcmap` bits to expose ref slots.
                crate::shadow_stack::trace_libc_jitframe(gcref.0, &mut |slot_ptr| {
                    libc_jf_slots.push(slot_ptr);
                });
            }
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_jf_roots(&mut visit_jf_root);
        } else {
            crate::shadow_stack::walk_jf_roots(&mut visit_jf_root);
        }
        // The jitframes the backend is still holding as DEADFRAMES. A frame
        // reaches this walk exactly when it has left the shadow stack — the
        // compiled epilogue pops it before returning — and has not yet been
        // freed, which is the window in which the frontend reads its slots.
        // `llmodel.py:240-250` reads those slots straight out of the frame, so
        // the interior refs are live for the whole window and nothing else in
        // this phase can see them. See `ActiveGcDeadFrameHooks`.
        crate::walk_active_live_deadframes(&mut |addr| {
            crate::shadow_stack::trace_libc_jitframe(addr, &mut |slot_ptr| {
                libc_jf_slots.push(slot_ptr);
            });
        });
        for slot_ptr in libc_jf_slots {
            let field_ref = unsafe { &mut *slot_ptr };
            self.drag_out_root(field_ref);
        }

        // Phase 1d: Process blackhole interpreter register banks.
        // blackhole.py BlackholeInterpreter.registers_r parity: each
        // active blackhole frame's ref register file is part of the GC
        // root set. RPython traces these via the RPython object graph
        // (Box arrays); pyre stores raw i64 in Vec<i64> so we walk the
        // explicit thread-local stack of register banks.
        let mut visit_bh_root = |gcref: &mut GcRef| {
            self.drag_out_root(gcref);
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_bh_regs(&mut visit_bh_root);
        } else {
            crate::shadow_stack::walk_bh_regs(&mut visit_bh_root);
        }

        // blackhole resume construction roots (`resume.py:1312`): the
        // virtuals_cache + each frame's registers_r are filled by lazily
        // materializing virtuals before `run()` re-roots them via
        // `push_bh_regs`; forward any already-materialized nursery refs so a
        // later materialization's collection does not strand them.
        let mut visit_resume_root = |gcref: &mut GcRef| {
            self.drag_out_root(gcref);
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_resume_ref_roots(&mut visit_resume_root);
        } else {
            crate::shadow_stack::walk_resume_ref_roots(&mut visit_resume_root);
        }

        // Phase 1e: framework.py `root_walker.walk_roots` parity — the
        // embedding runtime plugs a walker that visits
        // `PyFrame.locals_cells_stack_w` across the active f_backref chain
        // so nursery refs held only by a Python frame local or operand-stack
        // slot survive collection.
        //
        // Announce the collection kind so walkers mirroring incminimark's
        // prebuilt-object scanning can skip clean prebuilt structures during
        // a minor collection (incminimark.py:339-344
        // `old_objects_pointing_to_young`); restored to the conservative
        // Major default right after.
        crate::shadow_stack::set_extra_root_walk_kind(
            crate::shadow_stack::ExtraRootWalkKind::Minor,
        );
        let mut visit_extra_area = |gcref: &mut GcRef| {
            self.drag_out_root(gcref);
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_extra_areas(&mut visit_extra_area);
        } else {
            crate::shadow_stack::walk_my_extra_areas(&mut visit_extra_area);
        }
        crate::walk_active_extra_roots(&mut |gcref| {
            self.drag_out_root(gcref);
        });

        // Multi-registrar walker fan-out (rd_consts const-pool, etc.).
        crate::shadow_stack::walk_extra_roots(|gcref| {
            self.drag_out_root(gcref);
        });
        crate::shadow_stack::set_extra_root_walk_kind(
            crate::shadow_stack::ExtraRootWalkKind::Major,
        );

        // incminimark.py:1820-1832: old parents that reached a pinned child in
        // the previous minor must be traced again. `copy_nursery_object`
        // repopulates the list swapped in above, and only for parents that
        // still point to a pinned object.
        for obj_addr in old_parents_pointing_to_pinned {
            self.trace_and_update_object(obj_addr, "minor_old_parent_pinned");
        }

        // incminimark parity: during an active marking cycle, old objects
        // remembered by the write barrier may already be black. Requeue
        // those black objects so the major collector rescans their new
        // outgoing references before sweep.
        if self.gc_state == GcState::Marking {
            let remembered_now: Vec<usize> = self.remembered_set.to_vec();
            for obj_addr in remembered_now {
                let hdr = unsafe { header_of(obj_addr) };
                if unsafe { (*hdr).has_flag(flags::VISITED) } {
                    self.incr_state.more_gray_stack.push(obj_addr);
                }
            }
        }

        // incminimark.py:1834-1836: a mirror the C side still references roots
        // its linked object, so the P list joins the root walk before anything
        // decides which nursery objects die.
        if self.rrc.enabled {
            self.rrc_minor_collection_trace();
        }

        // incminimark.py:1838-1841: registered young finalizers all survive
        // this minor and move to the old-registration deque.
        if !self.probably_young_objects_with_finalizers.is_empty() {
            self.deal_with_young_objects_with_finalizers();
        }

        // incminimark.py:1843-1862: while True loop —
        // collect_cardrefs_to_nursery, then collect_oldrefs_to_nursery.
        // collect_oldrefs_to_nursery may cause new card-marked objects to
        // appear, so we must re-enter if old_objects_with_cards_set is
        // non-empty after processing the remembered set.
        loop {
            // incminimark.py:1846-1847
            if self.card_page_shift > 0 {
                self.collect_cardrefs_to_nursery();
            }

            // incminimark.py:1855: collect_oldrefs_to_nursery.
            let mut idx = 0;
            loop {
                if idx >= self.remembered_set.len() {
                    break;
                }
                let obj_addr = self.remembered_set[idx];
                idx += 1;

                // incminimark.py:2095-2098: re-set TRACK_YOUNG_PTRS.
                unsafe {
                    (*header_of(obj_addr)).set_flag(flags::TRACK_YOUNG_PTRS);
                }

                // Trace this old-gen object's fields and copy any nursery
                // objects they reference.
                self.trace_and_update_object(obj_addr, "minor_remembered_set");
            }
            self.remembered_set.clear();

            // incminimark.py:1859-1862: loop back if card-marked objects appeared.
            if self.card_page_shift > 0 && !self.old_objects_with_cards_set.is_empty() {
                continue;
            }
            break;
        }

        // incminimark.py:1865-1867 — now that every live nursery
        // object has been forwarded out, walk the young weakref
        // list to update or invalidate each WEAKREF's `weakptr` slot.
        if !self.young_objects_with_weakrefs.is_empty() {
            self.invalidate_young_weakrefs();
        }

        // incminimark.py:1868-1869 — then run the destructor of each
        // dead nursery object (and promote survivors to the
        // old-destructor list). Must happen after every live nursery
        // object is forwarded (so survival is detectable) and before the
        // nursery reset below reclaims the backing bytes.
        if !self.young_objects_with_destructors.is_empty() {
            self.deal_with_young_objects_with_destructors();
        }

        // An embedder side table keyed by owner address cannot be left holding
        // a nursery key past this point: the reset below hands that address to
        // the next allocation, which would then answer to the dead owner's
        // entry.  Ask the same survival question `invalidate_young_weakrefs`
        // just answered, while the forwarding headers still read.
        // The trace above rebuilt the live pin set in
        // `surviving_pinned_objects`; `pinned_objects` is the previous
        // collection's snapshot until the swap below.  A pin that died this
        // cycle must not keep an address-keyed owner table entry alive.
        let pinned = &self.surviving_pinned_objects;
        let nursery = &self.nursery;
        let mut classify_young_owner = |owner: usize| -> Option<usize> {
            if owner == 0 || !nursery.contains(owner) {
                return Some(owner);
            }
            // A pinned object is alive and stayed put, so it never forwarded.
            if pinned.contains(&owner) {
                return Some(owner);
            }
            let hdr = (owner - GcHeader::SIZE) as *const GcHeader;
            if unsafe { (*hdr).is_forwarded() } {
                Some(unsafe { GcHeader::forwarding_address(hdr) })
            } else {
                None
            }
        };
        crate::shadow_stack::reconcile_young_owner_tables(&mut classify_young_owner);

        // incminimark.py:1876-1882,1900-1944: replace the pin set with exactly
        // the objects reached this collection, then preserve identity shadows
        // and nursery barriers only for those survivors. Clear their temporary
        // VISITED bit while the addresses are still valid.
        self.pinned_objects = std::mem::take(&mut self.surviving_pinned_objects);
        debug_assert_eq!(self.pinned_objects_in_nursery, self.pinned_objects.len());
        for &obj_addr in &self.pinned_objects {
            unsafe { (*header_of(obj_addr)).clear_flag(flags::VISITED) };
        }

        // Clear this mapping now that every unpinned nursery object was
        // forwarded (or died). Upstream rebuilds it from the same surviving
        // pinned collection via `record_pinned_object_with_shadow`.
        // A pinned object's shadow is its stable identity address and MUST
        // survive the clear, else the next `id_or_identityhash`
        // re-allocates a fresh shadow and the identity hash changes.
        if !self.nursery_objects_shadows.is_empty() {
            if self.pinned_objects.is_empty() {
                self.nursery_objects_shadows.clear();
            } else {
                let pinned = &self.pinned_objects;
                let marking = self.gc_state == GcState::Marking;
                self.nursery_objects_shadows
                    .retain(|obj_addr, shadow_addr| {
                        let keep = pinned.contains(obj_addr);
                        if keep && marking {
                            // incminimark.py:1738-1752
                            // record_pinned_object_with_shadow: during the
                            // marking phase keep the retained shadow black so
                            // the in-progress sweep does not reclaim the
                            // pinned object's reserved identity address.  Safe
                            // because pinned objects hold no gcptrs, so the
                            // shadow needs no further tracing.
                            unsafe {
                                (*header_of(*shadow_addr)).set_flag(flags::VISITED);
                            }
                        }
                        keep
                    });
            }
        }

        // incminimark.py:1886-1888 — every live nursery object has been
        // forwarded by now, so each young mirror's link either follows its
        // object out or reports that the object died.  Before the nursery reset
        // below, which invalidates the addresses this reads.
        if self.rrc.enabled {
            self.rrc_minor_collection_free();
        }

        if let Some((used_before, promoted_before)) = drain_sample {
            crate::drain_census_record(
                used_before,
                self.bytes_made_old_since_cycle
                    .saturating_sub(promoted_before),
                self.pinned_objects.len(),
                self.nursery.size(),
            );
        }
        self.bh_probe_check_no_young_refs();

        // Reset nursery for new allocations, preserving pinned objects.
        if self.pinned_objects.is_empty() {
            self.nursery.reset();
        } else {
            self.reset_nursery_with_pinned();
        }
        // incminimark.py:1949-1951: the PINNED bit is reused on old objects as
        // PINNED_OBJECT_PARENT_KNOWN only within one minor collection.
        for &obj_addr in &self.old_objects_pointing_to_pinned {
            unsafe { (*header_of(obj_addr)).clear_flag(flags::PINNED) };
        }
        self.refresh_published_nursery_top();

        // incminimark.py:1965 `self.root_walker.finished_minor_collection()`,
        // the callback framework.py:135-138 reads out of `_jit2gc`: after the
        // nursery is reset and accounted for, and before the timing and the
        // gc-minor hook below.
        crate::invoke_after_minor_collection_hook();

        // incminimark.py:1962-1974 — report the completed minor before the
        // wrapper advances the incremental major state machine.
        let duration = start.elapsed_secs();
        self.total_gc_time += duration;
        self.hooks.fire_gc_minor(
            duration,
            self.get_total_memory_used(),
            self.pinned_objects_in_nursery,
        );

        // Minor collections must also drive incremental major-collection
        // progress. Like incminimark, take one or more major steps until
        // promoted bytes are back under the current step credit.
        self.run_major_progress_after_minor();

        // incminimark.py:862 — `minor_collection_with_major_progress` is where
        // upstream schedules the drain of whatever this collection queued.
        self.rrc_invoke_callback();
    }

    /// incminimark.py:3058-3105 `invalidate_young_weakrefs(self)`.
    ///
    /// For each WEAKREF object recorded in `young_objects_with_weakrefs`:
    ///   * If the weakref itself was not forwarded out, it died — skip.
    ///   * Otherwise read the `weakptr` slot off the forwarded payload.
    ///     If the target is a nursery object: forward → update slot;
    ///     not forwarded → invalidate slot (set to null). If the target
    ///     is in old-gen, the weakref survives — push onto
    ///     `old_objects_with_weakrefs` for the next major cycle.
    ///
    /// RPython's broader checks for young raw-malloced targets,
    /// pinned-target NULL semantics, and prebuilt-target skipping
    /// have no current pyre analog (no raw-malloc young region, no
    /// pinned-weakref code path, no immortal prebuilt objects with
    /// `GCFLAG_NO_HEAP_PTRS`), so this slice ports the core branch
    /// — the remaining filters are out of scope until the matching
    /// surfaces land.
    fn invalidate_young_weakrefs(&mut self) {
        while let Some(obj_addr) = self.young_objects_with_weakrefs.pop() {
            // incminimark.py:3065-3066: if not forwarded → weakref died.
            let hdr_ptr = (obj_addr - GcHeader::SIZE) as *const GcHeader;
            if !unsafe { (*hdr_ptr).is_forwarded() } {
                continue;
            }
            let new_obj = unsafe { GcHeader::forwarding_address(hdr_ptr) };

            // incminimark.py:3069-3070: pointing_to = (obj + offset)
            //                                            .address[0]
            // pyre's WEAKREF struct keeps `weakptr` at offset 0
            // (gctypelayout.py:592, weakref.rs:WEAKPTR_OFFSET).
            let weakptr_slot = (new_obj + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef;
            let pointing_to = unsafe { (*weakptr_slot).0 };

            // Null targets need no update or follow-up tracking.
            if pointing_to == 0 {
                continue;
            }

            // incminimark.py:3071-3079: pointing_to in nursery.
            if self.is_nursery_object_start(pointing_to) {
                let target_hdr = (pointing_to - GcHeader::SIZE) as *const GcHeader;
                if unsafe { (*target_hdr).is_forwarded() } {
                    let fwd = unsafe { GcHeader::forwarding_address(target_hdr) };
                    unsafe { (*weakptr_slot).0 = fwd };
                    // Target lives in old gen now — weakref also lives.
                    self.old_objects_with_weakrefs.push(new_obj);
                } else {
                    // Target dies; null out the weakptr and drop the
                    // weakref from any further tracking.
                    unsafe { (*weakptr_slot).0 = 0 };
                }
                continue;
            }

            // Target is in old-gen (or a foreign address pyre's GC
            // does not own). Either way, the minor cycle leaves the
            // weakptr untouched and the major cycle gets
            // the chance to invalidate it later.
            if self.oldgen.contains(pointing_to) {
                self.old_objects_with_weakrefs.push(new_obj);
            }
        }
    }

    /// A non-moving major leaves nursery objects in place, so the normal
    /// minor-only `invalidate_young_weakrefs` pass does not run.  A live
    /// nursery weakref may nevertheless point at an old object that this
    /// major is about to sweep.  Clear that edge while old-gen VISITED bits
    /// still distinguish survivors, and leave the nursery bookkeeping list
    /// intact for the next moving minor.
    fn invalidate_young_weakrefs_for_nonmoving_major(&mut self) {
        debug_assert!(self.oldgen_nonmoving_active);
        for obj_addr in self.young_objects_with_weakrefs.iter().copied() {
            let weakref_hdr = unsafe { header_of(obj_addr) };
            if unsafe { !(*weakref_hdr).has_flag(flags::VISITED) } {
                continue;
            }
            let weakptr_slot = (obj_addr + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef;
            let pointing_to = unsafe { (*weakptr_slot).0 };
            if pointing_to == 0 || !self.oldgen.contains(pointing_to) {
                continue;
            }
            let target_hdr = unsafe { header_of(pointing_to) };
            if unsafe {
                !(*target_hdr).has_flag(flags::VISITED)
                    || (*target_hdr).has_flag(flags::FINALIZATION_ORDERING)
            } {
                unsafe { (*weakptr_slot).0 = 0 };
            }
        }
    }

    // ----------
    // RawRefCount — incminimark.py:3157-3409

    /// incminimark.py:3172-3183 `rawrefcount_init`.
    ///
    /// Idempotent, as upstream's `if not self.rrc_enabled` is: whichever of the
    /// embedder's entry points runs first may call it.
    pub fn rawrefcount_init(&mut self, dealloc_trigger: rawrefcount::DeallocTriggerFn) {
        if !self.rrc.enabled {
            self.rrc = rawrefcount::RawRefCount {
                enabled: true,
                dealloc_trigger: Some(dealloc_trigger),
                ..Default::default()
            };
        }
    }

    /// `rrc_enabled` — whether [`rawrefcount_init`](Self::rawrefcount_init) has
    /// run.  Every phase call site is guarded on it.
    pub fn rawrefcount_enabled(&self) -> bool {
        self.rrc.enabled
    }

    /// incminimark.py:3196-3210 `rawrefcount_create_link_pypy`.
    ///
    /// Upstream files a link by two independent questions — which list (young
    /// or old), and which identity table (nursery-keyed or not).  They come
    /// apart only for a young raw-malloced object, which is young but not in
    /// the nursery; there is no young raw-malloc region here, so the two
    /// questions collapse into one and the third combination cannot arise.
    ///
    /// The caller must already have added [`rawrefcount::REFCNT_FROM_PYRE`] to
    /// the mirror's count: that share is what this link is worth, and what
    /// [`Self::_rrc_free`] gives back when the object dies.
    pub fn rawrefcount_create_link_pyre(&mut self, obj: usize, pyobject: usize) {
        debug_assert!(self.rrc.enabled, "rawrefcount.init not called");
        unsafe { (*rawrefcount::pyobj(pyobject)).ob_link = obj };
        if self.is_in_nursery(obj) {
            self.rrc.p_list_young.push(pyobject);
            self.rrc.p_dict_nurs.insert(obj, pyobject);
        } else {
            self.rrc.p_list_old.push(pyobject);
            self.rrc.p_dict.insert(obj, pyobject);
        }
    }

    /// incminimark.py:3212-3221 `rawrefcount_create_link_pyobj`.
    ///
    /// The mirror owns the interpreter object here, so no trace pass roots the
    /// object on the mirror's behalf and there is no identity table to keep.
    pub fn rawrefcount_create_link_pyobj(&mut self, obj: usize, pyobject: usize) {
        debug_assert!(self.rrc.enabled, "rawrefcount.init not called");
        if self.is_in_nursery(obj) {
            self.rrc.o_list_young.push(pyobject);
        } else {
            self.rrc.o_list_old.push(pyobject);
        }
        unsafe { (*rawrefcount::pyobj(pyobject)).ob_link = obj };
        // there is no rrc_o_dict
    }

    /// incminimark.py:3223-3227 `rawrefcount_mark_deallocating`.
    ///
    /// `marker` is not a GC object: it is the sentinel
    /// [`Self::rawrefcount_to_obj`] hands back while a mirror's deallocator
    /// runs, so C code that re-enters and asks for the interpreter object gets
    /// something it can recognise instead of a freed address.  A mirror is off
    /// every list by the time this is called, so no trace pass can mistake the
    /// sentinel for a reference.
    pub fn rawrefcount_mark_deallocating(&mut self, marker: usize, pyobject: usize) {
        debug_assert!(self.rrc.enabled, "rawrefcount.init not called");
        unsafe { (*rawrefcount::pyobj(pyobject)).ob_link = marker };
    }

    /// incminimark.py:3229-3235 `rawrefcount_from_obj`.  Zero when unlinked.
    pub fn rawrefcount_from_obj(&self, obj: usize) -> usize {
        let dct = if self.is_in_nursery(obj) {
            &self.rrc.p_dict_nurs
        } else {
            &self.rrc.p_dict
        };
        dct.get(&obj).copied().unwrap_or(0)
    }

    /// incminimark.py:3237-3239 `rawrefcount_to_obj`.
    pub fn rawrefcount_to_obj(&self, pyobject: usize) -> usize {
        unsafe { (*rawrefcount::pyobj(pyobject)).ob_link }
    }

    /// incminimark.py:3241-3245 `rawrefcount_next_dead`.  Zero when the queue
    /// is empty.
    pub fn rawrefcount_next_dead(&mut self) -> usize {
        if self.rrc.enabled {
            self.rrc.dealloc_pending.pop().unwrap_or(0)
        } else {
            0
        }
    }

    /// incminimark.py:3248-3250 `rrc_invoke_callback`.
    ///
    /// Called from the public collection entry points (incminimark.py:808,
    /// :821, :862), never from inside a phase: the collector is borrowed, so
    /// the callback may only schedule the drain, not perform it.
    fn rrc_invoke_callback(&mut self) {
        if self.rrc.enabled
            && !self.rrc.dealloc_pending.is_empty()
            && let Some(trigger) = self.rrc.dealloc_trigger
        {
            trigger();
        }
    }

    /// incminimark.py:3252-3257 `rrc_minor_collection_trace`.
    ///
    /// Every key in the nursery-keyed table is about to move, so the table is
    /// emptied here and refilled with the survivors' new addresses by
    /// [`Self::_rrc_minor_free`].
    fn rrc_minor_collection_trace(&mut self) {
        self.rrc.p_dict_nurs.clear();
        let young = std::mem::take(&mut self.rrc.p_list_young);
        for &pyobject in &young {
            self._rrc_minor_trace(pyobject);
        }
        self.rrc.p_list_young = young;
    }

    /// incminimark.py:3259-3270 `_rrc_minor_trace`.
    ///
    /// The forwarded address is deliberately not written back to the link:
    /// upstream drags the object out through a scratch cell, and
    /// [`Self::_rrc_minor_free`] then reads the forwarding bit at the *old*
    /// address to decide whether the mirror survives.
    fn _rrc_minor_trace(&mut self, pyobject: usize) {
        let rc = unsafe { (*rawrefcount::pyobj(pyobject)).ob_refcnt };
        if rc == rawrefcount::REFCNT_FROM_PYRE {
            // Nothing but the link references this mirror, so the linked
            // object may die.
            return;
        }
        let mut root = GcRef(unsafe { (*rawrefcount::pyobj(pyobject)).ob_link });
        debug_assert!(!root.is_null(), "a mirror on the P list has a link");
        self.drag_out_root(&mut root);
    }

    /// incminimark.py:3272-3282 `rrc_minor_collection_free`.
    ///
    /// `still_young` collects the entries that survived without leaving the
    /// nursery, which upstream has no case for; see [`Self::_rrc_minor_free`].
    fn rrc_minor_collection_free(&mut self) {
        debug_assert!(self.rrc.p_dict_nurs.is_empty(), "p_dict_nurs not empty 1");
        let mut young = std::mem::take(&mut self.rrc.p_list_young);
        let mut still_young = Vec::new();
        while let Some(pyobject) = young.pop() {
            self._rrc_minor_free(pyobject, RrcList::P, &mut still_young);
        }
        self.rrc.p_list_young = still_young;
        let mut young = std::mem::take(&mut self.rrc.o_list_young);
        let mut still_young = Vec::new();
        while let Some(pyobject) = young.pop() {
            self._rrc_minor_free(pyobject, RrcList::O, &mut still_young);
        }
        self.rrc.o_list_young = still_young;
    }

    /// incminimark.py:3284-3318 `_rrc_minor_free`.
    fn _rrc_minor_free(&mut self, pyobject: usize, list: RrcList, still_young: &mut Vec<usize>) {
        let obj = unsafe { (*rawrefcount::pyobj(pyobject)).ob_link };
        // incminimark.py:3300-3312 handles a young raw-malloced target, which
        // survives without moving and is recognised by a flag rather than a
        // forwarding pointer.  There is no young raw-malloc region here, so a
        // young-list link that is not a nursery address is a filing error.
        debug_assert!(
            self.is_nursery_object_start(obj),
            "a young rawrefcount list holds a non-nursery link"
        );
        let hdr = unsafe { header_of(obj) };
        if unsafe { (*hdr).is_forwarded() } {
            let moved = unsafe { GcHeader::forwarding_address(hdr) };
            unsafe { (*rawrefcount::pyobj(pyobject)).ob_link = moved };
            match list {
                RrcList::P => {
                    // It was keyed in `p_dict_nurs`, which this collection
                    // emptied; at its new address it belongs in `p_dict`.
                    self.rrc.p_dict.insert(moved, pyobject);
                    self.rrc.p_list_old.push(pyobject);
                }
                RrcList::O => self.rrc.o_list_old.push(pyobject),
            }
        } else if self.pinned_objects.contains(&obj) {
            // A pinned object that was reached survives *in place*
            // (`copy_nursery_object` records it and returns the same address),
            // so there is no forwarding pointer and the link already names its
            // final address.  It is still a nursery address, so the entry stays
            // on the young list and in the nursery-keyed table.
            //
            // incminimark.py:3287-3299 reads only the forwarding bit, so a
            // pinned P-linked object reports there as dead.  `pin` refuses any
            // type carrying GC pointers, which is why the combination is rare
            // rather than impossible: a pinned byte buffer with a mirror is
            // exactly what `PyBytes_AsString` over a pinned buffer produces.
            // `pinned_objects` holds this collection's survivors by this point
            // (it was swapped from `surviving_pinned_objects` above), so an
            // unreached pinned object still falls through to the free below.
            if list == RrcList::P {
                self.rrc.p_dict_nurs.insert(obj, pyobject);
            }
            still_young.push(pyobject);
        } else {
            self._rrc_free(pyobject);
        }
    }

    /// incminimark.py:3320-3351 `_rrc_free`.
    ///
    /// The linked object has died, so the interpreter's share of the count goes
    /// away.  incminimark.py:3328-3335's `REFCNT_FROM_PYPY_LIGHT` branch has no
    /// port — nothing here creates a light mirror — and the immortal branch
    /// (:3326) is unreachable, because a count above the link share forces the
    /// linked object alive in both trace passes.
    fn _rrc_free(&mut self, pyobject: usize) {
        let header = rawrefcount::pyobj(pyobject);
        let mut rc = unsafe { (*header).ob_refcnt };
        debug_assert!(
            rc < rawrefcount::REFCNT_IMMORTAL,
            "an immortal mirror reached the rawrefcount free pass"
        );
        debug_assert!(
            rc >= rawrefcount::REFCNT_FROM_PYRE,
            "rawrefcount refcount underflow"
        );
        rc -= rawrefcount::REFCNT_FROM_PYRE;
        unsafe { (*header).ob_link = 0 };
        if rc == 0 {
            // incminimark.py:3339-3349 — a mirror at count 0 cannot sit and
            // wait for its deallocator: some extensions read the raw pointer
            // back and expect the deallocator to have run the moment the count
            // reached 0.  Queue it and leave it at 1, so the drain's own
            // release is what frees it.
            self.rrc.dealloc_pending.push(pyobject);
            rc = 1;
        }
        unsafe { (*header).ob_refcnt = rc };
    }

    /// incminimark.py:3353-3354 `rrc_major_collection_trace`.
    fn rrc_major_collection_trace(&mut self) {
        let old = std::mem::take(&mut self.rrc.p_list_old);
        for &pyobject in &old {
            self._rrc_major_trace(pyobject);
        }
        self.rrc.p_list_old = old;
    }

    /// incminimark.py:3356-3372 `_rrc_major_trace`.
    fn _rrc_major_trace(&mut self, pyobject: usize) {
        let rc = unsafe { (*rawrefcount::pyobj(pyobject)).ob_refcnt };
        if rc == rawrefcount::REFCNT_FROM_PYRE {
            return;
        }
        let obj = unsafe { (*rawrefcount::pyobj(pyobject)).ob_link };
        self.seed_major_root(GcRef(obj), "rrc_major_trace");
        self.drain_gray_stack();
    }

    /// A non-moving major runs with a live nursery and no leading minor
    /// (`do_collect_oldgen_nonmoving`), so `rrc_minor_collection_trace` never
    /// ran and the young P list still holds mirrors whose linked nursery object
    /// is reachable from C alone.  Nothing sweeps the nursery here, but an old
    /// object reachable only *through* such a nursery object would be, so seed
    /// the same roots the moving trace would have.  Nothing moves and no entry
    /// changes list: the young bookkeeping stays for the next moving minor.
    ///
    /// The counterpart for weak references is
    /// [`Self::invalidate_young_weakrefs_for_nonmoving_major`].
    fn rrc_nonmoving_major_trace_young(&mut self) {
        debug_assert!(self.oldgen_nonmoving_active);
        let young = std::mem::take(&mut self.rrc.p_list_young);
        for &pyobject in &young {
            let rc = unsafe { (*rawrefcount::pyobj(pyobject)).ob_refcnt };
            if rc == rawrefcount::REFCNT_FROM_PYRE {
                continue;
            }
            let obj = unsafe { (*rawrefcount::pyobj(pyobject)).ob_link };
            self.seed_major_root(GcRef(obj), "rrc_nonmoving_major_young");
            self.drain_gray_stack();
        }
        self.rrc.p_list_young = young;
    }

    /// incminimark.py:3374-3392 `rrc_major_collection_free`.
    ///
    /// Only the old half is rebuilt.  The nursery-keyed table belongs to the
    /// minor that will next empty it, and every entry it holds names a mirror
    /// still on the young list, which this pass does not walk.
    fn rrc_major_collection_free(&mut self) {
        // incminimark.py:3375 asserts the nursery-keyed table is empty here,
        // which holds upstream because a minor precedes every major step.
        // `do_collect_oldgen_nonmoving` is the one entry that deliberately runs
        // without one — that is its contract — so there the table is legitimately
        // populated.
        debug_assert!(
            self.oldgen_nonmoving_active || self.rrc.p_dict_nurs.is_empty(),
            "p_dict_nurs not empty 2"
        );
        self.rrc.p_dict.clear();
        let mut old = std::mem::take(&mut self.rrc.p_list_old);
        let mut surviving = Vec::with_capacity(old.len());
        while let Some(pyobject) = old.pop() {
            self._rrc_major_free(pyobject, &mut surviving, RrcList::P);
        }
        self.rrc.p_list_old = surviving;
        let mut old = std::mem::take(&mut self.rrc.o_list_old);
        let mut surviving = Vec::with_capacity(old.len());
        while let Some(pyobject) = old.pop() {
            self._rrc_major_free(pyobject, &mut surviving, RrcList::O);
        }
        self.rrc.o_list_old = surviving;
    }

    /// incminimark.py:3394-3406 `_rrc_major_free`.
    fn _rrc_major_free(&mut self, pyobject: usize, surviving: &mut Vec<usize>, list: RrcList) {
        // incminimark.py:3395-3401 — the mirror survives exactly if its object
        // did: VISITED means marking reached it, NO_HEAP_PTRS an immortal
        // object marking never has to reach.
        let obj = unsafe { (*rawrefcount::pyobj(pyobject)).ob_link };
        let alive = if !self.is_managed_heap_object(obj) {
            // Upstream reaches the flag test unconditionally because every
            // RPython GC object has a header, prebuilt ones included — which is
            // what its GCFLAG_NO_HEAP_PTRS arm is for.  `malloc_typed` objects
            // here live outside both generations with no header at all, so
            // there is no flag word at that address to read.  Nothing sweeps
            // them either, so nothing can prove one dead.
            true
        } else {
            let hdr = unsafe { header_of(obj) };
            unsafe { (*hdr).has_flag(flags::VISITED) || (*hdr).has_flag(flags::NO_HEAP_PTRS) }
        };
        if alive {
            surviving.push(pyobject);
            if list == RrcList::P {
                self.rrc.p_dict.insert(obj, pyobject);
            }
        } else {
            self._rrc_free(pyobject);
        }
    }

    /// incminimark.py:1259 `get_possibly_forwarded_tid`: header access that
    /// follows the forwarding pointer when a young object was already moved
    /// by this collection. A forwarded nursery header holds FORWARDED_MARKER,
    /// whose all-ones flag region fakes every flag (IGNORE_FINALIZER
    /// included), so a raw read there would silently drop a finalizer entry.
    ///
    /// Latent today: every finalizer-queue registrant (instances, generators)
    /// is stable-allocated, so a queued object is never in the nursery. It
    /// becomes load-bearing when instance allocation converges back to the
    /// movable nursery (see `alloc_instance_object`).
    fn get_possibly_forwarded_header(&self, obj_addr: usize) -> *const GcHeader {
        let hdr = unsafe { header_of(obj_addr) };
        if self.is_nursery_object_start(obj_addr) && unsafe { (*hdr).is_forwarded() } {
            let fwd_addr = unsafe { GcHeader::forwarding_address(hdr) };
            unsafe { header_of(fwd_addr) }
        } else {
            hdr
        }
    }

    /// incminimark.py:2914-2926 `deal_with_young_objects_with_finalizers`.
    fn deal_with_young_objects_with_finalizers(&mut self) {
        while let Some((obj_addr, fq_index)) =
            self.probably_young_objects_with_finalizers.pop_front()
        {
            let current = if self.is_nursery_object_start(obj_addr) {
                // incminimark.py:2918: the IGNORE_FINALIZER read must go
                // through `get_possibly_forwarded_tid` — the object may have
                // been forwarded by an earlier root walk this collection.
                let hdr = self.get_possibly_forwarded_header(obj_addr);
                if unsafe { (*hdr).has_flag(flags::IGNORE_FINALIZER) } {
                    continue;
                }
                let mut root = GcRef(obj_addr);
                self.drag_out_root(&mut root);
                root.0
            } else {
                if !self.is_managed_heap_object(obj_addr) {
                    continue;
                }
                let hdr = unsafe { header_of(obj_addr) };
                if unsafe { (*hdr).has_flag(flags::IGNORE_FINALIZER) } {
                    continue;
                }
                obj_addr
            };
            self.old_objects_with_finalizers
                .push_back((current, fq_index));
        }
    }

    /// incminimark.py:2884-2895 `deal_with_young_objects_with_destructors`.
    ///
    /// "We can reasonably assume that destructors don't do anything fancy
    /// and *just* call them. Among other things they won't resurrect
    /// objects." For each recorded young object: if it was not forwarded
    /// out it died — run its destructor; otherwise it survived — move it
    /// (translated to its new old-gen address) to the old-destructor list
    /// so the next major collection can reclaim it.
    fn deal_with_young_objects_with_destructors(&mut self) {
        while let Some(obj_addr) = self.young_objects_with_destructors.pop() {
            let hdr_ptr = (obj_addr - GcHeader::SIZE) as *const GcHeader;
            if !unsafe { (*hdr_ptr).is_forwarded() } {
                // Dead: run the destructor before the nursery reset frees
                // the bytes.
                self.run_destructor(obj_addr);
            } else {
                // Surviving: track the promoted copy for the major cycle.
                let new_obj = unsafe { GcHeader::forwarding_address(hdr_ptr) };
                self.old_objects_with_destructors.push(new_obj);
            }
        }
    }

    /// minimark.py:1859-1881 `_allocate_shadow(obj)`.
    /// Allocate a shadow copy in old-gen for a nursery object.  The
    /// shadow's tid is copied from the nursery object; for varsize
    /// objects the length field is also copied.  The nursery header
    /// gets GCFLAG_HAS_SHADOW so `_find_shadow` knows to look it up.
    fn allocate_shadow(&mut self, obj_addr: usize) -> usize {
        let hdr_ptr = (obj_addr - GcHeader::SIZE) as *const GcHeader;
        let type_id = unsafe { (*hdr_ptr).type_id() };
        self.validate_type_id(type_id, obj_addr, "allocate_shadow");
        let payload_size = self.size_for_typeid(obj_addr, type_id, "allocate_shadow");
        let total_size = GcHeader::SIZE + payload_size;
        let (item_size, length_offset) = {
            let type_info = self.types.get(type_id);
            (type_info.item_size, type_info.length_offset)
        };
        let shadow_hdr_ptr = self.oldgen.alloc(total_size);
        let shadow_obj = shadow_hdr_ptr as usize + GcHeader::SIZE;
        unsafe {
            (*(shadow_hdr_ptr as *mut GcHeader)).tid_and_flags = (*hdr_ptr).tid_and_flags;
            // The shadow lives in old-gen, so it carries TRACK_YOUNG_PTRS like
            // every other old-gen object — the write barrier tracks it through
            // that flag. (The source header was copied from the nursery object,
            // which does not carry it.)
            (*(shadow_hdr_ptr as *mut GcHeader)).set_flag(flags::TRACK_YOUNG_PTRS);
            // A shadow reserved during marking is an old-gen object subject to
            // this cycle's future sweep; keep it black so marking does not
            // reclaim the reserved identity home before the object is copied in
            // (incminimark.py:1747 record_pinned_object_with_shadow). See
            // `oldgen_birth_flags`.
            if self.gc_state == GcState::Marking {
                (*(shadow_hdr_ptr as *mut GcHeader)).set_flag(flags::VISITED);
            }
            if item_size > 0 {
                *((shadow_obj + length_offset) as *mut usize) =
                    *((obj_addr + length_offset) as *const usize);
            }
            let nursery_hdr = (obj_addr - GcHeader::SIZE) as *mut GcHeader;
            (*nursery_hdr).set_flag(flags::HAS_SHADOW);
        }
        self.nursery_objects_shadows.insert(obj_addr, shadow_obj);
        shadow_obj
    }

    /// minimark.py:1883-1897 `_find_shadow(obj)`.
    /// Return the shadow address for a nursery object, allocating one
    /// if this is the first request.
    fn find_shadow(&mut self, obj_addr: usize) -> usize {
        let hdr = unsafe { *((obj_addr - GcHeader::SIZE) as *const GcHeader) };
        // A forwarded header is `FORWARDED_MARKER`, whose every flag bit reads
        // set, so it would answer the test below and then die on the map lookup
        // under the shadow message.  `_find_shadow`'s precondition is that the
        // object has not been copied yet, so name the real fault here.
        assert!(
            !hdr.is_forwarded(),
            "stale pointer into the nursery: find_shadow reached a forwarded header at {obj_addr:#x}"
        );
        if hdr.has_flag(flags::HAS_SHADOW) {
            // incminimark.py:2855-2857 `ll_assert(shadow != NULL,
            // "GCFLAG_HAS_SHADOW but no shadow found")`.  HAS_SHADOW
            // guarantees the map holds the shadow; a missing entry is a GC
            // invariant violation, not a cue to silently allocate a second
            // shadow (which would hand out an unstable identity address).
            return self
                .nursery_objects_shadows
                .get(&obj_addr)
                .copied()
                .expect("GCFLAG_HAS_SHADOW but no shadow found");
        }
        self.allocate_shadow(obj_addr)
    }

    /// minimark.py:1900-1915 `id_or_identityhash(gcobj)`.
    /// Return a stable address usable as identity hash.  For nursery
    /// objects, returns the shadow's address (which is where the object
    /// will be copied to at the next minor collection).  For old-gen
    /// objects, returns the object's own address (old-gen objects don't
    /// move in mark-sweep).
    pub fn id_or_identityhash(&mut self, obj_addr: usize) -> usize {
        if self.is_valid_gc_object(obj_addr) && self.is_in_nursery(obj_addr) {
            return self.find_shadow(obj_addr);
        }
        obj_addr
    }

    /// Copy a single nursery object to old gen.
    /// If already forwarded, returns the forwarding address.
    /// Pinned objects are left in place and returned as-is.
    /// Abort with a diagnostic if `type_id` is out of range for the
    /// registered type table. The raw header read that produces it is
    /// layout/ASLR-dependent, so an out-of-range id must fail
    /// deterministically at the trace site with context rather than as
    /// a bare `entries[..]` index panic deep inside `types.get`.
    #[inline]
    fn validate_type_id(&self, type_id: u32, obj_addr: usize, site: &str) {
        if type_id as usize >= self.types.len() {
            panic!(
                "GC BUG: invalid type_id={} at obj_addr={:#x} (header_addr={:#x}, nursery_start={:#x}, site={})",
                type_id,
                obj_addr,
                obj_addr - GcHeader::SIZE,
                self.nursery.start_ptr() as usize,
                site,
            );
        }
    }

    /// `base.py:134-144 _get_size_for_typeid` — the payload size of `obj_addr`,
    /// reading the length field when the type is varsize. `None` when the
    /// length cannot describe an allocation.
    ///
    /// Upstream rounds the result here. Pyre's callers each apply their own
    /// rounding (nursery geometry, arena minimum, inspector alignment), so the
    /// rounding stays at the call sites.
    fn try_size_for_typeid(&self, obj_addr: usize, type_id: u32) -> Option<usize> {
        let type_info = self.types.get(type_id);
        if type_info.item_size == 0 {
            return Some(type_info.size);
        }
        let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
        type_info
            .item_size
            .checked_mul(length)
            .and_then(|items| type_info.size.checked_add(items))
            // No object can be larger than `isize::MAX` — `Layout` refuses to
            // describe one — so a larger result is a decode failure, not a
            // request the allocator could ever serve.
            .filter(|&size| size <= isize::MAX as usize - GcHeader::SIZE)
    }

    /// Panicking [`Self::try_size_for_typeid`], for the collector paths that
    /// are about to allocate or copy that many bytes.
    ///
    /// A varsize length is read straight out of the object, so a collector that
    /// reaches an object before its length field is initialized computes a size
    /// that describes nothing. Report the inputs here: downstream the allocator
    /// sees only the product, and fails on a `Layout` it cannot even build.
    fn size_for_typeid(&self, obj_addr: usize, type_id: u32, site: &str) -> usize {
        match self.try_size_for_typeid(obj_addr, type_id) {
            Some(size) => size,
            None => {
                let type_info = self.types.get(type_id);
                let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
                panic!(
                    "GC BUG: varsize length describes no allocation: length={} (read at \
                     obj_addr={:#x} + length_offset={}) item_size={} fixed_size={} \
                     type_id={} header_addr={:#x} nursery_start={:#x} site={}",
                    length,
                    obj_addr,
                    type_info.length_offset,
                    type_info.item_size,
                    type_info.size,
                    type_id,
                    obj_addr - GcHeader::SIZE,
                    self.nursery.start_ptr() as usize,
                    site,
                );
            }
        }
    }

    fn copy_nursery_object(
        &mut self,
        obj_addr: usize,
        site: &str,
        // The child site names the slot kind; the root path identifies the producer.
        parent_site: &'static str,
        holder_addr: usize,
        slot_addr: usize,
    ) -> GcRef {
        // incminimark.py:2188-2210: a pinned object stays in the nursery, but
        // pinning is not a root. Record it only when a real traced edge reaches
        // it, and remember an old parent so that edge is revisited next minor.
        if self.pinned_objects.contains(&obj_addr) {
            // incminimark.py:2194-2199 records an old parent. The list outlives
            // the nursery reset at the end of this minor and is dereferenced in
            // the next one, so a nursery holder would become a read of recycled
            // nursery bytes; `is_managed_heap_object` alone accepts one.
            if holder_addr != 0
                && self.is_managed_heap_object(holder_addr)
                && !self.is_in_nursery(holder_addr)
            {
                let holder_hdr = unsafe { header_of(holder_addr) };
                if unsafe { !(*holder_hdr).has_flag(flags::PINNED) } {
                    self.old_objects_pointing_to_pinned.push(holder_addr);
                    unsafe { (*holder_hdr).set_flag(flags::PINNED) };
                }
            }
            if self.surviving_pinned_objects.insert(obj_addr) {
                unsafe { (*header_of(obj_addr)).set_flag(flags::VISITED) };
                self.pinned_objects_in_nursery += 1;
            }
            return GcRef(obj_addr);
        }

        // Keep the header access as a raw pointer rather than `&mut GcHeader`.
        // The subsequent `alloc_and_copy` performs a raw read over this same
        // byte range, which would invalidate a long-lived `&mut` under Rust's
        // aliasing rules; re-materialize the reference only for each scoped
        // read/write.
        let hdr_ptr = (obj_addr - GcHeader::SIZE) as *mut GcHeader;

        // Already forwarded?
        if unsafe { (*hdr_ptr).is_forwarded() } {
            let fwd_addr = unsafe { GcHeader::forwarding_address(hdr_ptr) };
            return GcRef(fwd_addr);
        }

        let type_id = unsafe { (*hdr_ptr).type_id() };
        if type_id as usize >= self.types.len() {
            let holder_type_id = if holder_addr == 0 {
                None
            } else {
                Some(unsafe { (*header_of(holder_addr)).type_id() })
            };
            let mut holder_words = [0usize; 8];
            if holder_addr != 0 {
                for (index, word) in holder_words.iter_mut().enumerate() {
                    *word = unsafe { *((holder_addr as *const usize).add(index)) };
                }
            }
            let mut child_words = [0usize; 8];
            for (index, word) in child_words.iter_mut().enumerate() {
                *word = unsafe { *((obj_addr as *const usize).add(index)) };
            }
            let holder_hdr_tid_and_flags = if holder_addr == 0 {
                0
            } else {
                unsafe { (*header_of(holder_addr)).tid_and_flags }
            };
            panic!(
                "GC BUG: invalid type_id={} at obj_addr={:#x} \
                 (minor={}, header_addr={:#x}, nursery_start={:#x}, site={}, \
                 parent_site={}, \
                 nursery_free={:#x}, nursery_top={:#x}, holder_addr={:#x}, \
                 holder_type_id={:?}, holder_offset={:?}, holder_words={:#x?}, \
                 child_tid_and_flags={:#x}, child_flag_complement={:#x}, \
                 child_words={:#x?}, child_vtable_type_id={:?}, \
                 nearest_header={}, \
                 child_nursery_offset={:#x}, child_gen={}, holder_gen={}, \
                 holder_tid_and_flags={:#x}, holder_in_remembered={}, \
                 enclosing={}, gc_state={:?}, minors={}, majors={})",
                type_id,
                obj_addr,
                self.minor_collections,
                obj_addr - GcHeader::SIZE,
                self.nursery.start_ptr() as usize,
                site,
                parent_site,
                self.nursery.free_ptr() as usize,
                self.nursery.top_ptr() as usize,
                holder_addr,
                holder_type_id,
                slot_addr.checked_sub(holder_addr),
                holder_words,
                unsafe { (*hdr_ptr).tid_and_flags },
                // FORWARDED_MARKER sets every flag bit, so a corpse whose
                // 64-bit equality no longer holds names the cleared bit here.
                (!unsafe { (*hdr_ptr).flags() }) as u32,
                child_words,
                self.vtable_to_type_id.get(&child_words[0]).copied(),
                self.describe_nearest_header(obj_addr),
                obj_addr.wrapping_sub(self.nursery.start_ptr() as usize),
                self.describe_generation(obj_addr),
                self.describe_generation(holder_addr),
                holder_hdr_tid_and_flags,
                self.remembered_set.contains(&holder_addr),
                self.describe_enclosing_container(holder_addr, slot_addr, &holder_words),
                self.gc_state,
                self.minor_collections,
                self.major_collections,
            );
        }
        // Compute the actual payload size (for varsize objects, read the length).
        let actual_payload_size = self.size_for_typeid(obj_addr, type_id, site);

        let total_size = GcHeader::SIZE + actual_payload_size;
        let has_gc_ptrs = self.types.get(type_id).has_gc_ptrs;

        // minimark.py:1513-1519: if the object has a pre-allocated
        // shadow (from id() or identityhash()), copy into it instead
        // of a fresh allocation.
        let header_ptr = obj_addr - GcHeader::SIZE;
        let new_header_ptr = if unsafe { (*hdr_ptr).has_flag(flags::HAS_SHADOW) } {
            // incminimark.py:2213-2214 `ll_assert(newobj != NULL,
            // "GCFLAG_HAS_SHADOW but no shadow found")` — HAS_SHADOW
            // guarantees the shadow map entry exists.
            let shadow_obj = *self
                .nursery_objects_shadows
                .get(&obj_addr)
                .expect("GCFLAG_HAS_SHADOW but no shadow found");
            let shadow_hdr = (shadow_obj - GcHeader::SIZE) as *mut u8;
            // incminimark.py:2215-2221: if the shadow is already black
            // (VISITED), the memcpy below overwrites its flags with the
            // nursery object's (white) flags, so a sweep running during this
            // incremental cycle could reclaim it (see test_pin_id_bug).
            // Remember VISITED and re-apply it after the copy.
            let shadow_was_visited =
                unsafe { (*(shadow_hdr as *const GcHeader)).has_flag(flags::VISITED) };
            // minimark.py:1518-1520: clear HAS_SHADOW before copy so the
            // flag does not propagate to the shadow object itself.
            unsafe { (*hdr_ptr).clear_flag(flags::HAS_SHADOW) };
            unsafe {
                std::ptr::copy_nonoverlapping(header_ptr as *const u8, shadow_hdr, total_size);
            }
            if shadow_was_visited {
                unsafe { (*(shadow_hdr as *mut GcHeader)).set_flag(flags::VISITED) };
                // The copy above replaced every field of an object the marker
                // has already accounted for as black, and the incoming values
                // are this cycle's first sight of them: the shadow was reserved
                // black before anything was copied into it (`allocate_shadow`),
                // so nothing has ever traced these slots.  A black object is
                // never pushed again by `grey_child`, so without this its
                // children stay white and the sweep frees them while the object
                // still points at them.  Re-greying a modified black object is
                // what `_add_to_more_objects_to_trace` (incminimark.py:2357-2360)
                // does for every other mid-cycle mutation.
                if self.gc_state == GcState::Marking {
                    self.incr_state.more_gray_stack.push(shadow_obj);
                }
            }
            shadow_hdr
        } else {
            unsafe {
                self.oldgen
                    .alloc_and_copy(header_ptr as *const u8, total_size)
            }
        };
        let new_obj_addr = new_header_ptr as usize + GcHeader::SIZE;
        if crate::gc_lifetime_log_enabled() {
            eprintln!(
                "[gc][alloc] addr={new_obj_addr:#x} type_id={type_id} kind=promotion state={:?}",
                self.gc_state
            );
        }
        crate::note_bh_object(
            new_obj_addr,
            total_size - GcHeader::SIZE,
            crate::BH_PROBE_ORIGIN_PROMOTED,
        );
        self.bytes_made_old_since_cycle =
            self.bytes_made_old_since_cycle.saturating_add(total_size);
        self.nursery_surviving_size = self.nursery_surviving_size.saturating_add(total_size);

        // Set TRACK_YOUNG_PTRS on the new old-gen object. (The source header was
        // copied from the nursery object, which does not carry it.)
        unsafe {
            let h = new_header_ptr as *mut GcHeader;
            (*h).set_flag(flags::TRACK_YOUNG_PTRS);
        }

        // Install forwarding pointer in the nursery copy.
        unsafe {
            GcHeader::set_forwarding_address(hdr_ptr, new_obj_addr);
        }

        // If this object has GC pointers, add it to the work list so we
        // trace its fields and update nursery references.
        if has_gc_ptrs {
            // Clear TRACK_YOUNG_PTRS temporarily so the processing loop
            // can re-set it after processing.
            unsafe {
                (*(new_header_ptr as *mut GcHeader)).clear_flag(flags::TRACK_YOUNG_PTRS);
            }
            self.remembered_set.push(new_obj_addr);
            if crate::gc_lifetime_log_enabled() {
                eprintln!(
                    "[gc][remember] addr={:#x} type_id={} source=promotion state={:?}",
                    new_obj_addr, type_id, self.gc_state
                );
            }
        }

        GcRef(new_obj_addr)
    }

    /// llarena debug-fill parity: a GC-visible slot containing the nursery
    /// poison is an uninitialized reference, not an unmanaged pointer to skip.
    /// Check before nursery-range filtering so poison mode fails closed at the
    /// trace site and identifies the exact holder and slot.
    #[inline]
    fn assert_traced_slot_initialized(
        &self,
        field_ref: GcRef,
        slot_addr: usize,
        holder_addr: usize,
        site: &str,
        // The child site names the slot kind; the root path identifies the producer.
        parent_site: &'static str,
    ) {
        const NURSERY_POISON_WORD: usize = (usize::MAX / 0xff) * 0xaa;
        if self.nursery.poison_enabled() && field_ref.0 == NURSERY_POISON_WORD {
            let holder_type_id = if holder_addr == 0 {
                None
            } else {
                Some(unsafe { (*header_of(holder_addr)).type_id() })
            };
            let holder_offset = slot_addr.checked_sub(holder_addr);
            panic!(
                "GC BUG: traced slot contains nursery poison at slot_addr={:#x} holder_addr={:#x} holder_type_id={:?} holder_offset={:?} site={} parent_site={}",
                slot_addr, holder_addr, holder_type_id, holder_offset, site, parent_site,
            );
        }
    }

    /// incminimark.py:2145-2263 `_trace_drag_out` + :2128-2143
    /// `_trace_drag_out1_marking_phase`, for a nursery object reached
    /// through a *root* slot during minor collection.
    ///
    /// Copies a nursery object out (updating `*gcref`) and, while major
    /// marking is active, greys any white root after that step — including an
    /// already-old object. This is the literal two-part shape of
    /// `_trace_drag_out1_marking_phase`: `_trace_drag_out(root, NULL)` first,
    /// then append `root.address[0]` iff it lacks VISITED/PINNED. The old-root
    /// half matters because stack roots can change after `collect_roots()`'s
    /// initial snapshot; every intervening minor reintroduces newly exposed
    /// white old objects to `more_objects_to_trace`.
    #[inline]
    fn drag_out_root(&mut self, gcref: &mut GcRef) {
        self.assert_traced_slot_initialized(
            *gcref,
            gcref as *mut GcRef as usize,
            0,
            "minor_root",
            "minor_root",
        );
        if self.is_nursery_object_start(gcref.0) {
            let slot_addr = gcref as *mut GcRef as usize;
            *gcref =
                self.copy_nursery_object(gcref.0, "minor_root_target", "minor_root", 0, slot_addr);
        }
        let pinned = self.pinned_objects.contains(&gcref.0);
        // incminimark.py:2140-2143: append iff (VISITED | PINNED) == 0. pyre's
        // marking convention sets VISITED at push time (see `seed_major_root`
        // / `mark_object`), so set it here rather than at pop.
        if self.gc_state == GcState::Marking && !pinned && self.is_managed_heap_object(gcref.0) {
            let hdr = unsafe { header_of(gcref.0) };
            if unsafe { !(*hdr).has_flag(flags::VISITED) } {
                unsafe { (*hdr).set_flag(flags::VISITED) };
                self.incr_state.more_gray_stack.push(gcref.0);
                self.note_nonmoving_nursery_mark(gcref.0);
            }
        }
    }

    /// Trace an object's GC pointer fields and update any that point
    /// into the nursery by copying the target.
    fn trace_and_update_object(&mut self, obj_addr: usize, site: &'static str) {
        crate::bh_probe_note_traced(obj_addr);
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        self.validate_type_id(type_id, obj_addr, site);
        let custom_trace = self.types.get(type_id).custom_trace;

        // custom_trace_hook parity: use custom trace function if registered.
        if let Some(trace_fn) = custom_trace {
            // A custom trace names its own slots — for a JITFRAME, `jf_gcmap`
            // decides them, not the type table. When one of those slots does
            // not decode as an object, the discriminating question is whether
            // the *rest* of the same trace is sound: one bad slot among sound
            // ones is a bad published value, while a trace whose slots are
            // mostly unsound is a map that no longer describes the object.
            // Defer the first undecodable slot so the whole walk completes and
            // the panic can report both.
            let mut deferred: Option<(usize, usize)> = None;
            unsafe {
                trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                    let field_ref = *slot_ptr;
                    self.assert_traced_slot_initialized(
                        field_ref,
                        slot_ptr as usize,
                        obj_addr,
                        "minor_custom_trace",
                        site,
                    );
                    if self.is_nursery_object_start(field_ref.0) {
                        if deferred.is_none() && !self.nursery_start_decodes(field_ref.0) {
                            deferred = Some((slot_ptr as usize, field_ref.0));
                            return;
                        }
                        let new_ref = self.copy_nursery_object(
                            field_ref.0,
                            "minor_custom_trace_target",
                            site,
                            obj_addr,
                            slot_ptr as usize,
                        );
                        *slot_ptr = new_ref;
                    }
                });
            }
            if let Some((slot_addr, field)) = deferred {
                let walk = self.describe_custom_trace_slots(obj_addr, trace_fn);
                eprintln!("GC BUG: custom-trace slot walk for holder={obj_addr:#x}: {walk}");
                self.copy_nursery_object(
                    field,
                    "minor_custom_trace_target",
                    site,
                    obj_addr,
                    slot_addr,
                );
            }
            return;
        }

        let type_info = self.types.get(type_id);
        let gc_ptr_offsets: Vec<usize> = type_info.gc_ptr_offsets.clone();
        let items_have_gc_ptrs = type_info.items_have_gc_ptrs;
        let item_size = type_info.item_size;
        let length_offset = type_info.length_offset;
        let base_size = type_info.size;

        // Process fixed-part GC pointer fields.
        for &offset in &gc_ptr_offsets {
            let slot = (obj_addr + offset) as *mut GcRef;
            let field_ref = unsafe { *slot };
            self.assert_traced_slot_initialized(
                field_ref,
                slot as usize,
                obj_addr,
                "minor_fixed_field",
                site,
            );
            if self.is_nursery_object_start(field_ref.0) {
                let new_ref = self.copy_nursery_object(
                    field_ref.0,
                    "minor_fixed_field_target",
                    site,
                    obj_addr,
                    slot as usize,
                );
                unsafe {
                    *slot = new_ref;
                }
            }
        }

        // Process variable-part items if they contain GC pointers.
        if items_have_gc_ptrs && item_size > 0 {
            let length = unsafe { *((obj_addr + length_offset) as *const usize) };
            let items_start = obj_addr + base_size;
            for i in 0..length {
                let slot = (items_start + i * item_size) as *mut GcRef;
                let field_ref = unsafe { *slot };
                self.assert_traced_slot_initialized(
                    field_ref,
                    slot as usize,
                    obj_addr,
                    "minor_varsize_item",
                    site,
                );
                if self.is_nursery_object_start(field_ref.0) {
                    let new_ref = self.copy_nursery_object(
                        field_ref.0,
                        "minor_varsize_item_target",
                        site,
                        obj_addr,
                        slot as usize,
                    );
                    unsafe {
                        *slot = new_ref;
                    }
                }
            }
        }
    }

    // ── Incremental marking ──

    /// incminimark.py:1264-1268 `get_total_memory_used`. Total memory the GC
    /// is responsible for, NOT counting the nursery: old-gen objects plus
    /// raw-malloced large objects. pyre's `oldgen.total_bytes()` already
    /// aggregates promoted objects and large/raw objects allocated straight
    /// into the old generation.
    fn get_total_memory_used(&self) -> usize {
        self.oldgen.total_bytes()
    }

    /// incminimark.py:1288-1290 `threshold_reached`. True once the old-gen
    /// total has caught up to within `extra` of the next-major threshold,
    /// i.e. it is time to make incremental major-collection progress.
    fn threshold_reached(&self, extra: usize) -> bool {
        (self.next_major_collection_threshold - self.get_total_memory_used() as f64) < extra as f64
    }

    /// incminimark.py:575-594 `set_major_threshold_from`. Set the next-major
    /// threshold, capping growth at `next_major_collection_initial *
    /// growth_rate_max`, flooring at `min_heap_size`, and bounding by
    /// `max_heap_size`. Returns whether the result was bounded by the heap max.
    fn set_major_threshold_from(&mut self, mut threshold: f64, reserving_size: f64) -> bool {
        let threshold_max = self.next_major_collection_initial * self.growth_rate_max;
        if threshold > threshold_max {
            threshold = threshold_max;
        }
        //
        threshold += reserving_size;
        if threshold < self.min_heap_size {
            threshold = self.min_heap_size;
        }
        //
        let bounded = if self.max_heap_size > 0.0 && threshold > self.max_heap_size {
            threshold = self.max_heap_size;
            true
        } else {
            false
        };
        //
        self.next_major_collection_initial = threshold;
        self.next_major_collection_threshold = threshold;
        bounded
    }

    /// Begin a new incremental marking cycle.
    ///
    /// Seeds the gray stack with all root-reachable old-gen objects.
    pub fn start_incremental_cycle(&mut self) {
        debug_assert_eq!(self.gc_state, GcState::Scanning);
        self.incr_state.gray_stack.clear();
        self.incr_state.more_gray_stack.clear();
        self.incr_state.objects_marked = 0;

        self.seed_major_roots();
        self.gc_state = GcState::Marking;
    }

    fn seed_major_root(&mut self, gcref: GcRef, site: &str) {
        // `incminimark.py:2739-2753 _collect_obj` performs NO probe on the root
        // word — the type system guarantees every `Ptr(GcStruct)` reaching
        // `_collect_ref_stk` is a real GC object, so the only tests are non-null
        // and `is_in_nursery`. pyre's root banks are untyped `i64` slices
        // (`shadow_stack::walk_bh_regs`, `walk_resume_ref_roots`) that can carry
        // a stale word from a pooled blackhole interp, so the arena range check
        // is what stands in for that guarantee and must precede every
        // dereference.
        //
        // An object outside both generations reaches a major only through
        // `prebuilt_root_objects`, which `collect_nonstack_roots`
        // (`incminimark.py:2705-2707`) walks separately. `incminimark.py:2782`
        // states the invariant that makes any other admission unsound: such an
        // object "should be in 'prebuilt_root_objects', and the GCFLAG_VISITED
        // will be reset at the end of the collection" — and the only reset pass
        // walks exactly that list.
        if gcref.is_null()
            || !self.is_managed_heap_object(gcref.0)
            || !self.may_enter_marking_worklist(gcref.0)
        {
            return;
        }
        let hdr = unsafe { header_of(gcref.0) };
        // A root pointing at freed or recycled memory decodes to a garbage
        // `type_id`. Without this check the address reaches the gray stack and
        // only fails once `mark_object` pops it, by which time the walker that
        // supplied it is no longer on the stack; `site` names that walker.
        self.validate_type_id(unsafe { (*hdr).type_id() }, gcref.0, site);
        // SAFETY: header_of returns a raw pointer; keep each access
        // short-lived to avoid creating overlapping exclusive borrows.
        let newly_marked = unsafe {
            if !(*hdr).has_flag(flags::VISITED) {
                (*hdr).set_flag(flags::VISITED);
                true
            } else {
                false
            }
        };
        if newly_marked {
            self.incr_state.gray_stack.push(gcref.0);
            self.note_nonmoving_nursery_mark(gcref.0);

            // incminimark.py:1322-1340 requires marking worklists to
            // contain no nursery objects after a minor. Pyre JITFRAME
            // gcmap spills are collector metadata, not SETFIELD_GC
            // writes: a frame first grayed in STATE_SCANNING can leave
            // the active shadow stack before the next minor without a
            // mutator barrier. Arm this newly seeded old root in the
            // existing old_objects_pointing_to_young shape once, so that
            // minor forwards any such spill before resetting nursery.
            if !self.is_in_nursery(gcref.0) && unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) } {
                unsafe { (*hdr).clear_flag(flags::TRACK_YOUNG_PTRS) };
                self.remembered_set.push(gcref.0);
                if crate::gc_lifetime_log_enabled() {
                    eprintln!(
                        "[gc][remember] addr={:#x} type_id={} source=major_seed state={:?}",
                        gcref.0,
                        unsafe { (*hdr).type_id() },
                        self.gc_state
                    );
                }
            }
        }
    }

    /// `incminimark.py:2739-2753 _collect_obj`: an object in the nursery is
    /// never appended to `objects_to_trace` — "such an object is handled by
    /// minor collections and shouldn't be specially handled by major
    /// collections" — and `visit` (:2797-2799) asserts
    /// `not self.is_in_nursery(obj)` on every popped entry.  The worklist
    /// outlives the mutator resuming, so a nursery address left on it is read
    /// back after the next `reset_nursery` has recycled those bytes: the
    /// header decodes to a garbage `type_id`.
    ///
    /// A young object stays reachable without the entry.  When the minor
    /// promotes it, `drag_out_root` greys the promoted copy
    /// (`_trace_drag_out1_marking_phase`); when it is only reachable from an
    /// old parent, that parent is in the remembered set and gets requeued
    /// while MARKING (`_add_to_more_objects_to_trace`).
    ///
    /// The non-moving oldgen major is the one mode that marks the nursery in
    /// place — it leaves those bytes untouched by contract, and
    /// [`Self::note_nonmoving_nursery_mark`] clears the marks afterwards.
    #[inline]
    fn may_enter_marking_worklist(&self, addr: usize) -> bool {
        self.oldgen_nonmoving_active || !self.is_in_nursery(addr)
    }

    /// Record a nursery object greyed during a non-moving major so its
    /// stale `flags::VISITED` is cleared as the strictly-last collection step.
    /// No-op (just a range check) outside a non-moving major; the normal
    /// incremental path runs after a minor so the nursery is empty here.
    #[inline]
    fn note_nonmoving_nursery_mark(&mut self, addr: usize) {
        if self.oldgen_nonmoving_active && self.is_in_nursery(addr) {
            self.oldgen_nonmoving_nursery_marks.push(addr);
        }
    }

    /// Snapshot the root walk shared by inspection and major marking. The
    /// finalizer lists are appended by the caller because
    /// `gc.enumerate_all_roots` includes live registered finalizers, whereas
    /// major marking must leave those objects to the finalization-order pass.
    fn enumerate_root_walker_values(&self) -> Vec<GcRef> {
        self.enumerate_labeled_root_walker_values()
            .into_iter()
            .map(|(gcref, _)| gcref)
            .collect()
    }

    /// [`Self::enumerate_root_walker_values`] with each value tagged by the
    /// walker that produced it, so a root carrying a freed address can be
    /// attributed to its source rather than to the marking loop that pops it.
    fn enumerate_labeled_root_walker_values(&self) -> Vec<(GcRef, &'static str)> {
        // incminimark.py:2717 collect_roots: root_walker.walk_roots()
        // walks the same root sets as minor collection.
        let mut result = Vec::new();
        // Read the registered roots in place. The minor path copies this list
        // first because `drag_out_root` takes `&mut self` while it walks; here
        // the walk only reads, so the copy would be one allocation and one pass
        // over every registered root per collection for nothing.
        for &root_ptr in &self.roots.roots {
            result.push((unsafe { *root_ptr }, "registered_root"));
        }

        let walk_all_mutators = crate::gc_sync::mutators_quiesced();
        let mut visit_shadow_root = |gcref: &mut GcRef| {
            result.push((*gcref, "shadow_stack_root"));
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_roots(&mut visit_shadow_root);
        } else {
            crate::shadow_stack::walk_roots(&mut visit_shadow_root);
        }

        let mut visit_jf_root = |gcref: &mut GcRef| {
            if !gcref.is_null() && crate::shadow_stack::is_libc_jitframe(gcref.0) {
                crate::shadow_stack::trace_libc_jitframe(gcref.0, &mut |slot_ptr| {
                    let field_ref = unsafe { *slot_ptr };
                    if !field_ref.is_null() {
                        result.push((field_ref, "jitframe_slot"));
                    }
                });
            } else {
                result.push((*gcref, "jf_root"));
            }
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_jf_roots(&mut visit_jf_root);
        } else {
            crate::shadow_stack::walk_jf_roots(&mut visit_jf_root);
        }
        // Same source the minor-collection root phase reads; enumerated here
        // too, so a root that exists for the collector is also a root this
        // listing reports. A listing that omitted it would say an object is
        // unreachable while the collector keeps it.
        crate::walk_active_live_deadframes(&mut |addr| {
            crate::shadow_stack::trace_libc_jitframe(addr, &mut |slot_ptr| {
                let field_ref = unsafe { *slot_ptr };
                if !field_ref.is_null() {
                    result.push((field_ref, "deadframe_slot"));
                }
            });
        });

        let mut visit_bh_root = |gcref: &mut GcRef| {
            result.push((*gcref, "blackhole_register"));
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_bh_regs(&mut visit_bh_root);
        } else {
            crate::shadow_stack::walk_bh_regs(&mut visit_bh_root);
        }

        // blackhole resume construction roots (`resume.py:1312`): see the
        // minor-collection path for why the in-flight virtuals_cache /
        // registers_r slices must be seeded as roots.
        let mut visit_resume_root = |gcref: &mut GcRef| {
            result.push((*gcref, "resume_ref_root"));
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_resume_ref_roots(&mut visit_resume_root);
        } else {
            crate::shadow_stack::walk_resume_ref_roots(&mut visit_resume_root);
        }

        let mut visit_extra_area = |gcref: &mut GcRef| {
            result.push((*gcref, "extra_area"));
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_extra_areas(&mut visit_extra_area);
        } else {
            crate::shadow_stack::walk_my_extra_areas(&mut visit_extra_area);
        }

        crate::walk_active_extra_roots(&mut |gcref| {
            result.push((*gcref, "active_extra_root"));
        });

        crate::shadow_stack::walk_extra_roots(|gcref| {
            result.push((*gcref, "extra_root"));
        });
        result
    }

    /// `inspector.py:get_rpy_roots` / `gc.enumerate_all_roots`: snapshot all
    /// inspection roots in the order from `gc/base.py:enumerate_all_roots`.
    /// The returned Vec is raw system-allocator storage, matching
    /// inspector.py's preallocated raw root list.
    fn enumerate_all_root_values(&self) -> Vec<GcRef> {
        let mut result = self.enumerate_root_walker_values();
        result.extend(self.prebuilt_root_objects.iter().copied().map(GcRef));
        // incminimark.py:2734-2736 `enum_live_with_finalizers`: registered
        // finalizer objects remain roots even before they become unreachable
        // and move to an app-visible death queue. AddressDeque.foreach(..., 2)
        // visits every object address while skipping each paired queue index.
        result.extend(
            self.probably_young_objects_with_finalizers
                .iter()
                .map(|&(addr, _)| GcRef(addr)),
        );
        result.extend(
            self.old_objects_with_finalizers
                .iter()
                .map(|&(addr, _)| GcRef(addr)),
        );

        // gc/base.py `enum_pending_finalizers`: objects already moved to a
        // death queue remain GC roots until app-level code pops them.
        let pending: Vec<usize> = self
            .finalizer_handlers
            .iter()
            .flat_map(|handler| handler.deque.iter().copied())
            .collect();
        for addr in pending {
            result.push(GcRef(addr));
        }
        result
    }

    fn seed_major_roots(&mut self) {
        // incminimark.py:2717 collect_roots: root_walker.walk_roots()
        // seeds stack roots for a major marking cycle. Mirror the same
        // root sets as minor collection, but mark old objects instead of
        // copying nursery objects.
        let prebuilt = self.prebuilt_root_objects.clone();
        for addr in prebuilt {
            self.seed_prebuilt_root(addr);
        }
        let mut roots = self.enumerate_labeled_root_walker_values();
        // Objects already moved to a death queue remain ordinary roots until
        // app-level code pops them. Registered live finalizers are deliberately
        // absent here; incminimark's finalization-order pass decides whether
        // they survive or move to the death queue.
        roots.extend(
            self.finalizer_handlers
                .iter()
                .flat_map(|handler| handler.deque.iter().copied())
                .map(|addr| (GcRef(addr), "finalizer_death_queue")),
        );
        for (gcref, site) in roots {
            self.seed_major_root(gcref, site);
        }
    }

    /// incminimark.py:2706-2707 `prebuilt_root_objects.foreach(_collect_obj)`.
    fn seed_prebuilt_root(&mut self, addr: usize) {
        let hdr = unsafe { header_of(addr) };
        debug_assert!(unsafe { !(*hdr).has_flag(flags::NO_HEAP_PTRS) });
        if unsafe { !(*hdr).has_flag(flags::VISITED) } {
            unsafe {
                (*hdr).set_flag(flags::VISITED);
                (*hdr).set_flag(flags::TRACK_YOUNG_PTRS);
            }
            self.incr_state.gray_stack.push(addr);
        }
    }

    /// Whether the collector traverses references out of this object — the
    /// property CPython's `gc.is_tracked` reports.
    ///
    /// Heap ownership alone is the wrong test: an atomic object such as an int
    /// or a float is collector-allocated on some paths (`PYRE_GC_INTERP`, the
    /// JIT, wasm) and immortal on others, so ownership would make the same
    /// Python value answer differently per backend. Its registered type has no
    /// reference to traverse either way, which is the stable property.
    pub fn object_is_tracked(&self, obj_addr: usize) -> bool {
        if !self.is_managed_heap_object(obj_addr) {
            return false;
        }
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        self.validate_type_id(type_id, obj_addr, "object_is_tracked");
        let type_info = self.types.get(type_id);
        type_info.has_gc_ptrs || (type_info.items_have_gc_ptrs && type_info.item_size > 0)
    }

    /// `inspector.py:get_rpy_referents`: trace one object's direct GC
    /// referents without changing collection state.
    fn visit_referents(&self, obj_addr: usize, visitor: &mut dyn FnMut(GcRef)) {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        self.visit_referents_with_type_id(obj_addr, type_id, visitor);
    }

    /// `visit_referents`, but handing the visitor each slot's address instead
    /// of its value, so a caller can report where a bad reference is stored.
    fn visit_referent_slots(&self, obj_addr: usize, visitor: &mut dyn FnMut(*mut GcRef)) {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        self.validate_type_id(type_id, obj_addr, "visit_referent_slots");
        let type_info = self.types.get(type_id);
        if let Some(trace_fn) = type_info.custom_trace {
            unsafe { trace_fn(obj_addr, visitor) };
            return;
        }
        for &offset in &type_info.gc_ptr_offsets {
            visitor((obj_addr + offset) as *mut GcRef);
        }
        if type_info.items_have_gc_ptrs && type_info.item_size > 0 {
            let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
            let items_start = obj_addr + type_info.size;
            for i in 0..length {
                visitor((items_start + i * type_info.item_size) as *mut GcRef);
            }
        }
    }

    /// `visit_referents` for a caller that already resolved the type id,
    /// because a prebuilt object carries no header to read it back out of.
    /// The bounds check sits here rather than beside the header read so that
    /// both callers get its diagnostic instead of `TypeTable::get`'s bare
    /// index panic.
    fn visit_referents_with_type_id(
        &self,
        obj_addr: usize,
        type_id: u32,
        visitor: &mut dyn FnMut(GcRef),
    ) {
        self.validate_type_id(type_id, obj_addr, "visit_referents");
        let type_info = self.types.get(type_id);
        if let Some(trace_fn) = type_info.custom_trace {
            unsafe {
                trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                    let field_ref = *slot_ptr;
                    if !field_ref.is_null() {
                        visitor(field_ref);
                    }
                });
            }
            return;
        }
        for &offset in &type_info.gc_ptr_offsets {
            let field_ref = unsafe { *((obj_addr + offset) as *const GcRef) };
            if !field_ref.is_null() {
                visitor(field_ref);
            }
        }
        if type_info.items_have_gc_ptrs && type_info.item_size > 0 {
            let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
            let items_start = obj_addr + type_info.size;
            for i in 0..length {
                let field_ref =
                    unsafe { *((items_start + i * type_info.item_size) as *const GcRef) };
                if !field_ref.is_null() {
                    visitor(field_ref);
                }
            }
        }
    }

    /// `rpython/rlib/rgc.py:1224 do_get_objects`, using MiniMark's
    /// GCFLAG_EXTRA exactly as the translated RPython path does.
    pub fn do_get_objects(&mut self, generation: i8, visitor: &mut dyn FnMut(GcRef)) {
        if generation == 1 {
            return;
        }
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let roots = self.enumerate_all_root_values();
        let mut pending = roots.clone();
        let mut result = Vec::new();
        while let Some(gcref) = pending.pop() {
            if gcref.is_null() || !self.is_managed_heap_object(gcref.0) {
                continue;
            }
            let hdr = unsafe { header_of(gcref.0) };
            if unsafe { (*hdr).has_flag(flags::EXTRA) } {
                continue;
            }
            unsafe { (*hdr).set_flag(flags::EXTRA) };
            let is_requested_generation = match generation {
                -1 => true,
                0 => self.is_in_nursery(gcref.0),
                2 => !self.is_in_nursery(gcref.0),
                _ => false,
            };
            let type_id = unsafe { (*hdr).type_id() };
            if is_requested_generation
                && !unsafe { (*hdr).has_flag(flags::DUMMY) }
                && self.types.get(type_id).is_object
                && !self.types.get(type_id).hide_from_app_level_inspector
            {
                result.push(gcref);
            }
            self.visit_referents(gcref.0, &mut |child| pending.push(child));
        }

        // rgc.clear_gcflag_extra(roots): repeat the root traversal and restore
        // every toggled header before returning to app-level code.
        let mut pending = roots;
        while let Some(gcref) = pending.pop() {
            if gcref.is_null() || !self.is_managed_heap_object(gcref.0) {
                continue;
            }
            let hdr = unsafe { header_of(gcref.0) };
            if unsafe { !(*hdr).has_flag(flags::EXTRA) } {
                continue;
            }
            unsafe { (*hdr).clear_flag(flags::EXTRA) };
            self.visit_referents(gcref.0, &mut |child| pending.push(child));
        }
        for gcref in result {
            visitor(gcref);
        }
    }

    /// `inspector.py:26-37 get_rpy_roots`: enumerate the raw root values
    /// exactly once, without the reachability traversal used by
    /// `rgc.do_get_objects`.
    pub fn do_get_rpy_roots(&mut self, visitor: &mut dyn FnMut(GcRef)) -> bool {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        for root in self.enumerate_all_root_values() {
            if !root.is_null() {
                visitor(root);
            }
        }
        true
    }

    /// `inspector.py:51-72 get_rpy_referents`: trace only the direct raw
    /// fields of `obj`.  In contrast, `do_get_referents` below recursively
    /// looks through non-object implementation nodes.
    ///
    /// `inspector.py:56-61 _do_append_rpy_referents` traces every gcref
    /// unconditionally, so heap membership cannot reject registered prebuilt
    /// objects. `get_actual_typeid` instead rejects null, tagged immediates,
    /// and unknown foreign pointers; the managed-only nursery check below is
    /// the only remaining `header_of` on this path.
    pub fn do_get_rpy_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) -> bool {
        if obj.is_null() {
            return true;
        }
        let Some(type_id) = self.get_actual_typeid(obj) else {
            return true;
        };
        let is_managed = self.is_managed_heap_object(obj.0);
        if is_managed && self.is_in_nursery(obj.0) && unsafe { (*header_of(obj.0)).is_forwarded() }
        {
            return true;
        }
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        self.visit_referents_with_type_id(obj.0, type_id, visitor);
        true
    }

    /// `inspector.py:209-272 HeapDumper`, preserving its two-phase walk:
    /// roots are emitted first, then `[0, 0, 0, -1]`, then the remaining
    /// reachable objects. `GCFLAG_EXTRA` is the forwarded/PtrInfo-equivalent
    /// visited slot used by upstream, so no address side table is introduced.
    pub fn do_dump_rpy_heap(&mut self, fd: i32) -> Result<bool, i32> {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let roots = self.enumerate_all_root_values();
        let mut root_pending = Vec::new();
        for &root in &roots {
            self.heap_dump_add(root, &mut root_pending);
        }
        let mut pending = Vec::new();
        let mut writer = HeapDumpWriter::new(fd);
        let result = (|| {
            while let Some(obj) = root_pending.pop() {
                self.heap_dump_writeobj(obj, &mut writer, &mut pending)?;
            }
            writer.write_marker()?;
            while let Some(obj) = pending.pop() {
                self.heap_dump_writeobj(obj, &mut writer, &mut pending)?;
            }
            writer.flush()?;
            Ok(true)
        })();

        // inspector.py:166-184 finish_processing: restore GCFLAG_EXTRA by
        // repeating the reachability walk from roots, including on write error
        // so a bad fd cannot poison later inspector calls.
        self.heap_dump_clear_gcflag(roots);
        result
    }

    /// `incminimark.py:1102-1110 raw_malloc_memory_pressure`, including the
    /// framework transform's object-owned field store
    /// (`gctransform/framework.py:861-878`). A negative size releases
    /// previously reported pressure just as upstream does.
    pub fn do_add_memory_pressure(&mut self, size: isize, object: GcRef) {
        // The forced-top store below rewrites the same published free/top words
        // compiled code reads inline, so park the other mutators for it the way
        // every other entry point that mutates published allocator state does.
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        if !object.is_null() && self.is_managed_heap_object(object.0) {
            let type_id = unsafe { (*header_of(object.0)).type_id() };
            if (type_id as usize) < self.types.len()
                && let Some(offset) = self.types.get(type_id).memory_pressure_offset
            {
                unsafe {
                    *((object.0 + offset) as *mut isize) = size;
                }
            }
        }

        // incminimark.py:1103-1106: account for the raw allocation plus a
        // small per-allocation overhead so many tiny allocations still drive
        // a major collection.
        self.next_major_collection_threshold -=
            (size as f64) + (2 * std::mem::size_of::<usize>()) as f64;
        if self.next_major_collection_threshold < 0.0 {
            // incminimark.py:1107-1110: make the next nursery allocation take
            // collect_and_reserve. Both the runtime and JIT read these same
            // published free/top words.
            let free = self.nursery.free_ptr();
            unsafe { self.nursery.set_top_ptr(free.cast_const()) };
            self.refresh_published_nursery_top();
        }
    }

    /// `inspector.py:178-195 MemoryPressureCounter`: walk the root-reachable
    /// heap, summing each type's translated `special_memory_pressure` field.
    /// `GCFLAG_EXTRA` is the upstream walker slot; no address side table is
    /// introduced.
    pub fn do_count_memory_pressure(&mut self) -> isize {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let roots = self.enumerate_all_root_values();
        let mut pending = Vec::new();
        for &root in &roots {
            self.heap_dump_add(root, &mut pending);
        }
        let mut count = 0isize;
        while let Some(obj) = pending.pop() {
            let type_id = unsafe { (*header_of(obj.0)).type_id() };
            if (type_id as usize) < self.types.len()
                && let Some(offset) = self.types.get(type_id).memory_pressure_offset
            {
                count = count.wrapping_add(unsafe { *((obj.0 + offset) as *const isize) });
            }
            let mut referents = Vec::new();
            self.visit_referents(obj.0, &mut |child| referents.push(child));
            for child in referents {
                self.heap_dump_add(child, &mut pending);
            }
        }
        self.heap_dump_clear_gcflag(roots);
        count
    }

    fn heap_dump_add(&self, obj: GcRef, pending: &mut Vec<GcRef>) {
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(flags::EXTRA) } {
            return;
        }
        unsafe { (*hdr).set_flag(flags::EXTRA) };
        pending.push(obj);
    }

    fn heap_dump_writeobj(
        &self,
        obj: GcRef,
        writer: &mut HeapDumpWriter,
        pending: &mut Vec<GcRef>,
    ) -> Result<(), i32> {
        let type_id = unsafe { (*header_of(obj.0)).type_id() };
        writer.write(obj.0 as isize)?;
        writer.write((type_id as isize) + 1)?;
        writer.write(self.rpy_memory_usage(obj).unwrap_or(0) as isize)?;
        let mut referents = Vec::new();
        self.visit_referents(obj.0, &mut |child| referents.push(child));
        for child in referents {
            writer.write(child.0 as isize)?;
            self.heap_dump_add(child, pending);
        }
        writer.write(-1)
    }

    fn heap_dump_clear_gcflag(&self, roots: Vec<GcRef>) {
        let mut pending = roots;
        while let Some(obj) = pending.pop() {
            if obj.is_null() || !self.is_managed_heap_object(obj.0) {
                continue;
            }
            let hdr = unsafe { header_of(obj.0) };
            if unsafe { !(*hdr).has_flag(flags::EXTRA) } {
                continue;
            }
            unsafe { (*hdr).clear_flag(flags::EXTRA) };
            self.visit_referents(obj.0, &mut |child| pending.push(child));
        }
    }

    /// `referents.py:17-33 try_cast_gcref_to_w_root`.  The translated
    /// `T_IS_RPYTHON_INSTANCE` bit is `TypeInfo::is_object`; the explicit hide
    /// bit covers internal structs that share a Python-object prefix but have
    /// no app-level typedef.
    ///
    /// This is the single predicate behind every app-level inspector — the
    /// `gc.get_objects` filter, the `get_rpy_*` wrap decision, and the walk
    /// terminator in [`Self::do_get_referents`] — because upstream passes the
    /// one `try_cast_gcref_to_w_root` to all three.
    fn is_app_level_object_ref(&self, obj: GcRef) -> bool {
        // `referents.py:18 rgc.get_gcflag_dummy(gcref)`: a dummy stands in for
        // an object the collector no longer holds, so it is never an app-level
        // object however its type reads.  Only a managed object carries the
        // header the flag lives in.
        if self.is_managed_heap_object(obj.0)
            && unsafe { (*header_of(obj.0)).has_flag(flags::DUMMY) }
        {
            return false;
        }
        let Some(type_id) = self.get_actual_typeid(obj) else {
            return false;
        };
        if type_id as usize >= self.types.len() {
            return false;
        }
        let info = self.types.get(type_id);
        info.is_object && !info.hide_from_app_level_inspector
    }

    /// `pypy/module/gc/referents.py:53-78 _list_w_obj_referents`: visit the
    /// app-level objects `obj` refers to directly, looking through the
    /// interpreter-internal structs in between. A visited app-level object
    /// terminates that branch of the walk; anything else is expanded, so a
    /// list reports its items rather than its item array. GCFLAG_EXTRA keeps
    /// each node out of the walk twice and is restored before returning.
    pub fn do_get_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) {
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        // A host-side raw local can still hold a just-forwarded nursery
        // address, whose header word is a forwarding pointer rather than
        // flags — toggling GCFLAG_EXTRA on it would write through that word.
        // Reject exactly that, not every young object: `referents.py:53-78`
        // does not require an empty nursery, and a live nursery object is an
        // ordinary app-level referent source.
        if self.is_in_nursery(obj.0) && unsafe { (*header_of(obj.0)).is_forwarded() } {
            return;
        }
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let mut pending: Vec<GcRef> = Vec::new();
        let mut result: Vec<GcRef> = Vec::new();
        let mut i = 0usize;
        let mut parent = obj;
        loop {
            let mut children: Vec<GcRef> = Vec::new();
            self.visit_referents(parent.0, &mut |child| children.push(child));
            for child in children {
                if child.is_null() || !self.is_managed_heap_object(child.0) {
                    continue;
                }
                let hdr = unsafe { header_of(child.0) };
                if unsafe { (*hdr).has_flag(flags::EXTRA) } {
                    continue;
                }
                unsafe { (*hdr).set_flag(flags::EXTRA) };
                pending.push(child);
            }
            // Walk the queue until a non-app-level node needs expanding; on
            // reaching the end without one, every branch has terminated.
            let mut expand = false;
            while i < pending.len() {
                parent = pending[i];
                i += 1;
                if self.is_app_level_object_ref(parent) {
                    result.push(parent);
                } else {
                    expand = true;
                    break;
                }
            }
            if !expand {
                break;
            }
        }
        for gcref in &pending {
            unsafe { (*header_of(gcref.0)).clear_flag(flags::EXTRA) };
        }
        for gcref in result {
            visitor(gcref);
        }
    }

    /// incminimark.py:2478-2481 `collect_nonstack_roots(); visit_all_objects()`.
    ///
    /// Non-stack roots may grow after the initial root snapshot while marking
    /// is incremental.  Revisit the process/interpreter-owned root walkers and
    /// pending-finalizer queues immediately before finalizer processing and
    /// sweep, then drain every newly greyed object.  Thread frame/shadow-stack
    /// roots are deliberately not repeated here: upstream repeats only
    /// `collect_nonstack_roots`, not `collect_roots`.
    fn rescan_major_nonstack_roots_and_drain(&mut self) {
        let prebuilt = self.prebuilt_root_objects.clone();
        for addr in prebuilt {
            self.seed_prebuilt_root(addr);
        }
        crate::shadow_stack::walk_extra_roots(|gcref| {
            self.seed_major_root(*gcref, "rescan_extra_root");
        });
        let pending: Vec<usize> = self
            .finalizer_handlers
            .iter()
            .flat_map(|handler| handler.deque.iter().copied())
            .collect();
        for addr in pending {
            self.seed_major_root(GcRef(addr), "rescan_finalizer_death_queue");
        }
        while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
            self.mark_object(obj_addr);
        }
    }

    /// Drive incremental major-collection progress after a minor collection.
    ///
    /// This follows incminimark's accounting rule: each major step grants
    /// `nursery_size / 2` bytes of promotion credit, and allocation-heavy
    /// minors may need multiple consecutive steps so old-gen growth does not
    /// outrun marking.
    fn run_major_progress_after_minor(&mut self) {
        // incminimark.py:832 — automatic major progress after a minor stops
        // while disabled; explicit collect() passes force_enabled and stays
        // ungated (collect_full / collect_oldgen_nonmoving here).
        if !self.enabled {
            return;
        }
        if self.gc_state == GcState::Scanning && !self.threshold_reached(0) {
            return;
        }

        loop {
            self.major_collection_step();

            // incminimark.py:849-855 target (A1).
            if self.gc_state == GcState::Scanning {
                break;
            }

            // incminimark.py:849-860 target (A2). Keep pyre's existing
            // consecutive-step credit loop: the enclosing call has already
            // completed the minor collection that supplied these promotions.
            if self.bytes_made_old_since_cycle <= self.threshold_bytes_made_old {
                break;
            }
        }
    }

    /// incminimark.py:2390-2634 `major_collection_step`.
    fn major_collection_step(&mut self) {
        let start = GcClock::start();
        let old_state = self.gc_state.encoded();
        let oom_was_pending = self.oom_pending;
        self.debug_check_consistency();

        // incminimark.py:2406-2436: each state-machine step grants half a
        // nursery of promotion credit.
        self.threshold_bytes_made_old = self
            .threshold_bytes_made_old
            .saturating_add(self.config.nursery_size / 2);

        match self.gc_state {
            GcState::Scanning => {
                // incminimark.py:2439-2448 STATE_SCANNING.
                self.bytes_made_old_since_cycle = 0;
                self.threshold_bytes_made_old = self.config.nursery_size / 2;
                self.start_incremental_cycle();
            }
            GcState::Marking => {
                // `incremental_mark_step` performs the MARKING->SWEEPING seam
                // itself when the gray stack is exhausted.
                self.incremental_mark_step();
            }
            GcState::Sweeping => self.incremental_sweep_step(),
            GcState::Finalizing => {
                // incminimark.py:2623-2631: recursive collections from a
                // handler must see a collector ready to start a new scan.
                self.gc_state = GcState::Scanning;
                self.execute_finalizer_triggers();
            }
        }

        // incminimark.py:2634-2644. A max-heap MemoryError exits upstream
        // before this site; pyre communicates it through `oom_pending`, so
        // suppress the transition event when this step newly raised it.
        if self.oom_pending == oom_was_pending {
            let duration = start.elapsed_secs();
            self.total_gc_time += duration;
            self.hooks
                .fire_gc_collect_step(duration, old_state, self.gc_state.encoded());
        }
    }

    /// incminimark.py:1316-1319 debug invariant.
    fn debug_check_consistency(&self) {
        if self.oldgen.rawmalloc_sweep_pending() {
            debug_assert_eq!(
                self.gc_state,
                GcState::Sweeping,
                "raw_malloc_might_sweep must be empty outside SWEEPING"
            );
        }
    }

    /// Perform one incremental marking step.
    ///
    /// Processes up to `mark_budget_per_step` bytes from the gray stack.
    /// Like incminimark, this is a byte budget, but we always process at least
    /// one object so very small budgets still make forward progress.
    /// Returns `true` if marking is complete (gray stack exhausted).
    pub fn incremental_mark_step(&mut self) -> bool {
        debug_assert_eq!(self.gc_state, GcState::Marking);
        // incminimark.py:2453-2457: a nursery with many survivors raises this
        // increment's byte budget so promotion cannot outrun tracing.
        let estimate = self
            .incr_state
            .mark_budget_per_step
            .max(self.nursery_surviving_size.saturating_mul(2))
            .max(1);
        let mut remaining = estimate;
        while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
            let obj_size = self.object_total_size(obj_addr);
            self.mark_object(obj_addr);
            self.incr_state.objects_marked += 1;
            if obj_size > remaining {
                remaining = 0;
                break;
            }
            remaining -= obj_size;
        }

        // incminimark.py:2458-2470: if the ordinary frontier used less than
        // half the step, swap in every object exposed by concurrent mutation
        // and drain that worklist completely.  This deliberately trades some
        // incrementality for termination.
        if self.incr_state.gray_stack.is_empty()
            && remaining >= estimate / 2
            && !self.incr_state.more_gray_stack.is_empty()
        {
            std::mem::swap(
                &mut self.incr_state.gray_stack,
                &mut self.incr_state.more_gray_stack,
            );
            while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
                self.mark_object(obj_addr);
                self.incr_state.objects_marked += 1;
            }
        }

        if self.incr_state.gray_stack.is_empty() && self.incr_state.more_gray_stack.is_empty() {
            self.finish_incremental_marking();
            return true;
        }
        false // more work to do
    }

    fn object_total_size(&self, obj_addr: usize) -> usize {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        self.validate_type_id(type_id, obj_addr, "object_total_size");
        GcHeader::SIZE + self.size_for_typeid(obj_addr, type_id, "object_total_size")
    }

    /// `base.py:135-141 get_size` / `inspector.py:76-77
    /// get_rpy_memory_usage`.  The inspector reports the translated object
    /// size, not the collector header.  Fixed-size rows are already rounded by
    /// `gctypelayout.encode_type_shape`; variable-size rows are rounded after
    /// reading the live length.
    fn rpy_memory_usage(&self, obj: GcRef) -> Option<usize> {
        let type_id = self.get_actual_typeid(obj)?;
        if type_id as usize >= self.types.len() {
            return None;
        }
        let size = self.try_size_for_typeid(obj.0, type_id)?;
        if self.types.get(type_id).item_size == 0 {
            return Some(size);
        }
        let align_mask = GcHeader::ALIGN - 1;
        size.checked_add(align_mask).map(|size| size & !align_mask)
    }

    /// `inspector.py:79-81 get_rpy_type_index`.  Pyre's compact type table
    /// stores one paired TYPE_INFO/CLASSTYPE row per type, so its zero-based
    /// row number differs from RPython's group-member index only by the dummy
    /// member that `TypeLayoutBuilder.make_type_info_group` installs at zero.
    fn rpy_type_index(&self, obj: GcRef) -> Option<usize> {
        let type_id = self.get_actual_typeid(obj)?;
        if type_id as usize >= self.types.len() {
            return None;
        }
        (type_id as usize).checked_add(1)
    }

    /// Total size of `addr`, or `None` when its header does not decode to a
    /// registered type. The panicking [`Self::object_total_size`] is unusable
    /// from a diagnostic that is already reporting a corrupt heap.
    fn try_object_total_size(&self, addr: usize) -> Option<usize> {
        let type_id = unsafe { (*header_of(addr)).type_id() } as usize;
        if type_id >= self.types.len() {
            return None;
        }
        let payload_size = self.try_size_for_typeid(addr, type_id as u32)?;
        Some(GcHeader::SIZE + payload_size)
    }

    /// Which generation an address falls in.
    ///
    /// A corrupt child separates into two unrelated defects depending on the
    /// answer: an old-gen holder naming a nursery child is a store that never
    /// reached the remembered set, while an old-gen child is a sweep that
    /// reclaimed a still-referenced object. The panic reports both ends so the
    /// two are never confused.
    fn describe_generation(&self, addr: usize) -> &'static str {
        if self.nursery.contains(addr) {
            "nursery"
        } else if self.oldgen.contains(addr) {
            "oldgen"
        } else {
            "outside"
        }
    }

    /// Whether the header in front of a nursery address decodes at all.
    ///
    /// `is_nursery_object_start` is a range check, so it answers "could be an
    /// object", not "is one". This is the cheap half of the question the
    /// `copy_nursery_object` panic answers in full.
    fn nursery_start_decodes(&self, obj_addr: usize) -> bool {
        let hdr = unsafe { *header_of(obj_addr) };
        hdr.is_forwarded() || (hdr.type_id() as usize) < self.types.len()
    }

    /// Every slot a custom trace names, with the state of the value in it.
    ///
    /// For a JITFRAME the slot set is whatever `jf_gcmap` says, so this is the
    /// only way to see the map the collector actually acted on.
    fn describe_custom_trace_slots(
        &self,
        obj_addr: usize,
        trace_fn: crate::trace::CustomTraceFn,
    ) -> String {
        let mut slots: Vec<(usize, usize)> = Vec::new();
        unsafe {
            trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                slots.push((slot_ptr as usize, (*slot_ptr).0));
            });
        }
        let mut out = format!("{} slots:", slots.len());
        for (slot_addr, value) in slots {
            let offset = slot_addr.wrapping_sub(obj_addr);
            let tag = if value == 0 {
                "null".to_string()
            } else if !self.is_managed_heap_object(value) {
                "unmanaged".to_string()
            } else if !self.nursery_start_decodes(value) {
                format!("{}-BAD", self.describe_generation(value))
            } else {
                let hdr = unsafe { *header_of(value) };
                if hdr.is_forwarded() {
                    format!("{}-forwarded", self.describe_generation(value))
                } else {
                    format!("{}-tid{}", self.describe_generation(value), hdr.type_id())
                }
            };
            out.push_str(&format!(" [+{offset}]={value:#x}:{tag}"));
        }
        out
    }

    /// The nearest preceding word that decodes as an object header whose
    /// extent covers `obj_addr`.
    ///
    /// `is_nursery_object_start` is a bare range check, so a slot holding an
    /// *interior* address passes it and the collector then reads a payload
    /// word as a header. That is a different defect from a header that was
    /// genuinely clobbered, and the header word alone does not separate them:
    /// finding a real object that contains the address settles it, and its
    /// `interior_offset` names the field the slot actually points at.
    fn describe_nearest_header(&self, obj_addr: usize) -> String {
        let word = std::mem::size_of::<usize>();
        let floor = self.nursery.start_ptr() as usize;
        for back in 1..=64usize {
            let Some(candidate) = obj_addr.checked_sub(back * word) else {
                break;
            };
            if candidate <= floor + GcHeader::SIZE {
                break;
            }
            let hdr = unsafe { *header_of(candidate) };
            if hdr.is_forwarded() {
                let fwd = unsafe { GcHeader::forwarding_address(header_of(candidate)) };
                return format!("forwarded back={back}w candidate={candidate:#x} fwd={fwd:#x}");
            }
            let tid = hdr.type_id();
            if tid as usize >= self.types.len() {
                continue;
            }
            let Some(size) = self.try_size_for_typeid(candidate, tid) else {
                continue;
            };
            if candidate + size <= obj_addr {
                continue;
            }
            return format!(
                "tid={tid} back={back}w candidate={candidate:#x} size={size} \
                 interior_offset={} custom_trace={}",
                obj_addr - candidate,
                self.types.get(tid).custom_trace.is_some(),
            );
        }
        "none".to_string()
    }

    /// Which object actually owns `slot_addr`.
    ///
    /// A `custom_trace` may hand the collector slots that live outside the
    /// object it was called on — the mapdict storage block and the module-dict
    /// storage boxes are walked that way. When the child in such a slot is
    /// corrupt, the discriminating question is whether the *container* was
    /// reclaimed (its own header would be a freelist link) or the child was, and
    /// the panic's `holder_*` fields answer neither. Locate the container by
    /// testing each of the holder's leading words for an extent that covers the
    /// slot, and report its header.
    fn describe_enclosing_container(
        &self,
        holder_addr: usize,
        slot_addr: usize,
        holder_words: &[usize],
    ) -> String {
        if self.is_managed_heap_object(holder_addr)
            && let Some(size) = self.try_object_total_size(holder_addr)
            && (holder_addr..holder_addr + size).contains(&slot_addr)
        {
            return "self".to_string();
        }
        for (index, &word) in holder_words.iter().enumerate() {
            if !self.is_managed_heap_object(word) {
                continue;
            }
            let Some(size) = self.try_object_total_size(word) else {
                continue;
            };
            if !(word..word + size).contains(&slot_addr) {
                continue;
            }
            let hdr = unsafe { *header_of(word) };
            return format!(
                "word{index}@{word:#x} tid={} tid_and_flags={:#x} size={size} slot_offset={}",
                hdr.type_id(),
                hdr.tid_and_flags,
                slot_addr - word,
            );
        }
        "unknown".to_string()
    }

    /// Grey one child reference: if it is a managed, not-yet-visited heap
    /// object, set VISITED and push it onto the gray stack. The
    /// `is_managed_heap_object` guard mirrors `seed_major_root`: a
    /// `Ptr(GcStruct)` field can transiently point at memory outside the
    /// GC-managed heap during the L1/L2 stepping-stone state (e.g.
    /// `W_TupleObject.wrappeditems` → `std::alloc`'d ItemsBlock). In that
    /// window calling `header_of` on the field would dereference memory before
    /// the std::alloc'd block. Upstream RPython `_collect_obj`
    /// (incminimark.py:2739-2752) does not need this guard because RPython's
    /// type system guarantees every `Ptr(GcStruct)` is GC-managed; it converges
    /// away once every `gc_ptr_offsets` target is a real GC allocation.
    fn grey_child(&mut self, addr: usize, holder_addr: usize, slot_addr: usize, site: &str) {
        if self.is_managed_heap_object(addr) && self.may_enter_marking_worklist(addr) {
            let hdr = unsafe { header_of(addr) };
            let type_id = unsafe { (*hdr).type_id() };
            if type_id as usize >= self.types.len() {
                let holder_type_id = unsafe { (*header_of(holder_addr)).type_id() };
                let mut holder_words = [0usize; 8];
                for (index, word) in holder_words.iter_mut().enumerate() {
                    *word = unsafe { *((holder_addr as *const usize).add(index)) };
                }
                let mut child_words = [0usize; 8];
                for (index, word) in child_words.iter_mut().enumerate() {
                    *word = unsafe { *((addr as *const usize).add(index)) };
                }
                let child_vtable_type_id = self.vtable_to_type_id.get(&child_words[0]).copied();
                let holder_hdr = unsafe { *header_of(holder_addr) };
                panic!(
                    "GC BUG: invalid major child type_id={} at child_addr={:#x} \
                     holder_addr={:#x} holder_type_id={} slot_addr={:#x} \
                     holder_offset={:?} site={} holder_words={:#x?} \
                     child_vtable_type_id={:?} child_words={:#x?} \
                     holder_tid_and_flags={:#x} holder_in_remembered={} \
                     child_gen={} holder_gen={} \
                     enclosing={} gc_state={:?} minors={} majors={}",
                    type_id,
                    addr,
                    holder_addr,
                    holder_type_id,
                    slot_addr,
                    slot_addr.checked_sub(holder_addr),
                    site,
                    holder_words,
                    child_vtable_type_id,
                    child_words,
                    holder_hdr.tid_and_flags,
                    self.remembered_set.contains(&holder_addr),
                    self.describe_generation(addr),
                    self.describe_generation(holder_addr),
                    self.describe_enclosing_container(holder_addr, slot_addr, &holder_words),
                    self.gc_state,
                    self.minor_collections,
                    self.major_collections,
                );
            }
            if unsafe { !(*hdr).has_flag(flags::VISITED) } {
                unsafe { (*hdr).set_flag(flags::VISITED) };
                self.incr_state.gray_stack.push(addr);
                self.note_nonmoving_nursery_mark(addr);
            }
        }
    }

    /// Mark a single object: trace its GC pointer fields and push
    /// unmarked children onto the gray stack.
    fn mark_object(&mut self, obj_addr: usize) {
        // Copy the trace descriptors out of the borrowed `type_info` so the
        // `self.types` borrow is released before greying (which mutates
        // `self.incr_state.gray_stack`). Each child is then streamed straight
        // to the gray stack — as RPython `_collect_obj` does — so a large
        // varsize GC-pointer array is never buffered (the gray stack already
        // retains the live children); only the bounded fixed-field offsets are
        // copied into the reused `mark_offsets` buffer.
        // incminimark.py:2797-2799 `visit`: `ll_assert(not
        // self.is_in_nursery(obj), "nursery object in 'objects_to_trace'")`.
        debug_assert!(
            self.may_enter_marking_worklist(obj_addr),
            "nursery object {obj_addr:#x} in the marking worklist",
        );
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        // A non-moving major can trace a live nursery object without first
        // copying it into its reserved old-gen shadow.  Keep that unoccupied
        // shadow black, but do not push/trace it: only its header (and varsize
        // length) is initialized until copy_nursery_object overwrites the full
        // block.  Upstream normally reaches the shadow through
        // GCFLAG_HAS_SHADOW during the leading minor; this is the equivalent
        // for pyre's oldgen-only non-moving major.
        if self.is_in_nursery(obj_addr) {
            // Every flag bit of `FORWARDED_MARKER` reads set, so a worklist
            // entry a minor collection forwarded would pass the shadow test
            // below and then fail the map lookup under a message about the
            // shadow map.  The fault is the stale worklist entry; say so.
            assert!(
                !unsafe { (*header_of(obj_addr)).is_forwarded() },
                "stale major worklist entry: forwarded header at {obj_addr:#x}"
            );
            if unsafe { (*header_of(obj_addr)).has_flag(flags::HAS_SHADOW) } {
                let shadow_obj = *self
                    .nursery_objects_shadows
                    .get(&obj_addr)
                    .expect("GCFLAG_HAS_SHADOW but no shadow found");
                unsafe { (*header_of(shadow_obj)).set_flag(flags::VISITED) };
            }
        }
        let custom_trace;
        let (item_size, length_offset, fixed_size, items_have_gc_ptrs);
        let mut offsets = std::mem::take(&mut self.incr_state.mark_offsets);
        offsets.clear();
        {
            let type_info = self.types.get(type_id);
            custom_trace = type_info.custom_trace;
            item_size = type_info.item_size;
            length_offset = type_info.length_offset;
            fixed_size = type_info.size;
            items_have_gc_ptrs = type_info.items_have_gc_ptrs;
            if custom_trace.is_none() {
                offsets.extend_from_slice(&type_info.gc_ptr_offsets);
            }
        }

        // custom_trace_hook parity for major GC marking.
        if let Some(trace_fn) = custom_trace {
            unsafe {
                trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                    let field_ref = *slot_ptr;
                    if !field_ref.is_null() {
                        self.grey_child(
                            field_ref.0,
                            obj_addr,
                            slot_ptr as usize,
                            "major_custom_trace",
                        );
                    }
                });
            }
        } else {
            // Fixed-part fields (count bounded by the struct's GC field count).
            for &offset in &offsets {
                let field_ref = unsafe { *((obj_addr + offset) as *const GcRef) };
                if !field_ref.is_null() {
                    self.grey_child(
                        field_ref.0,
                        obj_addr,
                        obj_addr + offset,
                        "major_fixed_field",
                    );
                }
            }
            // Variable-part items: streamed one at a time, never buffered, so a
            // large GC-managed pointer array does not double the marking-side
            // peak memory or retain that capacity for the collector's lifetime.
            if items_have_gc_ptrs && item_size > 0 {
                let length = unsafe { *((obj_addr + length_offset) as *const usize) };
                let items_start = obj_addr + fixed_size;
                for i in 0..length {
                    let field_ref = unsafe { *((items_start + i * item_size) as *const GcRef) };
                    if !field_ref.is_null() {
                        self.grey_child(
                            field_ref.0,
                            obj_addr,
                            items_start + i * item_size,
                            "major_varsize_item",
                        );
                    }
                }
            }
        }

        // Return the (small) offsets buffer for reuse.
        self.incr_state.mark_offsets = offsets;
    }

    /// incminimark.py:1793-1799 + :2461-2470 — final snapshot-at-the-beginning
    /// re-scan before the sweep completes.
    ///
    /// Every old object modified since the last minor collection is recorded in
    /// `remembered_set` (the write barrier's `old_objects_pointing_to_young`) with
    /// its `TRACK_YOUNG_PTRS` cleared. Such a store may install an `old -> old`
    /// edge to a still-white object. The per-minor black re-grey in
    /// `do_collect_nursery` only re-scans modifications up to the last minor; a
    /// cycle finished at a safepoint *between* minors (`do_collect_oldgen_nonmoving`
    /// branch A, `gc_step`) leaves the trailing delta unscanned. The later
    /// incremental sweep would otherwise free those white-but-reachable children.
    /// Re-grey every already-marked modified object and trace transitively so the
    /// delta is marked before the retain/sweep below. (pyre's marking convention
    /// sets VISITED at push time, so a black object is simply re-pushed; white
    /// entries are left for the normal frontier / retain to drop.)
    fn rescan_remembered_black_and_drain(&mut self) {
        let snapshot: Vec<usize> = self.remembered_set.to_vec();
        for addr in snapshot {
            if unsafe { (*header_of(addr)).has_flag(flags::VISITED) } {
                self.incr_state.gray_stack.push(addr);
            }
        }
        while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
            self.mark_object(obj_addr);
        }
    }

    /// incminimark.py:1792-1799 turns every old object modified since the cycle
    /// began back to gray — "precisely the old objects that have been modified
    /// and need rescanning" — before the sweep decides survivors, and
    /// :2478-2481 rescans the roots that can grow after the cycle's opening
    /// snapshot. Upstream needs only the non-stack half there, because its
    /// stack roots are covered by two invariants pyre does not share: a
    /// JITFRAME is nursery-allocated, so every minor re-traces it and a
    /// promotion during MARKING re-queues it (:2079-2083), and the objects a
    /// mutator stores into a stack slot mid-cycle were promoted during MARKING
    /// and are therefore born black.
    ///
    /// pyre's stack root sets are mutated with no write barrier and hold
    /// pre-cycle objects: a JitFrame lives in the old gen so its pointer stays
    /// valid across a collecting call while compiled code stores Refs into its
    /// gcmap slots, the blackhole register banks and resume-construction roots
    /// are plain slices, and `seed_major_root` arms a newly seeded old root
    /// into the remembered set only once — the next minor drains that set and
    /// nothing re-arms it. A black root can therefore come to hold the only
    /// reference to a white object. Walk the root sets once more here and turn
    /// the black ones gray again; this can only add survivors, never free a
    /// reachable object.
    fn rescan_major_stack_roots_black_and_drain(&mut self) {
        for gcref in self.enumerate_root_walker_values() {
            if gcref.is_null() {
                continue;
            }
            // incminimark.py:1322-1340 keeps nursery objects out of a marking
            // worklist, so a nursery root goes through the seeding path, which
            // marks it without queueing it.
            let regray = !self.is_in_nursery(gcref.0)
                && self.is_managed_heap_object(gcref.0)
                && unsafe { (*header_of(gcref.0)).has_flag(flags::VISITED) };
            if regray {
                self.incr_state.gray_stack.push(gcref.0);
            } else {
                self.seed_major_root(gcref, "marking_regray_root");
            }
        }
        while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
            self.mark_object(obj_addr);
        }
    }

    /// incminimark.py:2761-2763 `visit_all_objects`: mark until the worklist is
    /// empty.  Every caller that seeds a root outside the incremental budget
    /// finishes it here rather than leaving work for the next step.
    fn drain_gray_stack(&mut self) {
        while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
            self.mark_object(obj_addr);
        }
    }

    /// Mark conditional side-table edges to a fixed point.
    ///
    /// RPython sees an object's weakref lifeline as an ordinary field: marking
    /// the owner immediately marks the lifeline. Pyre's temporary carrier for
    /// builtin layouts is address-keyed, so it reports the same edges here,
    /// after ordinary marking has established which owners survived. A newly
    /// marked value may itself make another owner live, hence the fixed point.
    fn mark_ephemeron_values_to_fixed_point(&mut self) {
        loop {
            let mut classify_owner = |owner: usize| -> Option<usize> {
                if owner == 0 || !self.oldgen.contains(owner) {
                    return Some(owner);
                }
                let hdr = unsafe { header_of(owner) };
                unsafe { (*hdr).has_flag(flags::VISITED) }.then_some(owner)
            };
            let roots = crate::shadow_stack::mark_ephemeron_tables(&mut classify_owner);
            for root in roots {
                self.seed_major_root(root, "ephemeron_table_root");
            }
            // Progress is what this drain itself marked: `objects_marked` is
            // owned by `incremental_mark_step` and never moves here, so reading
            // it would end the walk after one pass and leave the second owner's
            // value unmarked while pruning keeps its now-live entry.  Seeding
            // an already-VISITED root pushes nothing, so an empty drain is
            // exactly "no owner became live in this pass".
            let mut drained = 0usize;
            while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
                self.mark_object(obj_addr);
                drained += 1;
            }
            if drained == 0 {
                break;
            }
        }
    }

    /// incminimark.py:2473-2533: finish MARKING and freeze this cycle's sweep
    /// candidates.  Every VISITED-dependent consumer runs before either raw or
    /// arena memory is freed.
    fn finish_incremental_marking(&mut self) {
        debug_assert_eq!(self.gc_state, GcState::Marking);
        // Re-scan the trailing snapshot-at-the-beginning delta (old objects
        // modified since the last minor) before the sweep, or a cycle finished
        // between minors frees reachable old->old targets.
        self.rescan_remembered_black_and_drain();
        // Barrier-less stack-root stores (JitFrame gcmap slots, blackhole
        // register banks, resume-construction roots): re-gray the black roots
        // before the sweep freezes survivors.
        self.rescan_major_stack_roots_black_and_drain();
        // incminimark.py:2478-2481: process-global/non-stack roots can grow
        // after the cycle's initial snapshot.  Rescan and trace them before
        // finalizers, weakrefs, and sweep inspect VISITED.
        self.rescan_major_nonstack_roots_and_drain();
        // incminimark.py:2486-2487 — the P list is a root source like any
        // other, and it is consulted here, after the ordinary roots and before
        // anything reads VISITED to decide what dies.
        if self.rrc.enabled {
            self.rrc_major_collection_trace();
            if self.oldgen_nonmoving_active {
                self.rrc_nonmoving_major_trace_young();
            }
        }
        self.mark_ephemeron_values_to_fixed_point();
        // incminimark.py:2492: this counter belongs to one finalization-order
        // pass.  `_bump_finalization_state_from_0_to_1` below repopulates it
        // with every otherwise-dead object retained for a finalizer.
        self.kept_alive_by_finalizer = 0;
        // incminimark.py:2961-2965 (and :2495-2499) — clear weak
        // pointers to dying objects before the sweep frees them. The
        // VISITED bit on every old-gen object is still meaningful at
        // this point; sweep below tears down `VISITED` per object as
        // it walks oldgen.
        if !self.old_objects_with_finalizers.is_empty() {
            self.deal_with_objects_with_finalizers();
        } else if !self.old_objects_with_weakrefs.is_empty() {
            self.invalidate_old_weakrefs();
        }
        if self.oldgen_nonmoving_active && !self.young_objects_with_weakrefs.is_empty() {
            self.invalidate_young_weakrefs_for_nonmoving_major();
        }
        // A non-moving major (do_collect_oldgen_nonmoving) runs with a live
        // nursery and no preceding minor, so the write barrier's
        // remembered_set / old_objects_with_cards_set are still populated and
        // may name old objects this cycle is about to free. Drop the dead ones
        // (VISITED is still set on every survivor at this point) so the next
        // minor does not trace freed memory. A no-op for do_collect_full, whose
        // leading minor already drained both sets.
        self.remembered_set
            .retain(|&addr| unsafe { (*header_of(addr)).has_flag(flags::VISITED) });
        self.old_objects_with_cards_set
            .retain(|&addr| unsafe { (*header_of(addr)).has_flag(flags::VISITED) });
        // Embedder side tables keyed by owner address hold their value as a
        // root, so an entry outlives its owner unless it is dropped here —
        // the same "the target died" question `invalidate_old_weakrefs` just
        // answered, asked on behalf of a table the collector cannot see.
        // An owner outside old-gen is either immortal (`malloc_typed`, no
        // header to read) or, under a non-moving major, still in the live
        // nursery; neither can be proven dead here, so both are kept.
        let mut classify_owner = |owner: usize| -> Option<usize> {
            if owner == 0 || !self.oldgen.contains(owner) {
                return Some(owner);
            }
            let hdr = unsafe { header_of(owner) };
            if unsafe { (*hdr).has_flag(flags::VISITED) } {
                Some(owner)
            } else {
                None
            }
        };
        crate::shadow_stack::prune_ephemeron_tables(&mut classify_owner);
        // The same question for tables a single mutator owns in its own TLS.
        // Those cannot go through the global registration: it names no thread,
        // so a major driven here would leave every other mutator's dead-owner
        // entries pinned. Reach exactly as far as this collection's own root
        // walk did (`enumerate_root_walker_values`) — foreign TLS only while
        // this thread owns STW.
        if crate::gc_sync::mutators_quiesced() {
            crate::shadow_stack::prune_all_mutator_areas(&mut classify_owner);
        } else {
            crate::shadow_stack::prune_my_mutator_areas(&mut classify_owner);
        }
        // incminimark.py:2510-2511 — run destructors of dying old objects
        // before the sweep frees them (VISITED still distinguishes
        // survivors from the dying at this point).
        if !self.old_objects_with_destructors.is_empty() {
            self.deal_with_old_objects_with_destructors();
        }
        // incminimark.py:2512-2514. ArenaCollection's prepare moves active
        // pages to old_* lists, and OldGen swaps old_rawmalloced_objects into
        // raw_malloc_might_sweep. Promotions between sweep steps therefore
        // allocate into fresh lists and are not candidates in this cycle.
        self.oldgen.sweep_prepare();

        // incminimark.py:2516-2526: dead old parents must leave the pin-parent
        // stack before arena sweep invalidates their addresses. Survivors have
        // VISITED set at this point; the oldgen sweep clears it later.
        self.old_objects_pointing_to_pinned
            .retain(|&obj_addr| unsafe { (*header_of(obj_addr)).has_flag(flags::VISITED) });
        // incminimark.py:2528-2529 — VISITED still distinguishes survivors, so
        // this is where each mirror learns whether its object made it.
        if self.rrc.enabled {
            self.rrc_major_collection_free();
        }
        // incminimark.py:2531-2532 — snapshot the pre-sweep accounting after
        // the candidate sets have been frozen and before the state changes.
        self.stat_ac_arenas_count = self.oldgen.arenas_count();
        self.stat_rawmalloced_total_size = self.oldgen.rawmalloced_bytes();
        self.gc_state = GcState::Sweeping;
    }

    /// incminimark.py:2535-2621 STATE_SWEEPING.
    fn incremental_sweep_step(&mut self) {
        debug_assert_eq!(self.gc_state, GcState::Sweeping);
        let done = if self.oldgen.rawmalloc_sweep_pending() {
            // incminimark.py:2537-2547: process a bounded number of rawmalloc
            // objects first. Even if this drains the stack, the arena half is
            // deliberately deferred to the next state-machine step.
            let limit = 3 * self.config.nursery_size / self.oldgen.small_request_threshold();
            // Upstream production geometry makes this non-zero. Pyre unit
            // tests use nurseries smaller than one rawmalloc threshold, so one
            // candidate is the minimum needed for forward progress while
            // retaining the incminimark.py:2543 upper-bound formula otherwise.
            self.oldgen.sweep_rawmalloc_step(limit.max(1));
            false
        } else {
            // incminimark.py:2549-2555: visit at most three nursery sizes of
            // frozen arena pages.
            let limit = 3 * self.config.nursery_size / self.oldgen.page_size();
            // As above, tiny test nurseries need one-page forward progress;
            // normal configurations use the literal incminimark.py:2553 value.
            self.oldgen.sweep_arenas_step(limit.max(1))
        };

        if done {
            self.finish_incremental_cycle();
        }
    }

    /// incminimark.py:2560-2621: complete SWEEPING, compute the next threshold
    /// from post-sweep accounting, then enter FINALIZING.
    fn finish_incremental_cycle(&mut self) {
        debug_assert_eq!(self.gc_state, GcState::Sweeping);
        debug_assert!(!self.oldgen.rawmalloc_sweep_pending());
        self.major_collections += 1;
        if crate::majit_log_enabled() {
            eprintln!(
                "[gc][major] complete count={} oldgen_bytes={}",
                self.major_collections,
                self.get_total_memory_used(),
            );
        }

        // incminimark.py:2563-2564: prebuilt objects are outside the arena
        // sweep, so reset their mark bit explicitly for the next cycle.
        for &addr in &self.prebuilt_root_objects {
            unsafe { (*header_of(addr)).clear_flag(flags::VISITED) };
        }
        // incminimark.py:2566-2577 — set the threshold for the next major
        // collection to `major_collection_threshold` times the surviving
        // size, but no more than `max_delta` above it, floored at
        // `min_heap_size` by set_major_threshold_from. incminimark.py:2570-2573
        // subtracts objects that survived only so their finalizers can run;
        // clamp at zero exactly as upstream does after the subtraction.
        let total_memory_used = self
            .get_total_memory_used()
            .saturating_sub(self.kept_alive_by_finalizer) as f64;
        // incminimark.py:2574-2577 — capped next-major threshold. `reserving_size`
        // is the byte size of the allocation that triggered this collection,
        // carried on the collector across `do_collect_nursery` (see
        // `pending_reserving_size`), matching the argument
        // `major_collection_step(reserving_size)` threads to both
        // `set_major_threshold_from` and `threshold_reached`.
        let reserving_size = self.pending_reserving_size;
        let bounded = self.set_major_threshold_from(
            (total_memory_used * self.major_collection_threshold)
                .min(total_memory_used + self.max_delta),
            reserving_size as f64,
        );
        self.bytes_made_old_since_cycle = 0;
        self.threshold_bytes_made_old = 0;

        // incminimark.py:2592-2600 — report post-sweep accounting after the
        // next threshold is computed, but before the max-heap check and the
        // transition into FINALIZING.
        self.hooks.fire_gc_collect(
            self.major_collections,
            self.stat_ac_arenas_count,
            self.oldgen.arenas_count(),
            self.oldgen.arenas_bytes(),
            self.stat_rawmalloced_total_size,
            self.oldgen.rawmalloced_bytes(),
            self.pinned_objects_in_nursery,
        );

        // incminimark.py:2601-2615 — max heap size (PYPY_GC_MAX). If the capped
        // threshold was bounded by `max_heap_size` and the heap has already
        // reached it, signal out-of-memory. The first time, ask the triggering
        // allocation to return NULL so `CHECK_MEMORY_ERROR` (compiled code) or
        // the interpreter allocation chokepoint raises `MemoryError`, giving the
        // program a chance to quit cleanly; a second occurrence aborts the
        // process (`out_of_memory` -> fatalerror). `max_heap_size == 0`
        // (unbounded default) never sets `bounded`, so this is inert unless
        // `PYPY_GC_MAX` is set.
        if bounded && self.threshold_reached(reserving_size) {
            if self.max_heap_size_already_raised {
                panic!("using too much memory, aborting");
            }
            self.max_heap_size_already_raised = true;
            self.oom_pending = true;
            // incminimark.py:2614-2615: STATE_SCANNING then an
            // immediate `raise MemoryError` exits `major_collection_step`
            // before the finalizing phase. Return before the queue-notification
            // triggers so none fire ahead of the `MemoryError` the pending NULL
            // will raise.
            self.gc_state = GcState::Scanning;
            return;
        }

        self.gc_state = GcState::Finalizing;
    }

    /// incminimark.py:3105-3126 `invalidate_old_weakrefs(self)`.
    ///
    /// For each old-gen WEAKREF recorded in `old_objects_with_weakrefs`:
    ///   * If the weakref struct itself was not marked → it dies; drop
    ///     it from the list (no slot mutation, sweep will reclaim it).
    ///   * Else read the `weakptr` slot and check the target's VISITED
    ///     bit. Live target → keep the weakref in a fresh list.
    ///     Dying target → null the slot and drop the weakref.
    ///
    /// incminimark.py:3120-3121 also treats FINALIZATION_ORDERING targets as
    /// dying for weakref purposes even though finalizer-queue reachability has
    /// marked them VISITED; the check below preserves that ordering.
    fn invalidate_old_weakrefs(&mut self) {
        let entries = std::mem::take(&mut self.old_objects_with_weakrefs);
        let mut new_with_weakref = Vec::with_capacity(entries.len());
        for obj_addr in entries {
            let hdr_ptr = (obj_addr - GcHeader::SIZE) as *const GcHeader;
            // incminimark.py:3112: weakref itself not marked → dies.
            if unsafe { !(*hdr_ptr).has_flag(flags::VISITED) } {
                continue;
            }
            let weakptr_slot = (obj_addr + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef;
            let pointing_to = unsafe { (*weakptr_slot).0 };
            if pointing_to == 0 {
                continue;
            }
            // A target that is not an old-gen object is immortal
            // (`malloc_typed`): it carries no GcHeader to read a VISITED bit
            // from and never dies, so the weakref stays valid. Keep tracking
            // it with the slot intact, mirroring the foreign-target handling
            // in `invalidate_young_weakrefs`. `alloc_in_oldgen` registers
            // born-old weakrefs whose target may be such an immortal object
            // (e.g. a weakref to a type), so this is reachable; the nursery is
            // empty at this point, so any non-old-gen target is immortal.
            if !self.oldgen.contains(pointing_to) {
                new_with_weakref.push(obj_addr);
                continue;
            }
            // incminimark.py:3120-3121: queued finalizers carry VISITED so
            // they survive this cycle, but FINALIZATION_ORDERING still makes
            // their weakrefs observe death before `__del__` runs.
            let target_hdr = (pointing_to - GcHeader::SIZE) as *const GcHeader;
            if unsafe {
                (*target_hdr).has_flag(flags::VISITED)
                    && !(*target_hdr).has_flag(flags::FINALIZATION_ORDERING)
            } {
                new_with_weakref.push(obj_addr);
            } else {
                unsafe { (*weakptr_slot).0 = 0 };
            }
        }
        self.old_objects_with_weakrefs = new_with_weakref;
    }

    fn finalization_state(&self, obj_addr: usize) -> u8 {
        let hdr = unsafe { header_of(obj_addr) };
        let visited = unsafe { (*hdr).has_flag(flags::VISITED) };
        let ordering = unsafe { (*hdr).has_flag(flags::FINALIZATION_ORDERING) };
        match (visited, ordering) {
            (false, false) => 0,
            (false, true) => 1,
            (true, true) => 2,
            (true, false) => 3,
        }
    }

    /// Return the GC-managed outgoing references of one object.  This is the
    /// collector `trace()` callback shape used by incminimark's finalization
    /// ordering walk, kept separate from normal VISITED marking.
    fn finalizer_children(&self, obj_addr: usize) -> Vec<usize> {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        let info = self.types.get(type_id);
        let mut children = Vec::new();
        if let Some(trace_fn) = info.custom_trace {
            unsafe {
                trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                    let child = *slot_ptr;
                    if !child.is_null() && self.is_managed_heap_object(child.0) {
                        children.push(child.0);
                    }
                });
            }
            return children;
        }
        for &offset in &info.gc_ptr_offsets {
            let child = unsafe { *((obj_addr + offset) as *const GcRef) };
            if !child.is_null() && self.is_managed_heap_object(child.0) {
                children.push(child.0);
            }
        }
        if info.items_have_gc_ptrs && info.item_size > 0 {
            let length = unsafe { *((obj_addr + info.length_offset) as *const usize) };
            let items_start = obj_addr + info.size;
            for i in 0..length {
                let child = unsafe { *((items_start + i * info.item_size) as *const GcRef) };
                if !child.is_null() && self.is_managed_heap_object(child.0) {
                    children.push(child.0);
                }
            }
        }
        children
    }

    fn recursively_clear_finalization_ordering(&mut self, obj_addr: usize) {
        let mut pending = vec![obj_addr];
        while let Some(addr) = pending.pop() {
            if self.finalization_state(addr) == 2 {
                unsafe { (*header_of(addr)).clear_flag(flags::FINALIZATION_ORDERING) };
                pending.extend(self.finalizer_children(addr));
            }
        }
    }

    /// incminimark.py:2928-2983 `deal_with_objects_with_finalizers`.
    fn deal_with_objects_with_finalizers(&mut self) {
        let mut new_with_finalizer = VecDeque::new();
        let mut marked = VecDeque::new();

        while let Some((obj_addr, fq_index)) = self.old_objects_with_finalizers.pop_front() {
            let hdr = unsafe { header_of(obj_addr) };
            if unsafe { (*hdr).has_flag(flags::IGNORE_FINALIZER) } {
                continue;
            }
            if unsafe { (*hdr).has_flag(flags::VISITED) } {
                new_with_finalizer.push_back((obj_addr, fq_index));
                continue;
            }

            marked.push_back((obj_addr, fq_index));
            let mut pending = vec![obj_addr];
            while let Some(addr) = pending.pop() {
                match self.finalization_state(addr) {
                    0 => {
                        unsafe { (*header_of(addr)).set_flag(flags::FINALIZATION_ORDERING) };
                        // incminimark.py:3011-3020
                        //
                        // Upstream reaches this seam only after a minor has
                        // emptied the nursery, so every `addr` contributes to
                        // `get_total_memory_used()`. Pyre's non-moving
                        // interpreter major deliberately skips that minor and
                        // may trace a young child here. The threshold metric
                        // likewise excludes nursery bytes, so subtract only
                        // the old-gen subset of the otherwise-dead graph.
                        if self.oldgen.contains(addr) {
                            self.kept_alive_by_finalizer = self
                                .kept_alive_by_finalizer
                                .wrapping_add(OldGen::allocation_size(
                                    self.object_total_size(addr),
                                ));
                        }
                        pending.extend(self.finalizer_children(addr));
                    }
                    2 => self.recursively_clear_finalization_ordering(addr),
                    _ => {}
                }
            }

            // `_recursively_bump_finalization_state_from_1_to_2`: enqueue the
            // root in normal marking and visit its complete object graph.
            self.seed_major_root(GcRef(obj_addr), "finalization_ordering_root");
            while let Some(addr) = self.incr_state.gray_stack.pop() {
                self.mark_object(addr);
            }
        }

        // Resurrecting a finalizer's object graph marks owners that the
        // caller's ephemeron pass had already written off, and an
        // address-keyed side table is not reached by the tracing that just
        // marked them — `_recursively_bump_finalization_state_from_1_to_2`
        // reaches a lifeline only because upstream stores it in an ordinary
        // field. Re-establish those edges here: `prune_ephemeron_tables` reads
        // the same VISITED bits below and would otherwise keep an entry whose
        // value nothing marked, leaving the table pointing into swept memory.
        self.mark_ephemeron_values_to_fixed_point();

        // PyPy clears weakrefs while queued objects are in state 2.
        if !self.old_objects_with_weakrefs.is_empty() {
            self.invalidate_old_weakrefs();
        }

        while let Some((obj_addr, fq_index)) = marked.pop_front() {
            if self.finalization_state(obj_addr) == 2 {
                self.finalizer_handlers[fq_index].deque.push_back(obj_addr);
                self.recursively_clear_finalization_ordering(obj_addr);
            } else {
                new_with_finalizer.push_back((obj_addr, fq_index));
            }
        }
        self.old_objects_with_finalizers = new_with_finalizer;
    }

    fn execute_finalizer_triggers(&mut self) {
        if self.finalizer_lock {
            return;
        }
        self.finalizer_lock = true;
        for handler in &self.finalizer_handlers {
            if !handler.deque.is_empty() {
                (handler.trigger)();
            }
        }
        self.finalizer_lock = false;
    }

    /// incminimark.py:2897-2912 `deal_with_old_objects_with_destructors`.
    ///
    /// Walk the old-destructor list: a VISITED (surviving) object is kept
    /// in a fresh list; a dying (not-VISITED) object's destructor runs
    /// before the imminent sweep frees its memory. Must run while VISITED
    /// bits are still meaningful — i.e. before incremental sweep steps clear
    /// them.
    fn deal_with_old_objects_with_destructors(&mut self) {
        let entries = std::mem::take(&mut self.old_objects_with_destructors);
        let mut new_objects = Vec::with_capacity(entries.len());
        for obj_addr in entries {
            let hdr_ptr = (obj_addr - GcHeader::SIZE) as *const GcHeader;
            if unsafe { (*hdr_ptr).has_flag(flags::VISITED) } {
                // surviving
                new_objects.push(obj_addr);
            } else {
                // dying
                self.run_destructor(obj_addr);
            }
        }
        self.old_objects_with_destructors = new_objects;
    }

    /// Whether an incremental marking cycle is currently in progress.
    pub fn is_incremental_marking(&self) -> bool {
        self.gc_state == GcState::Marking
    }

    /// Number of objects marked so far in the current incremental cycle.
    pub fn incremental_objects_marked(&self) -> usize {
        self.incr_state.objects_marked
    }

    /// incminimark.py:530-537 `enable` / `disable` / `isenabled`.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn isenabled(&self) -> bool {
        self.enabled
    }

    /// Set the per-step marking budget in bytes.
    pub fn set_mark_budget(&mut self, budget: usize) {
        self.incr_state.mark_budget_per_step = budget;
    }

    /// Enable or disable `gc_stress` forced-collection-per-allocation for this
    /// instance. Only present under the `gc_stress` feature; callers gate use
    /// with `#[cfg(feature = "gc_stress")]`.
    #[cfg(feature = "gc_stress")]
    pub fn set_stress_collect(&mut self, on: bool) {
        self.stress_collect = on;
    }

    /// Perform a full (major) mark-sweep collection.
    ///
    /// 1. First do a minor collection to promote all live nursery objects.
    /// 2. Mark phase: trace all roots and transitively mark reachable objects.
    /// 3. Sweep phase: free all unmarked old-gen objects.
    pub fn do_collect_full(&mut self) {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        // incminimark.py:2361-2378 `minor_and_major_collection` /
        // `gc_step_until`: first finish the current major cycle, if any, with
        // a minor collection before every major step.  Besides discovering
        // new roots, that minor forwards every nursery edge held by the
        // in-progress marking state before the next gray object is consumed.
        self.gc_step_until_scanning_with_minors();

        // Minor collection first to empty the nursery.
        // This is the `_minor_collection()` performed by upstream's
        // `gc_step_until(STATE_MARKING)` before it starts the fresh cycle.
        self.do_collect_nursery();

        if self.gc_state == GcState::Scanning {
            self.start_incremental_cycle();
        }
        self.gc_step_until_scanning_with_minors();

        // incminimark.py:808.
        self.rrc_invoke_callback();
    }

    /// incminimark.py:810-822 `collect_step`.
    pub fn collect_step(&mut self) -> crate::GcStepTransition {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let old_state = self.gc_state;

        // `_minor_collection`, not `minor_collection_with_major_progress`:
        // the explicit `major_collection_step` below is the one and only
        // state transition performed by this call.
        let was_enabled = self.enabled;
        self.enabled = false;
        self.do_collect_nursery();
        self.enabled = was_enabled;
        self.major_collection_step();

        // incminimark.py:821.
        self.rrc_invoke_callback();

        crate::GcStepTransition {
            old_state: old_state.encoded(),
            new_state: self.gc_state.encoded(),
        }
    }

    /// Reclaim dead old-gen objects WITHOUT moving the nursery.
    ///
    /// A non-moving major: seed roots, mark transitively, and sweep only the
    /// old generation — skipping the leading minor that [`do_collect_full`]
    /// runs. The nursery is left byte-for-byte intact (not moved, not freed),
    /// so an unrooted Rust-stack `PyObjectRef` into the nursery cannot dangle
    /// — the exact hazard that blocks a moving minor at an interp safepoint
    /// where there is no shadowstack pass. Reachability stays exact because
    /// `seed_major_root` / `mark_object` gate on `is_managed_heap_object`,
    /// which INCLUDES the nursery, so an `old -> nursery -> old` live edge is
    /// fully followed and the target old object is marked before the sweep.
    ///
    /// Unlike a moving minor, nursery survivors keep their addresses, so a
    /// `flags::VISITED` set on a marked-through nursery object would otherwise
    /// survive into the next minor's promoted copy (`copy_nursery_object`
    /// memcpys the header verbatim). The strictly-last step clears VISITED on
    /// exactly the nursery objects greyed this cycle — after
    /// sweep completion, hence after `invalidate_old_weakrefs`,
    /// which reads a nursery target's VISITED to decide weakref survival. The
    /// remembered set is left untouched: a non-moving major does not consume
    /// `old -> young` edges, so the next minor still finds them.
    pub fn do_collect_oldgen_nonmoving(&mut self) {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        self.oldgen_nonmoving_active = true;
        self.oldgen_nonmoving_nursery_marks.clear();

        if self.gc_state == GcState::Scanning {
            self.start_incremental_cycle();
        }
        // Keep this oldgen-only entry stop-the-world: it may enter while
        // MARKING or SWEEPING, but always returns after the complete cycle.
        self.gc_step_until_scanning();

        // Strictly-last: clear VISITED on every nursery object greyed this
        // cycle (the oldgen sweep already cleared it on old-gen survivors).
        let marks = std::mem::take(&mut self.oldgen_nonmoving_nursery_marks);
        for addr in marks {
            // Nothing moved, so each addr is still nursery-resident; the
            // guard is defensive against a duplicate already cleared.
            if self.is_in_nursery(addr) {
                let hdr = unsafe { header_of(addr) };
                unsafe { (*hdr).clear_flag(flags::VISITED) };
            }
        }
        self.oldgen_nonmoving_active = false;

        // This entry has no upstream counterpart, but it is a public collection
        // entry point and it can queue mirrors, so it owes the same schedule.
        self.rrc_invoke_callback();
    }

    fn gc_step_until_scanning(&mut self) {
        while self.gc_state != GcState::Scanning {
            self.major_collection_step();
        }
    }

    /// incminimark.py:2375-2378 `gc_step_until`: explicit full collection
    /// performs a minor before every major state-machine step.  The ordinary
    /// non-moving oldgen entry deliberately uses [`gc_step_until_scanning`]
    /// instead because its contract is to leave the nursery byte-for-byte in
    /// place.
    fn gc_step_until_scanning_with_minors(&mut self) {
        while self.gc_state != GcState::Scanning {
            // `do_collect_nursery` is pyre's public
            // `minor_collection_with_major_progress`, whereas upstream calls
            // the lower-level `_minor_collection` here and then advances one
            // explicit step.  Suppress that wrapper's automatic progress so
            // this remains the literal one-minor/one-major loop.
            let was_enabled = self.enabled;
            self.enabled = false;
            self.do_collect_nursery();
            self.enabled = was_enabled;
            self.major_collection_step();
        }
    }

    /// incminimark.py:1489-1493 write_barrier(addr_struct):
    /// if the object has GCFLAG_TRACK_YOUNG_PTRS, call remember_young_pointer.
    /// Nursery objects never have TRACK_YOUNG_PTRS, so the flag test alone
    /// selects the old-gen objects that may now point to young.
    ///
    /// Unlike incminimark — where every struct is GC-managed and the barrier
    /// only ever receives a header-bearing GC object — pyre runs a hybrid heap:
    /// host-side allocators (`w_list_new`, `w_dict_new`, …) fall back to a bare
    /// `Box::into_raw` when no GC hook is installed (bootstrap / import-time
    /// objects), producing PyObjects that are not in any managed generation and
    /// whose `obj - 8` word is Rust allocator metadata, not a `GcHeader`. Such
    /// an object legitimately reaches the interpreter barrier sites (a slice
    /// store on a bootstrap list, a namespace store on an import-time dict), so
    /// the barrier must ignore any address the GC does not own rather than read
    /// its non-header word. This centralizes the `try_gc_owns_object` guard that
    /// `object_array.rs` / `list_write_barrier` already apply per call site.
    ///
    /// The guard is equally load-bearing on the JIT path, which is why this
    /// entry point — not `jit_remember_young_pointer` — is what the backends'
    /// non-array COND_CALL_GC_WB helper calls for a JITFRAME base. An ordinary
    /// store's base takes `jit_remember_young_pointer` instead
    /// (`framework.py:538-544`), the unguarded entry the array barrier already
    /// reaches through `jit_remember_young_pointer_from_array`.
    /// `_reload_frame_if_necessary`
    /// (aarch64/assembler.py:967-980) re-applies the non-array barrier fast
    /// path to the *current jitframe* after every collecting helper call.
    /// Not every jitframe is nursery-allocated: the runner's entry frame, the
    /// realloc slowpath, and the JITFRAME nursery slowpath's fallback build
    /// frames off the GC — the class `shadow_stack::register_libc_jitframe`
    /// exists to track. Those go through `jitframe::alloc_off_gc_jitframe`,
    /// which reserves a zeroed header word so the inline test's read at
    /// `jit_wb_if_flag_byteofs` (negative, where a header would be) stays
    /// inside the allocation and finds the flag clear. Reaching this helper
    /// with such a frame therefore takes a set flag bit the allocator did not
    /// write; `is_managed_heap_object` is what still keeps the unmanaged block
    /// out of `remembered_set`, where the next minor would decode a type id
    /// from it. `write_barrier_ignores_unmanaged_jitframe_with_flag_byte_set`
    /// pins that case.
    pub fn do_write_barrier(&mut self, obj: GcRef) {
        // incminimark's write_barrier receives a typed, non-null struct pointer.
        // pyre's GcRef is nullable (GcRef::NULL is the sentinel) and reaches the
        // safe `write_barrier`/`gc_write_barrier` entry points, so guard null
        // before reading `header_of(obj)`; the card variant guards it likewise.
        if obj.is_null() || self.is_in_nursery(obj.0) {
            return;
        }
        if self.is_managed_heap_object(obj.0) {
            let hdr = unsafe { header_of(obj.0) };
            if unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) } {
                self.remember_young_pointer(obj);
            }
            return;
        }
        // `malloc_typed` bootstrap objects are incminimark's prebuilt family.
        // The witness is `registered_pyobject_header`, not vtable membership
        // alone: `w_tuple_new` / `w_specialisedtuple_new_*` fall back to a bare
        // `Box::into_raw` when no GC is installed, and those objects carry the
        // very same registered `ob_header` at offset 0 with NO preceding
        // `GcHeader`. A vtable-only test therefore reads the allocator word in
        // front of the box and — if its `TRACK_YOUNG_PTRS` bit happens to be
        // set — has `remember_young_pointer` write flags back into that
        // metadata. The tid equality is what separates the two families.
        let Some(hdr) = self.registered_pyobject_header(obj.0) else {
            return;
        };
        if unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) } {
            self.remember_young_pointer(obj);
        }
    }

    /// `do_write_barrier` sibling for a pointer returned by this collector's
    /// allocation API.  RPython's barrier receives only GC-managed structs;
    /// the membership guard exists solely for pyre's mixed bootstrap heap and
    /// is redundant for a freshly allocated managed result.
    fn do_write_barrier_managed(&mut self, obj: GcRef) {
        if obj.is_null() {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) } {
            self.remember_young_pointer(obj);
        }
    }

    /// incminimark.py:1503-1529 _remember_young_pointer_inlined(addr):
    /// append the object to the remembered set (`old_objects_pointing_to_young`)
    /// and clear GCFLAG_TRACK_YOUNG_PTRS. Callers have already verified the flag
    /// (the inline COND_CALL_GC_WB test, or `do_write_barrier`).
    fn remember_young_pointer(&mut self, obj: GcRef) {
        // incminimark.py does not special-case STATE_SWEEPING in its barrier.
        // After the seam, every retained entry names a VISITED survivor. A new
        // barrier entry can only name an object the mutator can reach: an old
        // candidate already known live, a candidate already swept, or a new
        // object on the fresh lists. An unreachable white candidate cannot be
        // mutated and therefore cannot be re-added between sweep steps.
        let type_id = unsafe { (*header_of(obj.0)).type_id() };
        self.validate_type_id(type_id, obj.0, "remember_young_pointer_insert");
        self.remembered_set.push(obj.0);
        crate::bh_probe_note_barriered(obj.0);
        let hdr = unsafe { header_of(obj.0) };
        if crate::gc_lifetime_log_enabled() {
            eprintln!(
                "[gc][remember] addr={:#x} type_id={} source=write_barrier state={:?}",
                obj.0,
                unsafe { (*hdr).type_id() },
                self.gc_state
            );
        }
        unsafe {
            (*hdr).clear_flag(flags::TRACK_YOUNG_PTRS);
            if (*hdr).has_flag(flags::NO_HEAP_PTRS) {
                (*hdr).clear_flag(flags::NO_HEAP_PTRS);
                self.prebuilt_root_objects.push(obj.0);
            }
        }
    }

    /// incminimark.py:1495-1501 write_barrier_from_array:
    /// called by non-JIT code. Checks TRACK_YOUNG_PTRS, then dispatches
    /// to remember_young_pointer_from_array2 (card path) or
    /// remember_young_pointer (generic).
    pub fn do_write_barrier_card(&mut self, obj: GcRef, index: usize, card_page_shift: u32) {
        // A non-GC (`Box::into_raw`) PyObject can reach the barrier here too;
        // ignore any address the GC does not own (see `do_write_barrier`).
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) } {
            self.remember_young_pointer_from_array2(obj, index, card_page_shift);
        }
    }

    /// incminimark.py:1557-1600 remember_young_pointer_from_array2:
    /// Called when TRACK_YOUNG_PTRS is set. If HAS_CARDS, marks the
    /// card WITHOUT clearing TRACK_YOUNG_PTRS. Otherwise falls back
    /// to the generic _remember_young_pointer_inlined.
    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    ) {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(flags::HAS_CARDS) } {
            // No cards — fall back to the generic _remember_young_pointer_inlined.
            self.remember_young_pointer(obj);
            return;
        }
        // Card path: mark the specific card. TRACK_YOUNG_PTRS stays set
        // so that subsequent writes still enter the barrier.
        self.mark_card(obj, index, card_page_shift);
    }

    /// incminimark.py:1606-1617 jit_remember_young_pointer_from_array:
    /// Minimal version called by the JIT when TRACK_YOUNG_PTRS is set
    /// but CARDS_SET is not. Tries to set CARDS_SET; otherwise falls
    /// back to remember_young_pointer (generic barrier).
    pub fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef) {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(flags::HAS_CARDS) } {
            // incminimark.py:1614-1615
            self.old_objects_with_cards_set.push(obj.0);
            unsafe { (*hdr).set_flag(flags::CARDS_SET) };
        } else {
            // No cards: the JIT already passed the inline flag test, so use
            // the corresponding remember_young_pointer path.
            self.jit_remember_young_pointer(obj);
        }
    }

    /// incminimark.py:1574-1598: set the card bit for a given array index.
    /// Card bits are stored inline before the GcHeader. If CARDS_SET not
    /// yet set, add obj to `old_objects_with_cards_set` and set CARDS_SET.
    fn mark_card(&mut self, obj: GcRef, index: usize, card_page_shift: u32) {
        // incminimark.py:1576-1578
        let bitindex = index >> card_page_shift;
        let byteindex = bitindex >> 3;
        let bitmask: u8 = 1 << (bitindex & 7);

        // incminimark.py:1581-1584: if bit already set, return.
        let addr_byte = Self::get_card_ptr(obj.0, byteindex);
        let byte = unsafe { *addr_byte };
        if byte & bitmask != 0 {
            return;
        }

        // incminimark.py:1594: set the bit.
        unsafe {
            *addr_byte = byte | bitmask;
        }

        // incminimark.py:1596-1598: if CARDS_SET not set, track and set.
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(flags::CARDS_SET) } {
            self.old_objects_with_cards_set.push(obj.0);
            unsafe { (*hdr).set_flag(flags::CARDS_SET) };
        }
    }

    /// Check whether a specific card of an object is dirty.
    /// Reads inline card bytes before the GcHeader.
    pub fn is_card_dirty(&self, obj: GcRef, card_index: usize) -> bool {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(flags::HAS_CARDS) } {
            return false;
        }
        let byteindex = card_index >> 3;
        let bitmask: u8 = 1 << (card_index & 7);
        let addr_byte = Self::get_card_ptr(obj.0, byteindex);
        unsafe { *addr_byte & bitmask != 0 }
    }

    /// Return all dirty card indices for the given object.
    /// Reads inline card bytes before the GcHeader.
    pub fn dirty_cards(&self, obj: GcRef) -> Vec<usize> {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(flags::HAS_CARDS) } {
            return Vec::new();
        }
        let type_id = unsafe { (*hdr).type_id() };
        let type_info = self.types.get(type_id);
        let length_offset = type_info.length_offset;
        let length = unsafe { *((obj.0 + length_offset) as *const usize) };
        let bytes = self.card_marking_bytes_for_length(length);
        let mut result = Vec::new();
        for bi in 0..bytes {
            let addr = Self::get_card_ptr(obj.0, bi);
            let cardbyte = unsafe { *addr };
            for bit in 0..8 {
                if cardbyte & (1 << bit) != 0 {
                    result.push(bi * 8 + bit);
                }
            }
        }
        result
    }

    /// incminimark.py:2009-2083: collect_cardrefs_to_nursery.
    /// Process `old_objects_with_cards_set` — for each object, scan
    /// dirty card ranges and copy nursery objects out.
    fn collect_cardrefs_to_nursery(&mut self) {
        while let Some(obj) = self.old_objects_with_cards_set.pop() {
            // incminimark.py:2016-2020
            let hdr = unsafe { header_of(obj) };
            debug_assert!(unsafe { (*hdr).has_flag(flags::HAS_CARDS) });
            debug_assert!(unsafe { (*hdr).has_flag(flags::CARDS_SET) });
            unsafe { (*hdr).clear_flag(flags::CARDS_SET) };

            // incminimark.py:2023-2026
            let type_id = unsafe { (*hdr).type_id() };
            let type_info = self.types.get(type_id);
            let length_offset = type_info.length_offset;
            let length = unsafe { *((obj + length_offset) as *const usize) };
            let bytes = self.card_marking_bytes_for_length(length);

            // incminimark.py:2033-2039: if !TRACK_YOUNG_PTRS, object is also
            // in old_objects_pointing_to_young and will be fully traced.
            // Just clear card bytes.
            if unsafe { !(*hdr).has_flag(flags::TRACK_YOUNG_PTRS) } {
                for bi in 0..bytes {
                    let p = Self::get_card_ptr(obj, bi);
                    unsafe {
                        *p = 0;
                    }
                }
                continue;
            }

            // incminimark.py:2041-2072: walk card bytes, trace dirty ranges.
            if type_info.items_have_gc_ptrs && type_info.item_size > 0 {
                let item_size = type_info.item_size;
                let base_size = type_info.size;
                let items_start = obj + base_size;
                let card_page_indices = 1usize << self.card_page_shift;

                let mut interval_start = 0usize;
                for bi in 0..bytes {
                    let p = Self::get_card_ptr(obj, bi);
                    let mut cardbyte = unsafe { *p };
                    unsafe {
                        *p = 0;
                    } // reset
                    let next_byte_start = interval_start + 8 * card_page_indices;

                    while cardbyte != 0 {
                        let mut interval_stop = interval_start + card_page_indices;
                        if cardbyte & 1 != 0 {
                            if interval_stop > length {
                                interval_stop = length;
                                if interval_stop <= interval_start {
                                    break;
                                }
                            }
                            // trace_and_drag_out_of_nursery_partial
                            for i in interval_start..interval_stop {
                                let slot = (items_start + i * item_size) as *mut GcRef;
                                let field_ref = unsafe { *slot };
                                self.assert_traced_slot_initialized(
                                    field_ref,
                                    slot as usize,
                                    obj,
                                    "minor_dirty_card_item",
                                    "minor_dirty_card",
                                );
                                if self.is_nursery_object_start(field_ref.0) {
                                    let new_ref = self.copy_nursery_object(
                                        field_ref.0,
                                        "minor_dirty_card_item_target",
                                        "minor_dirty_card",
                                        obj,
                                        slot as usize,
                                    );
                                    unsafe {
                                        *slot = new_ref;
                                    }
                                }
                            }
                        }
                        interval_start = interval_stop;
                        cardbyte >>= 1;
                    }
                    interval_start = next_byte_start;
                }
            } else {
                // No GC ptrs in items — just clear card bytes.
                for bi in 0..bytes {
                    let p = Self::get_card_ptr(obj, bi);
                    unsafe {
                        *p = 0;
                    }
                }
            }

            // incminimark.py:2079-2083: if incremental marking, re-add so the
            // object's outgoing references are (re)scanned before the sweep.
            // incminimark clears GCFLAG_VISITED here because its trace-list
            // drain re-marks on visit; pyre's `mark_object` does not set VISITED
            // on the popped object (the marking convention sets it at *push*
            // time — see `seed_major_root` and the remembered-set rescan above),
            // so keep the object black across the re-push. Clearing it would
            // leave the still-reachable card object white for the incremental
            // sweep to reclaim.
            if self.gc_state == GcState::Marking {
                unsafe { (*hdr).set_flag(flags::VISITED) };
                self.incr_state.more_gray_stack.push(obj);
            }
        }
    }

    /// Clear inline card bytes for a given object.
    pub fn clear_cards(&mut self, obj_addr: usize) {
        let hdr = unsafe { header_of(obj_addr) };
        if unsafe { !(*hdr).has_flag(flags::HAS_CARDS) } {
            unsafe { (*hdr).clear_flag(flags::CARDS_SET) };
            return;
        }
        let type_id = unsafe { (*hdr).type_id() };
        let type_info = self.types.get(type_id);
        let length_offset = type_info.length_offset;
        let length = unsafe { *((obj_addr + length_offset) as *const usize) };
        let bytes = self.card_marking_bytes_for_length(length);
        for bi in 0..bytes {
            let p = Self::get_card_ptr(obj_addr, bi);
            unsafe {
                *p = 0;
            }
        }
        unsafe { (*hdr).clear_flag(flags::CARDS_SET) };
    }

    // ── JIT integration hooks ──

    /// Fast-path write barrier for JIT-compiled code.
    ///
    /// Called from JIT-compiled code when a write barrier fires.
    /// Adds the object directly to the remembered set without the
    /// full flag-check logic of `do_write_barrier()`, because the
    /// JIT has already determined that the barrier is needed (via
    /// the inline flag test emitted by COND_CALL_GC_WB).
    ///
    /// Equivalent to incminimark's `jit_remember_young_pointer()`: the JIT has
    /// already passed the inline GCFLAG_TRACK_YOUNG_PTRS test emitted by
    /// COND_CALL_GC_WB, so go straight to the inlined remember logic without
    /// re-testing the flag.
    pub fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        self.remember_young_pointer(obj);
    }

    /// Returns true if the GC supports optimized conditional write barriers.
    ///
    /// When true, the JIT can emit COND_CALL_GC_WB (an inline flag test +
    /// conditional call) instead of a full write-barrier call. Nursery-based
    /// collectors always support this because the barrier check is a simple
    /// flag test on the object header.
    pub fn can_optimize_cond_call(&self) -> bool {
        true
    }

    /// Perform one incremental GC step. Called from JIT safepoints.
    ///
    /// If an incremental major cycle should start, it is initiated. If a cycle
    /// is already in progress, one bounded MARKING or SWEEPING step is
    /// performed. Returns true if any GC work was done.
    pub fn gc_step(&mut self) -> bool {
        if !self.enabled {
            return false;
        }
        if self.gc_state == GcState::Scanning && !self.threshold_reached(0) {
            return false;
        }
        self.major_collection_step();
        true
    }

    /// Reset the nursery while preserving pinned objects.
    ///
    /// Saves pinned object data, zeroes the nursery, restores pinned objects,
    /// and sets the free pointer past the highest pinned object.
    fn reset_nursery_with_pinned(&mut self) {
        let nursery_start = self.nursery.start_ptr() as usize;
        let nursery_end = nursery_start + self.nursery.size();

        // Collect (header_start, total_size, data) for each pinned object.
        let mut saved: Vec<(usize, usize, Vec<u8>)> = Vec::new();
        for &obj_addr in &self.pinned_objects {
            let type_id = unsafe { (*header_of(obj_addr)).type_id() };
            let payload_size = self.size_for_typeid(obj_addr, type_id, "pinned_snapshot");
            let total_size = (GcHeader::SIZE + payload_size).max(GcHeader::MIN_NURSERY_OBJ_SIZE);
            let total_size = (total_size + 7) & !7;
            let header_start = obj_addr - GcHeader::SIZE;
            let data = unsafe {
                std::slice::from_raw_parts(header_start as *const u8, total_size).to_vec()
            };
            saved.push((header_start, total_size, data));
        }

        // Rebuild the barrier range from the whole arena. A preceding
        // `reserve_nursery_gap` may have shortened `nursery_top` to the next
        // pinned object, exactly as upstream does while consuming one gap.
        unsafe {
            self.nursery.set_top_ptr(nursery_end as *const u8);
        }

        // Zero-fill the entire nursery.
        self.nursery.reset();

        // Restore pinned objects and compute the highest end.
        let mut max_end = nursery_start;
        for (header_start, total_size, data) in &saved {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), *header_start as *mut u8, *total_size);
            }
            let end = header_start + total_size;
            if end > max_end {
                max_end = end;
            }
        }

        // Set free pointer past the highest pinned object so new allocations
        // don't overwrite it.
        if max_end > nursery_start {
            unsafe {
                self.nursery.set_free_ptr(max_end as *mut u8);
            }
        }
    }

    /// incminimark.py:1121-1148 `pin`.
    pub fn pin(&mut self, obj: GcRef) -> bool {
        if self.pinned_objects_in_nursery >= self.max_number_of_pinned_objects {
            return false;
        }
        if obj.is_null() || !self.is_nursery_object_start(obj.0) {
            return false;
        }
        if self.is_pinned(obj) {
            return false;
        }
        let type_id = unsafe { (*header_of(obj.0)).type_id() };
        self.validate_type_id(type_id, obj.0, "pin");
        let info = self.types.get(type_id);
        // gctypelayout.py:89-92 `q_cannot_pin`: GC-pointer-bearing types,
        // weakrefs, and `customdata` (custom trace / destructor / finalizer)
        // cannot be pinned. A registered finalizer is represented by the
        // per-object flag in pyre's dynamic finalizer-queue surface.
        if info.has_gc_ptrs
            || info.is_weakref
            || info.custom_trace.is_some()
            || info.destructor.is_some()
            || unsafe { (*header_of(obj.0)).has_flag(flags::FINALIZER_REGISTERED) }
        {
            return false;
        }
        unsafe {
            (*header_of(obj.0)).set_flag(flags::PINNED);
        }
        self.pinned_objects.insert(obj.0);
        self.pinned_objects_in_nursery += 1;
        true
    }

    /// incminimark.py:1151-1155 `unpin`.
    pub fn unpin(&mut self, obj: GcRef) {
        assert!(self.is_pinned(obj), "unpin: object is already not pinned");
        unsafe {
            (*header_of(obj.0)).clear_flag(flags::PINNED);
        }
        self.pinned_objects.swap_remove(&obj.0);
        self.pinned_objects_in_nursery -= 1;
    }

    /// Check if an object is currently pinned.
    pub fn is_pinned(&self, obj: GcRef) -> bool {
        self.pinned_objects.contains(&obj.0)
    }

    /// Free memory associated with invalidated JIT compiled code.
    ///
    /// `code_ptr` and `size` identify the compiled code region to release.
    /// The region is looked up and removed from the compiled code registry
    /// so the GC no longer scans it for root references.
    pub fn jit_free(&mut self, code_ptr: usize, size: usize) {
        // Find and remove any compiled code region that matches the given range.
        self.compiled_code_registry
            .regions
            .retain(|r| !(r.code_start == code_ptr && r.code_size == size));
    }

    /// Number of objects in the remembered set (for testing / diagnostics).
    pub fn remembered_set_len(&self) -> usize {
        self.remembered_set.len()
    }
}

/// Safepoint GC map: records which frame slots contain GC references
/// at a specific program point (guard or call site).
///
/// The Cranelift backend builds these during compilation and stores them
/// alongside the compiled code. During collection, the GC uses them to
/// find live references on the stack.
#[derive(Debug, Clone)]
pub struct SafepointMap {
    /// Map from code offset to GcMap.
    pub entries: Vec<SafepointEntry>,
}

/// A single safepoint entry.
#[derive(Debug, Clone)]
pub struct SafepointEntry {
    /// Offset in the compiled code (bytes from function start).
    pub code_offset: u32,
    /// Bitmap of which frame slots contain GC references.
    pub gc_map: crate::GcMap,
}

impl SafepointMap {
    pub fn new() -> Self {
        SafepointMap {
            entries: Vec::new(),
        }
    }

    /// Add a safepoint entry.
    pub fn add(&mut self, code_offset: u32, gc_map: crate::GcMap) {
        self.entries.push(SafepointEntry {
            code_offset,
            gc_map,
        });
    }

    /// Look up the GcMap for a given code offset.
    pub fn lookup(&self, code_offset: u32) -> Option<&crate::GcMap> {
        self.entries
            .iter()
            .find(|e| e.code_offset == code_offset)
            .map(|e| &e.gc_map)
    }
}

impl Default for SafepointMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry of compiled code regions and their safepoint maps.
///
/// When the GC needs to scan the stack during collection, it uses the return
/// address to find which compiled code region is active, then looks up the
/// safepoint map to determine which frame slots contain GC references.
///
/// From rpython/jit/backend/llsupport/gc.py GcRootMap_asmgcc / GcRootMap_shadowstack.
pub struct CompiledCodeRegistry {
    /// Compiled code regions, sorted by start address for binary search.
    regions: Vec<CompiledCodeRegion>,
}

/// A single compiled code region with its safepoint map.
#[derive(Debug, Clone)]
pub struct CompiledCodeRegion {
    /// Start address of the compiled code.
    pub code_start: usize,
    /// Size of the compiled code in bytes.
    pub code_size: usize,
    /// Safepoint map for this region.
    pub safepoint_map: SafepointMap,
    /// Frame size in slots (each slot = 8 bytes).
    pub frame_size_slots: u32,
    /// JitCellToken number for identification.
    pub loop_token: u64,
}

impl CompiledCodeRegistry {
    pub fn new() -> Self {
        CompiledCodeRegistry {
            regions: Vec::new(),
        }
    }

    /// Register a compiled code region.
    pub fn register(&mut self, region: CompiledCodeRegion) {
        self.regions.push(region);
        // Keep sorted by code_start for binary search
        self.regions.sort_by_key(|r| r.code_start);
    }

    /// Unregister a compiled code region (e.g., when invalidating a loop).
    pub fn unregister(&mut self, loop_token: u64) {
        self.regions.retain(|r| r.loop_token != loop_token);
    }

    /// Look up a compiled code region containing the given return address.
    ///
    /// Returns the region and the offset within it.
    pub fn find_region(&self, return_addr: usize) -> Option<(&CompiledCodeRegion, u32)> {
        // Binary search for the region containing this address
        let idx = self
            .regions
            .binary_search_by(|r| {
                if return_addr < r.code_start {
                    std::cmp::Ordering::Greater
                } else if return_addr >= r.code_start + r.code_size {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()?;

        let region = &self.regions[idx];
        let offset = (return_addr - region.code_start) as u32;
        Some((region, offset))
    }

    /// Scan a compiled frame for GC references using the safepoint map.
    ///
    /// Given a return address (from the call stack) and the frame base pointer,
    /// enumerates all frame slots that contain GC references.
    ///
    /// # Safety
    /// `frame_base` must point to a valid JIT frame with at least
    /// `region.frame_size_slots` slots.
    pub unsafe fn scan_frame(
        &self,
        return_addr: usize,
        frame_base: *const usize,
    ) -> Vec<*mut GcRef> {
        let mut roots = Vec::new();

        let (region, offset) = match self.find_region(return_addr) {
            Some(r) => r,
            None => return roots,
        };

        let gc_map = match region.safepoint_map.lookup(offset) {
            Some(map) => map,
            None => return roots,
        };

        // Enumerate all slots marked as GC references
        for word_idx in 0..gc_map.ref_bitmap.len() {
            let mut bits = gc_map.ref_bitmap[word_idx];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let slot_idx = word_idx * 64 + bit;

                if slot_idx < region.frame_size_slots as usize {
                    let slot_ptr = unsafe { frame_base.add(slot_idx) } as *mut GcRef;
                    roots.push(slot_ptr);
                }

                bits &= bits - 1; // Clear lowest set bit
            }
        }

        roots
    }

    /// Number of registered regions.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

impl Default for CompiledCodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MiniMarkGC {
    fn default() -> Self {
        Self::new()
    }
}

impl GcAllocator for MiniMarkGC {
    fn debug_validate_oldgen_freeblocks(&self, site: &str) {
        self.oldgen.debug_validate_freeblocks(site);
    }

    fn alloc_nursery(&mut self, size: usize) -> GcRef {
        self.alloc_with_type(0, size)
    }

    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_with_type(type_id, size)
    }

    unsafe fn alloc_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe { self.alloc_with_type_rooted(type_id, size, root, needs_write_barrier) }
    }

    unsafe fn alloc_fast_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe { self.alloc_fast_with_type_rooted(type_id, size, root, needs_write_barrier) }
    }

    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
        self.alloc_with_type_no_collect(0, size)
    }

    fn alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_with_type_no_collect(type_id, size)
    }

    fn try_alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.try_alloc_with_type_no_collect(type_id, size)
    }

    unsafe fn try_alloc_fast_nursery_no_collect_typed(
        &mut self,
        type_id: u32,
        size: usize,
    ) -> GcRef {
        unsafe { self.try_alloc_fast_with_type_no_collect(type_id, size) }
    }

    unsafe fn try_alloc_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.try_alloc_with_type_no_collect_with_placement(type_id, size, needs_write_barrier)
        }
    }

    unsafe fn try_alloc_fast_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.try_alloc_fast_with_type_no_collect_with_placement(
                type_id,
                size,
                needs_write_barrier,
            )
        }
    }

    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef {
        let Some(payload_size) = Self::checked_varsize_payload_size(base_size, item_size, length)
        else {
            return GcRef(0);
        };
        self.alloc_with_type(0, payload_size)
    }

    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        let Some(payload_size) = Self::checked_varsize_payload_size(base_size, item_size, length)
        else {
            return GcRef(0);
        };
        self.alloc_with_type(type_id, payload_size)
    }

    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        let Some(payload_size) = Self::checked_varsize_payload_size(base_size, item_size, length)
        else {
            return GcRef(0);
        };
        self.alloc_with_type_no_collect(0, payload_size)
    }

    fn alloc_oldgen_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        let Some(total_size) = GcHeader::SIZE.checked_add(size) else {
            return GcRef(0);
        };
        self.alloc_in_oldgen(type_id, total_size)
    }

    fn collection_counts(&self) -> (usize, usize) {
        (self.minor_collections, self.major_collections)
    }

    fn get_write_barrier_descr(&self) -> Option<crate::WriteBarrierDescr> {
        let mut descr = crate::WriteBarrierDescr::for_current_gc();
        if self.card_page_shift == 0 {
            descr.jit_wb_cards_set = 0;
            descr.jit_wb_card_page_shift = 0;
            descr.jit_wb_cards_set_byteofs = 0;
            descr.jit_wb_cards_set_singlebyte = 0;
        } else {
            descr.jit_wb_card_page_shift = self.card_page_shift;
        }
        Some(descr)
    }

    fn type_alloc_is_plain(&self, type_id: u32) -> bool {
        (type_id as usize) < self.types.len() && {
            let info = self.types.get(type_id);
            info.destructor.is_none() && !info.is_weakref
        }
    }

    fn is_managed_heap_object(&self, addr: usize) -> bool {
        MiniMarkGC::is_managed_heap_object(self, addr)
    }

    fn is_nursery_object(&self, addr: usize) -> bool {
        self.is_nursery_object_start(addr)
    }

    fn nursery_bounds(&self) -> Option<(usize, usize)> {
        let start = self.nursery.start_ptr() as usize;
        Some((start, start + self.nursery.size()))
    }

    fn taggedpointers(&self) -> bool {
        self.config.taggedpointers
    }

    fn write_barrier(&mut self, obj: GcRef) {
        self.do_write_barrier(obj);
    }

    fn write_barrier_managed(&mut self, obj: GcRef) {
        self.do_write_barrier_managed(obj);
    }

    fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef) {
        self.jit_remember_young_pointer_from_array(obj);
    }

    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    ) {
        self.remember_young_pointer_from_array2(obj, index, card_page_shift);
    }

    fn collect_nursery(&mut self) {
        self.do_collect_nursery();
    }

    fn collect_full(&mut self) {
        self.do_collect_full();
    }

    fn collect_step(&mut self) -> crate::GcStepTransition {
        self.collect_step()
    }

    fn get_objects(&mut self, generation: i8, visitor: &mut dyn FnMut(GcRef)) {
        self.do_get_objects(generation, visitor)
    }

    fn get_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) {
        self.do_get_referents(obj, visitor)
    }

    fn is_tracked(&mut self, obj: GcRef) -> bool {
        self.object_is_tracked(obj.0)
    }

    fn get_rpy_memory_usage(&mut self, obj: GcRef) -> Option<usize> {
        self.rpy_memory_usage(obj)
    }

    fn get_rpy_type_index(&mut self, obj: GcRef) -> Option<usize> {
        self.rpy_type_index(obj)
    }

    fn get_rpy_roots(&mut self, visitor: &mut dyn FnMut(GcRef)) -> bool {
        self.do_get_rpy_roots(visitor)
    }

    fn get_rpy_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) -> bool {
        self.do_get_rpy_referents(obj, visitor)
    }

    fn dump_rpy_heap(&mut self, fd: i32) -> Result<bool, i32> {
        self.do_dump_rpy_heap(fd)
    }

    fn get_typeids_text(&self) -> Option<Vec<u8>> {
        Some(self.types.typeids_text())
    }

    fn get_typeids_list(&self) -> Option<Vec<usize>> {
        Some(self.types.typeids_list())
    }

    fn add_memory_pressure(&mut self, size: isize, object: GcRef) {
        self.do_add_memory_pressure(size, object);
    }

    fn total_memory_pressure(&mut self) -> isize {
        self.do_count_memory_pressure()
    }

    fn is_app_level_object(&mut self, obj: GcRef) -> bool {
        self.is_app_level_object_ref(obj)
    }

    fn collect_oldgen_nonmoving(&mut self) {
        self.do_collect_oldgen_nonmoving();
    }

    fn enable(&mut self) {
        self.enable();
    }

    fn disable(&mut self) {
        self.disable();
    }

    fn isenabled(&self) -> bool {
        self.isenabled()
    }

    fn register_finalizer(&mut self, fq_index: usize, obj: GcRef, trigger: FinalizerTriggerFn) {
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        assert!(fq_index <= self.finalizer_handlers.len());
        if fq_index == self.finalizer_handlers.len() {
            self.finalizer_handlers.push(FinalizerHandler {
                deque: VecDeque::new(),
                trigger,
            });
        }
        // `register_finalizer` is contracted to run at most once per object
        // (`rgc.py:648-649`). A second deque entry survives the first
        // `deal_with_objects_with_finalizers` pass through `new_with_finalizer`
        // (incminimark.py:2944-2946) and delivers the object a second time on
        // the next major collection, so honour the contract here rather than
        // leaving it to every caller.
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(flags::FINALIZER_REGISTERED) } {
            return;
        }
        unsafe { (*hdr).set_flag(flags::FINALIZER_REGISTERED) };
        if self.oldgen.contains(obj.0) {
            // Pyre's host allocations can be born directly in old-gen and an
            // explicit non-moving major intentionally skips the leading minor.
            // This is the post-`_trace_drag_out1` destination of PyPy's
            // probably-young deque, reached immediately for an already-old obj.
            self.old_objects_with_finalizers
                .push_back((obj.0, fq_index));
        } else {
            self.probably_young_objects_with_finalizers
                .push_back((obj.0, fq_index));
        }
    }

    fn finalizer_next_dead(&mut self, fq_index: usize) -> Option<GcRef> {
        self.finalizer_handlers
            .get_mut(fq_index)
            .and_then(|handler| handler.deque.pop_front())
            .map(GcRef)
    }

    fn id_or_identityhash(&mut self, obj_addr: usize) -> usize {
        self.id_or_identityhash(obj_addr)
    }

    unsafe fn add_root(&mut self, root: *mut GcRef) {
        unsafe { self.roots.add(root) };
    }

    fn remove_root(&mut self, root: *mut GcRef) {
        self.roots.remove(root);
    }

    fn nursery_free(&self) -> *mut u8 {
        self.nursery.free_ptr()
    }

    fn nursery_free_addr(&self) -> usize {
        self.nursery.free_addr()
    }

    fn nursery_top(&self) -> *const u8 {
        self.nursery.top_ptr()
    }

    fn nursery_top_addr(&self) -> usize {
        self.published_nursery_top.as_ptr() as usize
    }

    fn max_nursery_object_size(&self) -> usize {
        self.config.large_object_threshold
    }

    fn card_page_shift(&self) -> u32 {
        self.card_page_shift
    }

    fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        self.jit_remember_young_pointer(obj);
    }

    fn can_optimize_cond_call(&self) -> bool {
        self.can_optimize_cond_call()
    }

    fn gc_step(&mut self) -> bool {
        self.gc_step()
    }

    fn jit_free(&mut self, code_ptr: usize, size: usize) {
        self.jit_free(code_ptr, size);
    }

    fn pin(&mut self, obj: GcRef) -> bool {
        self.pin(obj)
    }

    fn unpin(&mut self, obj: GcRef) {
        self.unpin(obj);
    }

    fn is_pinned(&self, obj: GcRef) -> bool {
        self.is_pinned(obj)
    }

    fn register_type(&mut self, info: TypeInfo) -> u32 {
        self.types.register(info)
    }

    fn type_count(&self) -> usize {
        self.types.len()
    }

    fn heap_byte_stats(&self) -> (usize, usize) {
        (self.get_total_memory_used(), self.nursery.used())
    }

    fn gc_memory_stats(&self) -> crate::GcMemoryStats {
        // incminimark.py:3128-3154. The nursery contribution is its reserved
        // capacity, not its current bump-pointer fill.
        let nursery_size = self.config.nursery_size;
        crate::GcMemoryStats {
            total_gc_memory: self.get_total_memory_used() + nursery_size,
            total_allocated_memory: self.oldgen.total_allocated_bytes() + nursery_size,
            peak_memory: self.oldgen.peak_used_bytes() + nursery_size,
            peak_allocated_memory: self.oldgen.peak_allocated_bytes() + nursery_size,
            total_arena_memory: self.oldgen.arenas_bytes(),
            total_rawmalloced_memory: self.oldgen.rawmalloced_bytes(),
            peak_arena_memory: self.oldgen.peak_arena_bytes(),
            peak_rawmalloced_memory: self.oldgen.peak_rawmalloced_bytes(),
            nursery_size,
            total_gc_time_ms: (self.total_gc_time * 1000.0) as usize,
        }
    }

    fn major_threshold_reached(&self) -> bool {
        self.threshold_reached(0)
    }

    fn type_size(&self, type_id: u32) -> Option<usize> {
        if (type_id as usize) < self.types.len() {
            Some(self.types.get(type_id).size)
        } else {
            None
        }
    }

    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        // gc.py:563-590 GcLLDescr_framework
        //   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
        // RPython derives the typeid arithmetically:
        //   expected_typeid = classptr - sizeof_ti - type_info_group
        // pyre keeps an explicit vtable→type_id table populated via
        // register_vtable_for_type, mirroring the same contract.
        self.vtable_to_type_id.get(&classptr).copied()
    }

    fn register_vtable_for_type(&mut self, vtable: usize, type_id: u32) {
        self.vtable_to_type_id.insert(vtable, type_id);
    }

    /// gctypelayout.py:393-398 `encode_type_shapes_now` parity:
    /// closes the type-registration phase on the underlying
    /// `TypeRegistry`. The materialized `type_info_group` base
    /// address is stable from this point on, and every is_object
    /// type's preorder `subclassrange_{min,max}` is computed via
    /// `assign_inheritance_ids` (normalizecalls.py:373-389).
    fn freeze_types(&mut self) {
        self.types.freeze_types();
    }

    /// Owns a `TypeRegistry`, so a shape id can always be resolved against
    /// this allocator — including before anything has been registered in it.
    fn has_type_registry(&self) -> bool {
        true
    }

    /// gc.py:318 `GcLLDescr_framework.supports_guard_gc_type = True`.
    /// MiniMarkGC owns a `TypeRegistry` that materializes a `TYPE_INFO`
    /// table on demand, so the framework-equivalent flag is always
    /// true here. (Boehm-equivalent stubs override the trait default
    /// and leave it `false`.)
    fn supports_guard_gc_type(&self) -> bool {
        true
    }

    /// gc.py:631-642 `check_is_object` parity.
    ///
    /// Implemented line-by-line as the RPython source:
    ///     typeid = self.get_actual_typeid(gcptr)
    ///     base_type_info, shift_by, sizeof_ti = (
    ///         self.get_translated_info_for_typeinfo())
    ///     infobits_offset, IS_OBJECT_FLAG = (
    ///         self.get_translated_info_for_guard_is_object())
    ///     p = base_type_info + (typeid << shift_by) + infobits_offset
    ///     p = rffi.cast(rffi.CCHARP, p)
    ///     return (ord(p[0]) & IS_OBJECT_FLAG) != 0
    ///
    /// `get_actual_typeid` is the only majit-specific adaptation: managed
    /// objects carry a `GcHeader` immediately before the payload, while
    /// pyre's host-allocated foreign objects keep their classptr at
    /// offset 0 and rely on the `vtable→type_id` table populated via
    /// `register_vtable_for_type`.
    fn check_is_object(&self, gcref: GcRef) -> bool {
        if gcref.is_null() {
            return false;
        }
        // gc.py:634: typeid = self.get_actual_typeid(gcptr).
        let Some(typeid) = self.get_actual_typeid(gcref) else {
            return false;
        };
        // gc.py:636-637: gc_ll_descr lookups.
        let (base_type_info, shift_by, _sizeof_ti) = self.get_translated_info_for_typeinfo();
        let (infobits_offset, is_object_flag) = self.get_translated_info_for_guard_is_object();
        // gc.py:640-642: addr arithmetic + byte test.
        let typeid = typeid as usize;
        if typeid >= self.types.len() {
            return false;
        }
        let p = base_type_info + (typeid << shift_by) + infobits_offset;
        let byte = unsafe { *(p as *const u8) };
        (byte & is_object_flag) != 0
    }

    /// gc/base.py:380-383 `is_valid_gc_object` tagged-immediate test.
    /// Delegates to the inherent guard the collector uses internally,
    /// exposing it through the trait so backend-agnostic callers can ask
    /// whether an odd-valued constant address is an unboxed immediate.
    fn is_tagged_immediate(&self, addr: usize) -> bool {
        MiniMarkGC::is_tagged_immediate(self, addr)
    }

    /// incminimark.py:1117-1119 `can_move`: nursery membership alone. Pinning
    /// temporarily prevents a move but deliberately does not change this
    /// answer; callers use it to decide whether pinning is meaningful.
    fn can_move(&self, gcref: GcRef) -> bool {
        if gcref.is_null() || self.is_tagged_immediate(gcref.0) {
            return false;
        }
        self.is_in_nursery(gcref.0)
    }

    /// gc.py:592 `get_translated_info_for_typeinfo` parity.
    /// Returns `(type_info_group_base, shift_by, sizeof_ti)` — the
    /// base address of the materialized `type_info_group` table, the
    /// SIB-style scale `genop_guard_guard_is_object` /
    /// `genop_guard_guard_subclass` apply to the typeid register
    /// (x86/assembler.py:1934, 1967), and `rffi.sizeof(TYPE_INFO)`.
    ///
    /// RPython's 64-bit port sets `shift_by = 0` because its typeid
    /// is already `GROUP_MEMBER_OFFSET`, i.e. the raw byte offset of
    /// the member inside the group struct
    /// (translator/c/src/llgroup.h:36). majit keeps typeid as a
    /// small-integer index (returned by `register_type`) and encodes
    /// the per-entry stride here: `shift_by = log2(TypeEntry::STRIDE)`
    /// so the backend formula
    /// `base + (typeid << shift_by) + offset` lands on the correct
    /// `TypeEntry[typeid]` field. `sizeof_ti` stays equal to
    /// `rffi.sizeof(TYPE_INFO)` because that is the distance to the
    /// paired `CLASSTYPE` entry the `genop_guard_guard_subclass`
    /// formula needs (`+ sizeof_ti + offset2`
    /// x86/assembler.py:1968-1969, gctypelayout.py:359-374
    /// `add_vtable_after_typeinfo`).
    fn get_translated_info_for_typeinfo(&self) -> (usize, u8, usize) {
        let table = self.types.type_info_table();
        let base = table.as_ptr() as usize;
        (base, TypeEntry::SHIFT_BY, TypeInfoLayout::SIZE_OF_TI)
    }

    /// gc.py:619-622 `get_translated_info_for_guard_is_object` parity.
    /// Reads the instance state populated by `_setup_guard_is_object`
    /// and returns `(infobits_offset + infobits_offset_plus,
    /// T_IS_RPYTHON_INSTANCE_BYTE)`. Mirrors:
    ///
    /// ```python
    /// def get_translated_info_for_guard_is_object(self):
    ///     infobits_offset = rffi.cast(lltype.Signed, self._infobits_offset)
    ///     infobits_offset += self._infobits_offset_plus
    ///     return (infobits_offset, self._T_IS_RPYTHON_INSTANCE_BYTE)
    /// ```
    fn get_translated_info_for_guard_is_object(&self) -> (usize, u8) {
        let infobits_offset = self._infobits_offset + self._infobits_offset_plus;
        (infobits_offset, self._T_IS_RPYTHON_INSTANCE_BYTE)
    }

    /// x86/assembler.py:1951 `cpu.subclassrange_min_offset` parity.
    /// Byte offset of `subclassrange_min` inside `ClassTypeLayout`,
    /// the paired `rclass.CLASSTYPE` member that immediately follows
    /// the `TYPE_INFO` entry in the type_info_group
    /// (gctypelayout.py:359-374).
    fn subclassrange_min_offset(&self) -> usize {
        ClassTypeLayout::SUBCLASSRANGE_MIN_OFFSET
    }

    /// x86/assembler.py:1971-1974 codegen-time bounds lookup parity.
    /// Resolves `classptr` through the registered vtable→typeid map and
    /// returns the type's `(subclassrange_min, subclassrange_max)`.
    fn subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        let type_id = self.vtable_to_type_id.get(&classptr).copied()?;
        let info = self.types.get(type_id);
        Some((info.subclassrange_min, info.subclassrange_max))
    }

    /// Companion to `subclass_range` keyed by typeid. Used by the
    /// executor's `GuardSubclass` arm after it resolves `value.typeptr`
    /// via `get_actual_typeid` (llgraph/runner.py:1271-1281).
    fn typeid_subclass_range(&self, typeid: u32) -> Option<(i64, i64)> {
        if (typeid as usize) >= self.types.len() {
            return None;
        }
        let info = self.types.get(typeid);
        Some((info.subclassrange_min, info.subclassrange_max))
    }

    /// gc.py:624-629 `get_actual_typeid` parity.
    ///
    /// RPython reads the typeid from the GC header word
    /// (`llop.extract_ushort(llgroup.HALFWORD, hdr.tid)`). MiniMarkGC's
    /// managed objects use the same layout — `GcHeader::SIZE` bytes
    /// before the payload, type id in the lower `TYPE_ID_BITS`. pyre's
    /// host-allocated "foreign" objects do not carry a `GcHeader`; for
    /// those we recover the type id by reading the classptr (offset 0)
    /// and looking it up in `vtable_to_type_id`, which is what
    /// `register_vtable_for_type` populated.
    fn get_actual_typeid(&self, gcref: GcRef) -> Option<u32> {
        if gcref.is_null() {
            return None;
        }
        // gc/base.py:380 `is_valid_gc_object`: a tagged immediate carries no
        // header/classptr — reading offset 0 would deref the value bits.
        // Mirrors `can_move`'s guard.
        if self.is_tagged_immediate(gcref.0) {
            return None;
        }
        if self.is_managed_heap_object(gcref.0) {
            let header_addr = gcref.0.wrapping_sub(crate::header::GcHeader::SIZE);
            let header: crate::header::GcHeader = unsafe { *(header_addr as *const _) };
            return Some(header.type_id());
        }
        // Foreign object — read classptr at offset 0 and consult the
        // vtable→type_id table populated via register_vtable_for_type.
        let vtable = unsafe { *(gcref.0 as *const usize) };
        self.vtable_to_type_id.get(&vtable).copied()
    }

    /// Companion to `check_is_object` keyed by typeid. Reads the
    /// `T_IS_RPYTHON_INSTANCE` bit directly from the materialized
    /// TYPE_INFO entry (gctypelayout.py:642).
    fn typeid_is_object(&self, typeid: u32) -> Option<bool> {
        if (typeid as usize) >= self.types.len() {
            return None;
        }
        Some(self.types.get(typeid).is_object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static SHADOW_STACK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A supplied environment answers a name the process does not define, and
    /// yields to one it does. The name is not in [`GC_ENV_NAMES`], so a
    /// concurrently built collector cannot see this table.
    #[test]
    fn supplied_env_fills_in_only_what_the_process_lacks() {
        let absent = "PYRE_TEST_SUPPLIED_ENV_ABSENT";
        let present = "PYRE_TEST_SUPPLIED_ENV_PRESENT";
        // SAFETY: single-threaded within this test; the names are unique to it.
        unsafe { std::env::set_var(present, "2m") };

        assert_eq!(read_uint_from_env(absent), None);
        set_supplied_env(vec![
            (absent.to_string(), "1m".to_string()),
            (present.to_string(), "4m".to_string()),
        ]);
        assert_eq!(read_uint_from_env(absent), Some(1024 * 1024));
        assert_eq!(read_uint_from_env(present), Some(2 * 1024 * 1024));

        set_supplied_env(Vec::new());
        assert_eq!(read_uint_from_env(absent), None);
        unsafe { std::env::remove_var(present) };
    }

    /// Helper: create a GC with a small nursery for testing.
    fn test_gc(nursery_size: usize) -> MiniMarkGC {
        MiniMarkGC::with_config(GcConfig {
            nursery_size,
            large_object_threshold: nursery_size / 2,
            ..GcConfig::default()
        })
    }

    #[test]
    fn root_set_removes_lifo_and_out_of_order_roots() {
        let mut roots = RootSet::new();
        let mut a = GcRef(1);
        let mut b = GcRef(2);
        let mut c = GcRef(3);
        let (a, b, c) = (
            &mut a as *mut GcRef,
            &mut b as *mut GcRef,
            &mut c as *mut GcRef,
        );
        unsafe {
            roots.add(a);
            roots.add(b);
            roots.add(c);
        }

        roots.remove(c);
        assert_eq!(roots.roots, vec![a, b]);
        roots.remove(a);
        assert_eq!(roots.roots, vec![b]);
        roots.remove(b);
        assert!(roots.is_empty());
    }

    #[test]
    fn gc_memory_stats_follow_incminimark_selectors() {
        let nursery_size = 4096;
        let mut gc = test_gc(nursery_size);
        let initial = GcAllocator::gc_memory_stats(&gc);
        assert_eq!(initial.nursery_size, nursery_size);
        assert_eq!(initial.total_gc_memory, nursery_size);
        assert_eq!(initial.total_allocated_memory, nursery_size);
        assert_eq!(initial.total_gc_time_ms, 0);

        let tid = gc.register_type(TypeInfo::object(64));
        let _small = gc.alloc_in_oldgen(tid, GcHeader::SIZE + 64);
        let _large = gc.alloc_in_oldgen(
            tid,
            gc.oldgen.small_request_threshold() + std::mem::size_of::<usize>(),
        );
        let after = GcAllocator::gc_memory_stats(&gc);
        assert_eq!(
            after.total_gc_memory,
            gc.get_total_memory_used() + nursery_size
        );
        assert_eq!(after.total_arena_memory, gc.oldgen.arenas_bytes());
        assert_eq!(
            after.total_rawmalloced_memory,
            gc.oldgen.rawmalloced_bytes()
        );
        assert!(after.peak_memory >= after.total_gc_memory);
        assert!(after.peak_allocated_memory >= after.total_allocated_memory);

        gc.do_collect_nursery();
        assert!(GcAllocator::gc_memory_stats(&gc).total_gc_time_ms <= 60_000);
    }

    #[test]
    fn memory_pressure_matches_framework_field_and_incminimark_threshold() {
        let word = std::mem::size_of::<isize>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(word).with_memory_pressure_offset(0));
        assert_ne!(
            crate::trace::encode_type_shape(gc.types.get(tid), tid)
                & TypeInfoLayout::T_HAS_MEMORY_PRESSURE,
            0
        );

        let mut object = gc.alloc_with_type(tid, word);
        unsafe { gc.roots.add(&mut object) };
        let threshold = gc.next_major_collection_threshold;
        gc.do_add_memory_pressure(123, object);
        assert_eq!(unsafe { *(object.0 as *const isize) }, 123);
        assert_eq!(gc.do_count_memory_pressure(), 123);
        assert_eq!(
            gc.next_major_collection_threshold,
            threshold - 123.0 - (2 * std::mem::size_of::<usize>()) as f64
        );

        // The object-bearing transform uses bare_setfield, not +=, and a
        // negative estimate releases the object's previously owned pressure.
        gc.do_add_memory_pressure(-7, object);
        assert_eq!(gc.do_count_memory_pressure(), -7);

        // Upstream forces the next nursery slow path once pressure drives the
        // threshold below zero.
        gc.next_major_collection_threshold = 0.0;
        let free = gc.nursery.free_ptr();
        gc.do_add_memory_pressure(1, GcRef::NULL);
        assert_eq!(gc.nursery.top_ptr(), free.cast_const());
        assert_eq!(
            gc.published_nursery_top.load(Ordering::Acquire),
            free as usize
        );
        gc.roots.clear();
    }

    /// A born-old allocation that crosses the next-major threshold asks for a
    /// collection the way `external_malloc` (incminimark.py:987-994) does — and
    /// asks the consumer first, every time.
    ///
    /// The refusing arm is the load-bearing one. The threshold stays reached
    /// until a major completes, so arming past a consumer that will not answer
    /// re-arms on every following allocation; the compiled back-edge poll then
    /// fails continuously and the guard grows a bridge chain that stops
    /// returning to the dispatch loop, starving the rest of the eval-breaker
    /// word — signals included.
    #[test]
    fn born_old_allocation_past_the_threshold_asks_the_consumer_every_time() {
        fn refuse() -> bool {
            false
        }
        fn accept() -> bool {
            true
        }

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::object(64));
        // Put the threshold within reach of any single allocation.
        gc.next_major_collection_threshold = gc.get_total_memory_used() as f64;
        let size = GcHeader::SIZE + 64;
        majit_ir::eval_breaker_word::take_gc();

        set_deferred_major_request_probe(None);
        gc.alloc_in_oldgen(tid, size);
        assert!(
            !majit_ir::eval_breaker_word::take_gc(),
            "no consumer installed, so no request should be armed"
        );

        set_deferred_major_request_probe(Some(refuse));
        for _ in 0..4 {
            gc.alloc_in_oldgen(tid, size);
        }
        assert!(
            !majit_ir::eval_breaker_word::take_gc(),
            "the consumer refused, so none of the four allocations may arm"
        );

        set_deferred_major_request_probe(Some(accept));
        gc.alloc_in_oldgen(tid, size);
        assert!(majit_ir::eval_breaker_word::take_gc());

        set_deferred_major_request_probe(None);
        majit_ir::eval_breaker_word::take_gc();
    }

    #[test]
    fn get_objects_walks_referents_filters_rpython_structs_and_generations() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let mut holder_info = TypeInfo::object(ptr_size);
        holder_info.has_gc_ptrs = true;
        holder_info.gc_ptr_offsets = vec![0];
        let holder_tid = gc.register_type(holder_info);
        let raw_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::object(ptr_size));

        let leaf = gc.alloc_with_type(leaf_tid, ptr_size);
        let raw = gc.alloc_with_type(raw_tid, ptr_size);
        let mut holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe {
            *(raw.0 as *mut GcRef) = leaf;
            *(holder.0 as *mut GcRef) = raw;
            gc.roots.add(&mut holder);
        }

        let get_objects = |gc: &mut MiniMarkGC, generation| {
            let mut result = Vec::new();
            gc.do_get_objects(generation, &mut |gcref| result.push(gcref));
            result
        };
        let all = get_objects(&mut gc, -1);
        assert_eq!(all.len(), 2);
        assert!(all.contains(&holder));
        assert!(all.contains(&leaf));
        assert_eq!(get_objects(&mut gc, 0).len(), 2);
        assert!(get_objects(&mut gc, 1).is_empty());
        assert!(get_objects(&mut gc, 2).is_empty());
        for object in [holder, raw, leaf] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(flags::EXTRA) });
        }

        gc.do_collect_nursery();
        let raw = unsafe { *(holder.0 as *const GcRef) };
        let leaf = unsafe { *(raw.0 as *const GcRef) };
        assert!(get_objects(&mut gc, 0).is_empty());
        let old = get_objects(&mut gc, 2);
        assert_eq!(old.len(), 2);
        assert!(old.contains(&holder));
        assert!(old.contains(&leaf));
        for object in [holder, raw, leaf] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(flags::EXTRA) });
        }
        gc.roots.clear();
    }

    #[cfg(unix)]
    #[test]
    fn dump_rpy_heap_uses_native_words_root_marker_and_type_indexes() {
        use std::os::fd::AsRawFd;

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let holder_tid = gc.register_type(TypeInfo::object_with_gc_ptrs(ptr_size, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::object(ptr_size));
        let leaf = gc.alloc_with_type(leaf_tid, ptr_size);
        let mut holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe {
            *(holder.0 as *mut GcRef) = leaf;
            gc.roots.add(&mut holder);
        }

        let path = std::env::temp_dir().join(format!(
            "majit-gc-heap-dump-{}-{:p}",
            std::process::id(),
            &gc
        ));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert_eq!(gc.do_dump_rpy_heap(file.as_raw_fd()), Ok(true));
        drop(file);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(bytes.len() % std::mem::size_of::<isize>(), 0);
        let words: Vec<isize> = bytes
            .chunks_exact(std::mem::size_of::<isize>())
            .map(|chunk| isize::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        let marker = words.windows(4).position(|w| w == [0, 0, 0, -1]).unwrap();

        let holder_pos = words
            .iter()
            .position(|&word| word == holder.0 as isize)
            .unwrap();
        assert!(holder_pos < marker);
        assert_eq!(words[holder_pos + 1], holder_tid as isize + 1);
        assert_eq!(words[holder_pos + 2], ptr_size as isize);
        assert_eq!(words[holder_pos + 3], leaf.0 as isize);
        let leaf_pos = marker
            + 4
            + words[marker + 4..]
                .iter()
                .position(|&word| word == leaf.0 as isize)
                .unwrap();
        assert!(leaf_pos > marker);
        assert_eq!(words[leaf_pos + 1], leaf_tid as isize + 1);
        assert_eq!(words[leaf_pos + 3], -1);

        assert_eq!(gc.do_dump_rpy_heap(-1), Err(libc::EBADF));
        for object in [holder, leaf] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(flags::EXTRA) });
        }
        assert!(gc.types.typeids_text().starts_with(b"member0"));
        assert_eq!(gc.types.typeids_list(), vec![0, 0, 1]);
        gc.roots.clear();
    }

    /// `referents.py:66-70` terminates a branch on `try_cast_gcref_to_w_root`,
    /// so the two rejections that helper opens with — the dummy flag and the
    /// missing typedef — look *through* a node here rather than reporting it.
    #[test]
    fn get_referents_looks_through_a_hidden_or_dummy_node() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let object_tid = gc.register_type(TypeInfo::object_with_gc_ptrs(ptr_size, vec![0]));
        let mut hidden_info = TypeInfo::object_with_gc_ptrs(ptr_size, vec![0]);
        hidden_info.hide_from_app_level_inspector = true;
        let hidden_tid = gc.register_type(hidden_info);

        // holder -> hidden -> leaf, and holder -> dummy -> tail.
        let leaf = gc.alloc_with_type(object_tid, ptr_size);
        let tail = gc.alloc_with_type(object_tid, ptr_size);
        let hidden = gc.alloc_with_type(hidden_tid, ptr_size);
        let dummy = gc.alloc_with_type(object_tid, ptr_size);
        let mut holder = gc.alloc_with_type(object_tid, ptr_size);
        unsafe {
            (*header_of(dummy.0)).set_flag(flags::DUMMY);
            *(hidden.0 as *mut GcRef) = leaf;
            *(dummy.0 as *mut GcRef) = tail;
            *(holder.0 as *mut GcRef) = hidden;
            gc.roots.add(&mut holder);
        }

        let mut referents = Vec::new();
        gc.do_get_referents(holder, &mut |gcref| referents.push(gcref));
        assert_eq!(referents, vec![leaf]);

        unsafe { *(holder.0 as *mut GcRef) = dummy };
        let mut referents = Vec::new();
        gc.do_get_referents(holder, &mut |gcref| referents.push(gcref));
        assert_eq!(referents, vec![tail]);

        for object in [holder, hidden, dummy, leaf, tail] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(flags::EXTRA) });
        }
        gc.roots.clear();
    }

    #[test]
    fn get_referents_looks_through_rpython_structs_and_stops_at_objects() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let pair_info = || TypeInfo::with_gc_ptrs(2 * ptr_size, vec![0, ptr_size]);
        let raw_tid = gc.register_type(pair_info());
        let mut object_info = pair_info();
        object_info.is_object = true;
        let object_tid = gc.register_type(object_info);

        // holder -> raw -> {near, far}; `far` in turn points back at `holder`,
        // which must not be reported: the walk stops at the first object.
        let near = gc.alloc_with_type(object_tid, 2 * ptr_size);
        let far = gc.alloc_with_type(object_tid, 2 * ptr_size);
        let raw = gc.alloc_with_type(raw_tid, 2 * ptr_size);
        let mut holder = gc.alloc_with_type(object_tid, 2 * ptr_size);
        unsafe {
            *(raw.0 as *mut GcRef) = near;
            *((raw.0 + ptr_size) as *mut GcRef) = far;
            *(far.0 as *mut GcRef) = holder;
            *(holder.0 as *mut GcRef) = raw;
            gc.roots.add(&mut holder);
        }

        let mut referents = Vec::new();
        gc.do_get_referents(holder, &mut |gcref| referents.push(gcref));
        assert_eq!(referents.len(), 2);
        assert!(referents.contains(&near));
        assert!(referents.contains(&far));
        assert!(!referents.contains(&raw));
        assert!(!referents.contains(&holder));
        for object in [holder, raw, near, far] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(flags::EXTRA) });
        }

        // A self-referential object terminates instead of looping forever.
        let mut referents = Vec::new();
        gc.do_get_referents(far, &mut |gcref| referents.push(gcref));
        assert_eq!(referents, vec![holder]);
        gc.roots.clear();
    }

    #[test]
    fn rpy_inspector_keeps_raw_roots_and_referents_unexpanded() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let raw_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let object_tid = gc.register_type(TypeInfo::object_with_gc_ptrs(ptr_size, vec![0]));
        let mut hidden_info = TypeInfo::object(ptr_size);
        hidden_info.hide_from_app_level_inspector = true;
        let hidden_tid = gc.register_type(hidden_info);

        let leaf = gc.alloc_with_type(object_tid, ptr_size);
        let raw = gc.alloc_with_type(raw_tid, ptr_size);
        let hidden = gc.alloc_with_type(hidden_tid, ptr_size);
        let mut holder = gc.alloc_with_type(object_tid, ptr_size);
        unsafe {
            *(raw.0 as *mut GcRef) = leaf;
            *(holder.0 as *mut GcRef) = raw;
            gc.roots.add(&mut holder);
        }

        let mut roots = Vec::new();
        assert!(gc.do_get_rpy_roots(&mut |root| roots.push(root)));
        assert!(roots.contains(&holder));

        let mut referents = Vec::new();
        assert!(gc.do_get_rpy_referents(holder, &mut |child| { referents.push(child) }));
        assert_eq!(referents, vec![raw]);
        assert!(gc.is_app_level_object_ref(holder));
        assert!(!gc.is_app_level_object_ref(raw));
        assert!(!gc.is_app_level_object_ref(hidden));
        gc.roots.clear();
    }

    #[test]
    fn is_tracked_follows_the_type_not_the_heap_the_instance_landed_in() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let atomic_tid = gc.register_type(TypeInfo::object(ptr_size));
        let holder_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            ptr_size,
            atomic_tid,
            vec![0],
        ));

        let atomic = gc.alloc_with_type(atomic_tid, ptr_size);
        let mut holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe {
            *(holder.0 as *mut GcRef) = atomic;
            gc.roots.add(&mut holder);
        }
        assert!(!gc.object_is_tracked(atomic.0));
        assert!(gc.object_is_tracked(holder.0));

        // The same two types answer the same way once promoted out of the
        // nursery — the heap the instance landed in must not matter.
        gc.do_collect_nursery();
        let atomic = unsafe { *(holder.0 as *const GcRef) };
        assert!(!gc.object_is_tracked(atomic.0));
        assert!(gc.object_is_tracked(holder.0));

        // An address the collector does not own is never tracked.
        let mut off_heap = 0usize;
        assert!(!gc.object_is_tracked(&mut off_heap as *mut usize as usize));
        gc.roots.clear();
    }

    #[test]
    fn rpy_introspection_reports_payload_size_and_positive_type_group_index() {
        let mut gc = test_gc(4096);
        let fixed_tid = gc.register_type(TypeInfo::simple(24));
        let var_tid = gc.register_type(TypeInfo::varsize(8, 3, 0, false, Vec::new()));

        let fixed = gc.alloc_with_type(fixed_tid, 24);
        let var = gc.alloc_varsize_typed(var_tid, 8, 3, 2);
        unsafe { *(var.0 as *mut usize) = 2 };

        // inspector.py:get_rpy_memory_usage calls get_size_incl_hash, whose
        // incminimark implementation excludes the GC header. Variable objects
        // are rounded for the following arena allocation (8 + 2 * 3 -> 16).
        assert_eq!(gc.rpy_memory_usage(fixed), Some(24));
        assert_eq!(gc.rpy_memory_usage(var), Some(16));
        assert_eq!(gc.object_total_size(fixed.0), GcHeader::SIZE + 24);

        // TypeLayoutBuilder reserves group member zero as a dummy.
        assert_eq!(gc.rpy_type_index(fixed), Some(1));
        assert_eq!(gc.rpy_type_index(var), Some(2));
        assert_eq!(gc.rpy_memory_usage(GcRef::NULL), None);
        assert_eq!(gc.rpy_type_index(GcRef::NULL), None);
    }

    #[test]
    fn get_objects_includes_live_objects_registered_with_finalizers() {
        fn trigger() {}

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let object_tid = gc.register_type(TypeInfo::object(ptr_size));
        let object = gc.alloc_with_type(object_tid, ptr_size);
        gc.register_finalizer(0, object, trigger);

        let mut objects = Vec::new();
        gc.do_get_objects(-1, &mut |gcref| objects.push(gcref));
        assert!(objects.contains(&object));
    }

    #[test]
    fn write_barrier_descr_exposes_minimark_card_geometry() {
        let gc = test_gc(4096);
        let descr = gc
            .get_write_barrier_descr()
            .expect("MiniMark must expose its write-barrier descriptor");
        assert_eq!(descr.jit_wb_if_flag, flags::TRACK_YOUNG_PTRS);
        assert_eq!(descr.jit_wb_cards_set, flags::CARDS_SET);
        assert_eq!(descr.jit_wb_card_page_shift, gc.card_page_shift);
    }

    #[test]
    fn write_barrier_descr_disables_cards_with_collector_config() {
        let gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 4096,
            large_object_threshold: 2048,
            card_page_indices: 0,
            ..GcConfig::default()
        });
        let descr = gc
            .get_write_barrier_descr()
            .expect("MiniMark must retain its generic write barrier");
        assert_eq!(descr.jit_wb_if_flag, flags::TRACK_YOUNG_PTRS);
        assert_eq!(descr.jit_wb_cards_set, 0);
        assert_eq!(descr.jit_wb_card_page_shift, 0);
    }

    #[test]
    fn test_basic_alloc() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(0, 16);
        assert!(!obj.is_null());
        assert!(gc.is_in_nursery(obj.0));
    }

    #[test]
    fn collect_and_reserve_uses_gap_before_pinned_object() {
        fn arrange_pinned_tail(gc: &mut MiniMarkGC) -> (u32, GcRef) {
            let pinned_tid = gc.register_type(TypeInfo::simple(440));
            let result_tid = gc.register_type(TypeInfo::simple(400));
            let _dead_prefix = gc.alloc_with_type(pinned_tid, 440);
            let pinned = gc.alloc_with_type(pinned_tid, 440);
            assert!(gc.pin(pinned));
            (result_tid, pinned)
        }

        // incminimark.py:865-930 walks back to the free gap before a pinned
        // object. The old fallback returned an old-gen object here, violating
        // the fresh nursery allocation contract used by the GC rewrite.
        let mut gc = test_gc(1024);
        let (result_tid, mut pinned) = arrange_pinned_tail(&mut gc);
        unsafe { gc.roots.add(&mut pinned) };
        let result = gc.alloc_with_type(result_tid, 400);
        assert!(gc.is_in_nursery(result.0));
        assert!(result.0 < pinned.0);
        assert_eq!(
            unsafe { &*(gc.nursery_top_addr() as *const AtomicUsize) }.load(Ordering::Acquire),
            pinned.0 - GcHeader::SIZE,
            "compiled allocators must stop at the same pinned barrier",
        );
        gc.roots.clear();

        // The rooted slow path has the same collect-and-reserve contract and
        // must continue reporting that initialization needs no write barrier.
        let mut gc = test_gc(1024);
        let (result_tid, mut pinned) = arrange_pinned_tail(&mut gc);
        unsafe { gc.roots.add(&mut pinned) };
        let mut root = GcRef::NULL;
        let mut needs_write_barrier = true;
        let result = unsafe {
            gc.alloc_with_type_rooted(result_tid, 400, &mut root, &mut needs_write_barrier)
        };
        assert!(gc.is_in_nursery(result.0));
        assert!(result.0 < pinned.0);
        assert!(!needs_write_barrier);
        gc.roots.clear();
    }

    /// incminimark.py:2601-2615 `PYPY_GC_MAX` out-of-memory policy: a bounded
    /// major collection over `max_heap_size` signals OOM the first time
    /// (`oom_pending` so the triggering allocation returns NULL) and aborts on
    /// the second occurrence. `max_heap_size` below `min_heap_size` makes any
    /// threshold bounded, so the decision fires on an otherwise empty heap.
    #[test]
    fn bounded_max_heap_size_signals_oom_then_aborts() {
        let mut gc = test_gc(4096);
        // PYPY_GC_MAX = 1 byte (below min_heap_size), so set_major_threshold_from
        // caps at 1 and reports `bounded`.
        gc.max_heap_size = 1.0;
        // The allocation that triggered the collection is larger than the
        // remaining headroom (1 - total_memory_used), so threshold_reached holds.
        gc.pending_reserving_size = 4096;

        // First bounded breach: flag + signal, no abort.
        gc.gc_state = GcState::Sweeping;
        gc.finish_incremental_cycle();
        assert!(
            gc.max_heap_size_already_raised,
            "first bounded breach records max_heap_size_already_raised"
        );
        assert!(
            gc.oom_pending,
            "first bounded breach asks the allocation to fail (NULL)"
        );

        // Second bounded breach aborts (out_of_memory -> fatalerror == panic).
        gc.pending_reserving_size = 4096;
        gc.gc_state = GcState::Sweeping;
        let second = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gc.finish_incremental_cycle();
        }));
        assert!(second.is_err(), "second bounded breach aborts the process");
    }

    /// The default (unbounded, `max_heap_size == 0`) config never sets `bounded`,
    /// so a completed major collection leaves the OOM signals untouched.
    #[test]
    fn unbounded_heap_never_signals_oom() {
        let mut gc = test_gc(4096);
        assert_eq!(gc.max_heap_size, 0.0);
        gc.pending_reserving_size = 4096;
        gc.gc_state = GcState::Sweeping;
        gc.finish_incremental_cycle();
        assert!(!gc.max_heap_size_already_raised);
        assert!(!gc.oom_pending);
        assert_eq!(gc.gc_state, GcState::Finalizing);
    }

    /// incminimark.py:2617-2631 keeps finalizer execution in its own major
    /// step. In particular, a handler cannot run in the SWEEPING ->
    /// FINALIZING transition; the following step first publishes SCANNING and
    /// only then invokes it, so a recursive collection can start safely.
    #[test]
    fn finalizer_triggers_run_in_a_distinct_finalizing_step() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static TRIGGERS: AtomicUsize = AtomicUsize::new(0);
        fn trigger() {
            TRIGGERS.fetch_add(1, Ordering::Relaxed);
        }

        TRIGGERS.store(0, Ordering::Relaxed);
        let mut gc = test_gc(4096);
        gc.finalizer_handlers.push(FinalizerHandler {
            deque: VecDeque::from([0xfeed]),
            trigger,
        });
        gc.gc_state = GcState::Sweeping;

        gc.finish_incremental_cycle();
        assert_eq!(gc.gc_state, GcState::Finalizing);
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 0);

        gc.major_collection_step();
        assert_eq!(gc.gc_state, GcState::Scanning);
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn explicit_collect_step_runs_one_minor_and_one_major_transition() {
        use crate::GcStepTransition as Step;

        let mut gc = test_gc(4096);
        gc.disable();
        let mut transitions = Vec::new();
        loop {
            let transition = gc.collect_step();
            transitions.push((transition.old_state, transition.new_state));
            if transition.is_done() {
                break;
            }
        }
        assert_eq!(
            transitions,
            vec![
                (Step::SCANNING, Step::MARKING),
                (Step::MARKING, Step::SWEEPING),
                (Step::SWEEPING, Step::FINALIZING),
                (Step::FINALIZING, Step::SCANNING),
            ]
        );
        assert!(!gc.isenabled());
        assert_eq!(gc.minor_collections, 4);
    }

    #[test]
    fn published_nursery_top_tracks_real_top_across_minor_collection() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let published_addr = gc.nursery_top_addr();
        let real_top = gc.nursery_top() as usize;
        assert_eq!(
            unsafe { &*(published_addr as *const AtomicUsize) }.load(Ordering::Acquire),
            real_top
        );

        // Prove the collection reset republishes the real limit instead of
        // merely leaving the construction-time value untouched.
        unsafe { &*(published_addr as *const AtomicUsize) }.store(0, Ordering::Release);
        gc.collect_nursery();
        assert_eq!(gc.nursery_top_addr(), published_addr);
        assert_eq!(
            unsafe { &*(published_addr as *const AtomicUsize) }.load(Ordering::Acquire),
            real_top
        );

        let obj = gc.alloc_nursery(16);
        assert!(!obj.is_null());
        assert!(gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_multiple_allocs() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let mut refs = Vec::new();
        for _ in 0..10 {
            refs.push(gc.alloc_with_type(0, 16));
        }

        // All should be non-null and distinct.
        for i in 0..refs.len() {
            assert!(!refs[i].is_null());
            for j in (i + 1)..refs.len() {
                assert_ne!(refs[i], refs[j]);
            }
        }
    }

    #[test]
    fn test_large_object_goes_to_oldgen() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(1024));

        // 1024 > large_object_threshold (512), so goes to old gen.
        let obj = gc.alloc_with_type(0, 1024);
        assert!(!obj.is_null());
        assert!(!gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_nursery_collection_basic() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate an object and root it.
        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write something to the object payload.
        unsafe {
            *(obj.0 as *mut u64) = 0xDEADBEEF;
        }

        // Root it.
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger collection.
        gc.collect_nursery();

        // The root should now point to old gen.
        assert!(!gc.is_in_nursery(root.0));
        assert!(!root.is_null());

        // The data should be preserved.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xDEADBEEF);

        gc.roots.clear();
    }

    #[test]
    fn test_unrooted_object_dies() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate but don't root.
        let _obj = gc.alloc_with_type(0, 16);

        // Collection should run without issues.
        gc.collect_nursery();

        // Nursery is reset, the object is gone.
        assert_eq!(gc.nursery.used(), 0);
    }

    #[test]
    fn test_fill_nursery_triggers_collection() {
        let mut gc = test_gc(256);
        gc.register_type(TypeInfo::simple(16));

        // Keep allocating until we must have triggered at least one collection.
        for _ in 0..100 {
            gc.alloc_with_type(0, 16);
        }
        assert!(gc.minor_collections > 0);
    }

    #[test]
    fn rooted_collecting_alloc_keeps_fast_bump_in_nursery() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut child = gc.alloc_with_type(tid, 16);
        let roots_before = gc.roots.len();
        let mut needs_write_barrier = true;

        let parent = unsafe {
            gc.alloc_with_type_rooted(tid, 16, &mut child as *mut GcRef, &mut needs_write_barrier)
        };

        assert!(gc.is_in_nursery(child.0));
        assert!(gc.is_in_nursery(parent.0));
        assert!(!needs_write_barrier);
        assert_eq!(gc.minor_collections, 0);
        assert_eq!(gc.roots.len(), roots_before);
    }

    #[test]
    fn rooted_collecting_alloc_forwards_child_only_on_slow_path() {
        let mut gc = test_gc(256);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut child = gc.alloc_with_type(tid, 16);

        // A 16-byte payload plus the GC header occupies 24 aligned bytes.
        // Leave less than one object free without collecting.
        while gc.nursery.remaining() >= GcHeader::SIZE + 16 {
            let filler = gc.alloc_with_type_no_collect(tid, 16);
            assert!(gc.is_in_nursery(filler.0));
        }
        let roots_before = gc.roots.len();
        let mut needs_write_barrier = true;

        let parent = unsafe {
            gc.alloc_with_type_rooted(tid, 16, &mut child as *mut GcRef, &mut needs_write_barrier)
        };

        assert_eq!(gc.minor_collections, 1);
        assert!(!gc.is_in_nursery(child.0));
        assert!(gc.is_in_nursery(parent.0));
        assert!(!needs_write_barrier);
        assert_eq!(gc.roots.len(), roots_before);
    }

    #[test]
    fn rooted_collecting_alloc_reports_oldgen_creation_barrier() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut child = gc.alloc_with_type(tid, 16);
        let mut needs_write_barrier = false;
        let payload_size = gc.config.large_object_threshold;

        let parent = unsafe {
            gc.alloc_with_type_rooted(
                tid,
                payload_size,
                &mut child as *mut GcRef,
                &mut needs_write_barrier,
            )
        };

        assert!(!gc.is_in_nursery(parent.0));
        assert!(needs_write_barrier);
    }

    #[test]
    fn no_collect_alloc_reports_nursery_and_oldgen_placement() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut needs_write_barrier = true;

        let young = unsafe {
            gc.try_alloc_with_type_no_collect_with_placement(tid, 16, &mut needs_write_barrier)
        };
        assert!(gc.is_in_nursery(young.0));
        assert!(!needs_write_barrier);

        while gc.nursery.remaining() >= GcHeader::SIZE + 16 {
            let filler = gc.alloc_with_type_no_collect(tid, 16);
            assert!(gc.is_in_nursery(filler.0));
        }
        let spilled = unsafe {
            gc.try_alloc_with_type_no_collect_with_placement(tid, 16, &mut needs_write_barrier)
        };
        assert!(!gc.is_in_nursery(spilled.0));
        assert!(needs_write_barrier);

        let oversized = unsafe {
            gc.try_alloc_with_type_no_collect_with_placement(
                tid,
                gc.config.large_object_threshold,
                &mut needs_write_barrier,
            )
        };
        assert!(!gc.is_in_nursery(oversized.0));
        assert!(needs_write_barrier);
    }

    // --- Lightweight destructor tests (incminimark.py:2884-2912 parity) ---
    //
    // The counting destructor below deliberately does NOT `drop_in_place`
    // the dummy payload (the test allocations are raw bytes, not a real
    // `Drop` type); it only records that the collector dispatched it on
    // the right object at the right time. A serializing lock keeps the
    // shared counter races-free under cargo's parallel test runner.

    static DESTRUCTOR_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static DESTRUCTOR_RUNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static DESTRUCTOR_LAST_ADDR: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    unsafe fn counting_destructor(obj_addr: usize) {
        DESTRUCTOR_RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        DESTRUCTOR_LAST_ADDR.store(obj_addr, std::sync::atomic::Ordering::SeqCst);
    }

    fn destructor_runs() -> usize {
        DESTRUCTOR_RUNS.load(std::sync::atomic::Ordering::SeqCst)
    }

    #[test]
    fn destructor_runs_on_nursery_death() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock().unwrap();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_destructor(16, counting_destructor));

        let obj = gc.alloc_with_type(tid, 16);
        // Recorded on the young-destructor list at allocation.
        assert_eq!(gc.young_objects_with_destructors, vec![obj.0]);

        // Not rooted → dies on the next minor collection.
        gc.collect_nursery();

        assert_eq!(destructor_runs(), 1);
        assert_eq!(
            DESTRUCTOR_LAST_ADDR.load(std::sync::atomic::Ordering::SeqCst),
            obj.0
        );
        // Both lists drained: dead object never reaches the old list.
        assert!(gc.young_objects_with_destructors.is_empty());
        assert!(gc.old_objects_with_destructors.is_empty());
    }

    #[test]
    fn destructor_not_run_on_survival_then_run_on_oldgen_death() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock().unwrap();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_destructor(16, counting_destructor));

        let obj = gc.alloc_with_type(tid, 16);
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Survives the minor → promoted, NOT destructed; moves young→old list.
        gc.collect_nursery();
        assert_eq!(destructor_runs(), 0);
        assert!(!gc.is_in_nursery(root.0));
        assert!(gc.young_objects_with_destructors.is_empty());
        assert_eq!(gc.old_objects_with_destructors, vec![root.0]);

        // Unroot → dies at the next major collection.
        gc.roots.clear();
        gc.collect_full();
        assert_eq!(destructor_runs(), 1);
        assert!(gc.old_objects_with_destructors.is_empty());
    }

    #[test]
    fn destructor_survives_multiple_minors_no_double_count() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock().unwrap();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_destructor(16, counting_destructor));

        let obj = gc.alloc_with_type(tid, 16);
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Multiple minors while rooted: promoted once, never re-destructed,
        // and the old list does not accumulate duplicates.
        gc.collect_nursery();
        gc.collect_nursery();
        gc.collect_nursery();
        assert_eq!(destructor_runs(), 0);
        assert_eq!(gc.old_objects_with_destructors, vec![root.0]);

        gc.roots.clear();
        gc.collect_full();
        assert_eq!(destructor_runs(), 1);
    }

    #[test]
    fn destructor_on_direct_oldgen_alloc_runs_on_major_death() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock().unwrap();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        // large_object_threshold = nursery/2 = 512; a 1024-byte payload
        // bypasses the nursery and is registered straight onto the old list.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_destructor(1024, counting_destructor));

        let obj = gc.alloc_with_type(tid, 1024);
        assert!(!gc.is_in_nursery(obj.0));
        assert!(gc.young_objects_with_destructors.is_empty());
        assert_eq!(gc.old_objects_with_destructors, vec![obj.0]);

        // Unrooted → reclaimed by the major collection.
        gc.collect_full();
        assert_eq!(destructor_runs(), 1);
        assert!(gc.old_objects_with_destructors.is_empty());
    }

    #[test]
    fn no_destructor_means_no_list_entry() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock().unwrap();
        let mut gc = test_gc(4096);
        // A plain type (no destructor) is never recorded on either list.
        let tid = gc.register_type(TypeInfo::simple(16));
        gc.alloc_with_type(tid, 16);
        assert!(gc.young_objects_with_destructors.is_empty());
        assert!(gc.old_objects_with_destructors.is_empty());
    }

    #[test]
    fn test_write_barrier() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate a large object (goes to old gen).
        let old_obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        assert!(!gc.is_in_nursery(old_obj.0));

        // The old object should have TRACK_YOUNG_PTRS.
        let hdr = unsafe { header_of(old_obj.0) };
        assert!(unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) });

        // Write barrier clears the flag and adds to remembered set.
        gc.do_write_barrier(old_obj);
        assert!(unsafe { !(*hdr).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert_eq!(gc.remembered_set.len(), 1);

        // Second call: flag already cleared, should not add again.
        gc.do_write_barrier(old_obj);
        assert_eq!(gc.remembered_set.len(), 1);
    }

    #[test]
    fn prebuilt_write_barrier_registers_root_once_and_traces_children() {
        let mut gc = test_gc(1024);
        let child_tid = gc.register_type(TypeInfo::simple(16));
        let prebuilt_tid = gc.register_type(TypeInfo::with_gc_ptrs(
            2 * std::mem::size_of::<usize>(),
            vec![std::mem::size_of::<usize>()],
        ));
        let vtable = 0x1234_5678usize;
        crate::GcAllocator::register_vtable_for_type(&mut gc, vtable, prebuilt_tid);

        let child = gc.alloc_with_type(child_tid, 16);
        let prebuilt = crate::header::alloc_with_gc_header([vtable, child.0], prebuilt_tid);
        let prebuilt_ref = GcRef(prebuilt as usize);
        let hdr = unsafe { header_of(prebuilt_ref.0) };
        // `alloc_with_gc_header` is pyre's runtime host allocator, not a
        // translated prebuilt (see its docstring), so it leaves the flags
        // clear. Stamp incminimark's `init_gc_object_immortal` shape by hand
        // to exercise the lifecycle this test is about.
        unsafe {
            (*hdr).set_flag(flags::NO_HEAP_PTRS);
            (*hdr).set_flag(flags::TRACK_YOUNG_PTRS);
        }
        assert!(unsafe { (*hdr).has_flag(flags::NO_HEAP_PTRS) });

        gc.do_write_barrier(prebuilt_ref);
        assert!(unsafe { !(*hdr).has_flag(flags::NO_HEAP_PTRS) });
        assert_eq!(gc.remembered_set, vec![prebuilt_ref.0]);
        assert_eq!(gc.prebuilt_root_objects, vec![prebuilt_ref.0]);

        gc.do_collect_nursery();
        let promoted_child = unsafe { GcRef((*prebuilt)[1]) };
        assert_ne!(promoted_child, child);
        assert!(gc.oldgen.contains(promoted_child.0));

        gc.do_write_barrier(prebuilt_ref);
        assert_eq!(gc.prebuilt_root_objects, vec![prebuilt_ref.0]);
        gc.collect_full();
        assert!(gc.oldgen.contains(promoted_child.0));
        assert!(unsafe { !(*hdr).has_flag(flags::VISITED) });
    }

    // incminimark gates the write barrier solely on GCFLAG_TRACK_YOUNG_PTRS:
    // every old-gen producer sets it, and the barrier appends on the flag test
    // alone. These lock that invariant down across all four producers, plus a
    // zeroed-header case (no flag) that the barrier must skip.

    #[test]
    fn track_young_ptrs_set_on_direct_oldgen_alloc() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        assert!(unsafe { (*header_of(obj.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
    }

    #[test]
    fn track_young_ptrs_set_on_card_alloc() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::varsize(8, ptr_size, 0, true, Vec::new()));
        let length = 4usize;
        let total_size = GcHeader::SIZE + 8 + ptr_size * length;
        let obj = gc.alloc_in_oldgen_with_cards(tid, total_size, length, true);
        assert!(unsafe { (*header_of(obj.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
    }

    #[test]
    fn track_young_ptrs_set_on_nursery_promotion() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));
        let mut root = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe { gc.roots.add(&mut root) };
        gc.collect_nursery();
        // `root` now holds the promoted old-gen address.
        assert!(!gc.is_in_nursery(root.0));
        assert!(unsafe { (*header_of(root.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
        gc.roots.clear();
    }

    #[test]
    fn track_young_ptrs_set_on_shadow_alloc() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(0, 16);
        let shadow = gc.allocate_shadow(obj.0);
        assert!(unsafe { (*header_of(shadow)).has_flag(flags::TRACK_YOUNG_PTRS) });
    }

    #[test]
    fn born_old_typed_payload_is_zero_filled() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(24));
        let obj = gc.alloc_in_oldgen(tid, GcHeader::SIZE + 24);
        let payload = unsafe { std::slice::from_raw_parts(obj.0 as *const u8, 24) };
        assert!(payload.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn nonmoving_major_keeps_unoccupied_shadow_until_minor_copy() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(tid, 16);
        let shadow = gc.allocate_shadow(obj.0);
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };

        // The shadow payload has not been copied yet.  A non-moving major must
        // retain the reserved block without tracing its uninitialized fields.
        gc.do_collect_oldgen_nonmoving();
        assert_eq!(gc.oldgen.object_count(), 1);
        assert!(gc.oldgen.contains(shadow));
        assert!(!unsafe { (*header_of(shadow)).has_flag(flags::VISITED) });

        // The following minor occupies exactly that reserved identity home.
        gc.do_collect_nursery();
        assert_eq!(root.0, shadow);
        gc.roots.clear();
    }

    #[test]
    fn write_barrier_skips_object_without_track_young_ptrs() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));
        // A fabricated header with TRACK_YOUNG_PTRS clear is skipped. Nursery
        // objects have the same flag state but return through the range check.
        let mut buf = Box::new([0u64; 2]);
        let base = buf.as_mut_ptr() as usize;
        unsafe { *(base as *mut GcHeader) = GcHeader::new(0) };
        let payload = base + GcHeader::SIZE;
        gc.do_write_barrier(GcRef(payload));
        assert_eq!(gc.remembered_set.len(), 0);
        drop(buf);
    }

    #[test]
    fn write_barrier_null_is_noop() {
        // GcRef::NULL reaches the safe write_barrier entry; it must not read a
        // header at `0 - GcHeader::SIZE`.
        let mut gc = test_gc(1024);
        gc.do_write_barrier(GcRef(0));
        assert_eq!(gc.remembered_set.len(), 0);
    }

    #[test]
    fn write_barrier_ignores_unmanaged_jitframe_with_flag_byte_set() {
        // `_reload_frame_if_necessary` (aarch64/assembler.py:967-980,
        // x86/assembler.py:1369) re-applies the NON-array write-barrier fast
        // path to the current jitframe after a collecting helper call, so a
        // plain COND_CALL_GC_WB on the frame reaches the generic barrier at
        // runtime.
        //
        // A jitframe allocated off the GC is not in any managed generation,
        // so the word in front of it is not a header this collector owns.
        // `jitframe::alloc_off_gc_jitframe` keeps that word reserved and zeroed,
        // which is what stops the inline test from entering the helper at all;
        // this test covers the residue — the helper being entered anyway, with
        // the flag bit set, for a block the GC does not manage. Without the
        // `is_managed_heap_object` guard in `do_write_barrier`, that block
        // would enter `remembered_set` and the next minor would decode a type
        // id from those bytes.
        let mut gc = test_gc(1024);

        // Same shape the off-GC jitframe allocator produces: one reserved
        // leading word, then the fixed `JitFrame` words plus the trailing slot
        // array, zero-filled.
        let slots = 32usize;
        let frame_size = std::mem::size_of::<usize>() * (7 + 1 + slots);
        let layout =
            std::alloc::Layout::from_size_align(GcHeader::SIZE + frame_size, GcHeader::SIZE)
                .unwrap();
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null());
        let frame = unsafe { base.add(GcHeader::SIZE) };
        let obj = frame as usize;

        // The frame is a fresh host allocation, so it cannot overlap either
        // managed generation — that is precisely what the guard detects.
        assert!(!gc.nursery.contains(obj));
        assert!(!gc.oldgen.contains(obj));

        // Set the bit the inline COND_CALL_GC_WB test reads, so the helper is
        // entered exactly as it is when the bytes before a real malloc'd
        // frame happen to carry TRACK_YOUNG_PTRS.
        let descr = crate::WriteBarrierDescr::for_current_gc();
        let flag_byte = unsafe { frame.offset(descr.jit_wb_if_flag_byteofs as isize) };
        unsafe { *flag_byte |= descr.jit_wb_if_flag_singlebyte };
        let flag_byte_before = unsafe { *flag_byte };

        gc.do_write_barrier(GcRef(obj));

        assert_eq!(gc.remembered_set.len(), 0);
        // `remember_young_pointer` clears TRACK_YOUNG_PTRS, so an unchanged
        // byte also proves the barrier never wrote through the fake header.
        assert_eq!(unsafe { *flag_byte }, flag_byte_before);

        unsafe { std::alloc::dealloc(base, layout) };
    }

    #[test]
    fn write_barrier_ignores_headerless_pyobject_with_registered_vtable() {
        // `w_tuple_new` / `w_specialisedtuple_new_*` fall back to a bare
        // `Box::into_raw` when `try_gc_alloc_stable` finds no installed
        // collector, and the struct they build still starts with `ob_header` —
        // the very vtable `register_vtable_for_type` knows. Such an object is
        // in no managed generation, so `do_write_barrier` reaches its
        // bootstrap-prebuilt arm; a vtable-only test there would read the
        // allocator word in FRONT of the box as a `GcHeader` and, on a set
        // `TRACK_YOUNG_PTRS` bit, have `remember_young_pointer` write flags
        // back into that metadata. `registered_pyobject_header`'s tid equality
        // is what rejects it.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(2 * std::mem::size_of::<usize>()));
        let vtable = 0x1234_5678usize;
        crate::GcAllocator::register_vtable_for_type(&mut gc, vtable, tid);

        // One leading word standing in for the allocator metadata a real
        // `Box` keeps there, then the headerless object itself.
        let layout = std::alloc::Layout::from_size_align(
            GcHeader::SIZE + 2 * std::mem::size_of::<usize>(),
            GcHeader::SIZE,
        )
        .unwrap();
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null());
        let obj = unsafe { base.add(GcHeader::SIZE) } as usize;
        unsafe { (obj as *mut usize).write(vtable) };
        assert!(!gc.is_managed_heap_object(obj));

        // Make the fake header in front of the object look live: both the
        // barrier's flag bit and a type id that is registered — but NOT this
        // object's, which is the whole discriminator.
        let other_tid = gc.register_type(TypeInfo::simple(8));
        assert_ne!(other_tid, tid);
        let fake = base as *mut GcHeader;
        unsafe {
            fake.write(GcHeader::new(other_tid));
            (*fake).set_flag(flags::TRACK_YOUNG_PTRS);
        }
        let fake_bits_before = unsafe { (*fake).tid_and_flags };

        gc.do_write_barrier(GcRef(obj));

        assert_eq!(gc.remembered_set.len(), 0);
        // `remember_young_pointer` clears TRACK_YOUNG_PTRS, so unchanged bits
        // also prove the barrier never wrote through the fake header.
        assert_eq!(unsafe { (*fake).tid_and_flags }, fake_bits_before);

        unsafe { std::alloc::dealloc(base, layout) };
    }

    #[test]
    fn write_barrier_skips_forwarded_nursery_address() {
        // incminimark.py:1510-1512: nursery objects never carry
        // GCFLAG_TRACK_YOUNG_PTRS.  After a moving minor the old nursery
        // header is a forwarding word, not a live header whose flag bits may
        // be tested.  A host raw local can briefly retain that old address.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));
        let nursery_obj = gc.alloc_with_type(tid, 16);
        let stale_addr = nursery_obj;
        let mut root = nursery_obj;
        unsafe { gc.roots.add(&mut root) };

        gc.do_collect_nursery();
        assert_ne!(root, stale_addr);
        assert!(gc.is_in_nursery(stale_addr.0));
        assert!(!gc.is_in_nursery(root.0));

        gc.do_write_barrier(stale_addr);
        assert!(gc.remembered_set.is_empty());
        gc.roots.clear();
    }

    #[test]
    fn test_nursery_collection_with_pointers() {
        // Object layout: one GcRef field at offset 0 (payload = 8 bytes).
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create two objects: parent -> child.
        let child = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        let parent = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());

        // Write child's address into parent's first field.
        unsafe {
            *(parent.0 as *mut GcRef) = child;
        }

        // Root only the parent.
        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger collection.
        gc.collect_nursery();

        // Parent should have survived.
        assert!(!gc.is_in_nursery(root.0));
        assert!(!root.is_null());

        // The pointer field should now point to the child's new location.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_forwarding_dedup() {
        // Two roots pointing to the same nursery object should get the
        // same forwarded address.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        let shared = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        let mut root1 = shared;
        let mut root2 = shared;

        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }

        gc.collect_nursery();

        // Both roots should point to the same old-gen location.
        assert_eq!(root1, root2);
        assert!(!gc.is_in_nursery(root1.0));

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Allocate some objects and root one.
        let obj1 = gc.alloc_with_type(tid, 16);
        let _obj2 = gc.alloc_with_type(tid, 16); // unreachable

        let mut root = obj1;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Full collection: promotes to old gen and sweeps.
        gc.collect_full();

        assert!(!root.is_null());
        assert!(!gc.is_in_nursery(root.0));

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection_frees_unreachable_old_objects() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote two objects to old gen.
        let obj1 = gc.alloc_with_type(tid, 16);
        let obj2 = gc.alloc_with_type(tid, 16);
        let mut root1 = obj1;
        let mut root2 = obj2;
        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }
        gc.collect_nursery();
        assert!(!gc.is_in_nursery(root1.0));
        assert!(!gc.is_in_nursery(root2.0));
        assert_eq!(gc.oldgen.object_count(), 2);

        // Now unroot obj2 and do a full collection.
        gc.roots.remove(&mut root2);
        gc.collect_full();

        // Only obj1 should survive.
        assert_eq!(gc.oldgen.object_count(), 1);
        assert!(!root1.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_repeated_collections() {
        let mut gc = test_gc(512);
        let tid = gc.register_type(TypeInfo::simple(16));

        let mut root = GcRef::NULL;
        unsafe {
            gc.roots.add(&mut root);
        }

        for i in 0..50 {
            let obj = gc.alloc_with_type(tid, 16);
            // Write a marker value.
            unsafe {
                *(obj.0 as *mut u64) = i as u64;
            }
            root = obj;

            if i % 10 == 0 {
                gc.collect_nursery();
                // Root should survive and preserve its value.
                if !root.is_null() {
                    let val = unsafe { *(root.0 as *const u64) };
                    assert_eq!(val, i as u64);
                }
            }
        }

        gc.roots.clear();
    }

    #[test]
    fn test_gc_allocator_trait() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(32));

        let obj = gc.alloc_nursery(32);
        assert!(!obj.is_null());

        let varobj = gc.alloc_varsize(8, 4, 10);
        assert!(!varobj.is_null());

        gc.collect_nursery();
        gc.collect_full();
    }

    #[test]
    fn varsize_allocation_overflow_returns_null() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // incminimark.py external_malloc turns either multiplication or
        // addition overflow into MemoryError; the JIT allocator reports that
        // condition as NULL to its CHECK_MEMORY_ERROR caller.
        assert!(gc.alloc_varsize(0, 2, usize::MAX).is_null());
        assert!(gc.alloc_varsize_typed(tid, usize::MAX, 1, 1).is_null());
        assert!(gc.alloc_varsize_no_collect(0, usize::MAX, 2).is_null());

        // Include the GC header in the same checked-size contract.
        assert!(gc.alloc_nursery(usize::MAX).is_null());
        assert!(gc.alloc_nursery_no_collect(usize::MAX).is_null());
        assert!(gc.alloc_oldgen_typed(tid, usize::MAX).is_null());
    }

    /// A varsize length is read out of the object, so a collector reaching an
    /// object before its length is initialized computes a size that describes
    /// nothing.  Both shapes must be rejected — in particular the second, where
    /// the multiplication does *not* overflow and the size is merely far past
    /// anything `Layout` can express.
    #[test]
    fn varsize_length_that_describes_no_allocation_is_rejected() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(16, 8, 0, false, Vec::new()));

        // The length field lives at offset 0, so a local stands in for the
        // object: nothing here allocates, and nothing is dereferenced beyond it.
        let length = std::cell::Cell::new(0usize);
        let obj_addr = length.as_ptr() as usize;

        length.set(4);
        assert_eq!(gc.try_size_for_typeid(obj_addr, tid), Some(16 + 8 * 4));

        length.set(usize::MAX);
        assert_eq!(gc.try_size_for_typeid(obj_addr, tid), None);

        length.set(isize::MAX as usize / 8);
        assert_eq!(gc.try_size_for_typeid(obj_addr, tid), None);
    }

    #[test]
    #[should_panic(expected = "varsize length describes no allocation")]
    fn varsize_length_that_describes_no_allocation_names_its_inputs() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(16, 8, 0, false, Vec::new()));
        let length = std::cell::Cell::new(usize::MAX);
        gc.size_for_typeid(length.as_ptr() as usize, tid, "test");
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// pyre's GC stores an explicit vtable→type_id table; verify that
    /// register_vtable_for_type / get_typeid_from_classptr_if_gcremovetypeptr
    /// round-trip via the GcAllocator trait.
    #[test]
    fn test_gcremovetypeptr_vtable_lookup() {
        let mut gc = test_gc(4096);
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let float_tid = gc.register_type(TypeInfo::simple(16));

        // Register two distinct vtables
        let int_vtable: usize = 0x1234_5670;
        let float_vtable: usize = 0x1234_5680;
        crate::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);
        crate::GcAllocator::register_vtable_for_type(&mut gc, float_vtable, float_tid);

        // Round-trip
        assert_eq!(
            crate::GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, int_vtable),
            Some(int_tid)
        );
        assert_eq!(
            crate::GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, float_vtable),
            Some(float_tid)
        );
        // Unknown classptr → None
        assert_eq!(
            crate::GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, 0xCAFEBABE),
            None
        );
    }

    #[test]
    fn test_alloc_nursery_no_collect_does_not_trigger_collection() {
        let mut gc = test_gc(64);
        gc.register_type(TypeInfo::simple(24));

        let obj1 = gc.alloc_nursery_no_collect(24);
        let obj2 = gc.alloc_nursery_no_collect(24);

        assert!(!obj1.is_null());
        assert!(!obj2.is_null());
        assert_eq!(gc.minor_collections, 0);
        // The second allocation may have fallen back to old gen, but it must
        // still succeed without forcing a collection.
        assert_ne!(obj1, obj2);
    }

    #[test]
    fn test_alloc_varsize_no_collect_does_not_trigger_collection() {
        let mut gc = test_gc(64);
        gc.register_type(TypeInfo::simple(32));

        let obj = gc.alloc_varsize_no_collect(16, 8, 8);

        assert!(!obj.is_null());
        assert_eq!(gc.minor_collections, 0);
        // This request is too large for the tiny nursery, so no-collect mode
        // must have used the old generation instead.
        assert!(!gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_write_barrier_with_collection() {
        // Scenario: old object points to young object, write barrier ensures
        // the young object survives collection.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create an old-gen object.
        let old_obj = gc.alloc_in_oldgen(tid, GcHeader::SIZE + std::mem::size_of::<GcRef>());

        // Create a young object.
        let young_obj = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(young_obj.0 as *mut u64) = 0x42424242;
        }

        // Store young ref into old object's field.
        unsafe {
            *(old_obj.0 as *mut GcRef) = young_obj;
        }
        // Write barrier.
        gc.do_write_barrier(old_obj);

        // Root only the old object.
        let mut root = old_obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Collect.
        gc.collect_nursery();

        // The old object's field should be updated to the new location.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());
        let val = unsafe { *(child_ref.0 as *const u64) };
        assert_eq!(val, 0x42424242);

        gc.roots.clear();
    }

    #[test]
    fn test_chain_of_pointers() {
        // Test a chain: root -> A -> B -> C, all in nursery.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        let c = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(c.0 as *mut GcRef) = GcRef::NULL;
        }

        let b = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(b.0 as *mut GcRef) = c;
        }

        let a = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(a.0 as *mut GcRef) = b;
        }

        let mut root = a;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // Verify the entire chain survived.
        assert!(!gc.is_in_nursery(root.0));
        let new_b = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(new_b.0));
        assert!(!new_b.is_null());
        let new_c = unsafe { *(new_b.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(new_c.0));
        assert!(!new_c.is_null());
        let tail = unsafe { *(new_c.0 as *const GcRef) };
        assert!(tail.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection_with_graph() {
        // Test major collection with a graph: root -> A -> B, root -> C (unreachable D).
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Type with one pointer field.
        let tid1 = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        // Type with two pointer fields.
        let tid2 = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size * 2, vec![0, ptr_size]));

        let b = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(b.0 as *mut GcRef) = GcRef::NULL;
        }

        let a = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(a.0 as *mut GcRef) = b;
        }

        let c = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(c.0 as *mut GcRef) = GcRef::NULL;
        }

        let d = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(d.0 as *mut GcRef) = GcRef::NULL;
        }
        let _ = d; // unreachable

        // Root object points to both A and C.
        let root_obj = gc.alloc_with_type(tid2, ptr_size * 2);
        unsafe {
            *(root_obj.0 as *mut GcRef) = a;
            *((root_obj.0 + ptr_size) as *mut GcRef) = c;
        }

        let mut root = root_obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Promote to old gen.
        gc.collect_nursery();

        // Now do full collection. A, B, C should survive; D should be freed.
        // We have 5 objects in old gen (root, A, B, C, D).
        // Wait, D was unreachable from root, so it wouldn't have been promoted.
        // Actually it was in the nursery unreachable, so it was wiped on nursery reset.
        assert_eq!(gc.oldgen.object_count(), 4); // root, A, B, C

        gc.collect_full();

        // All 4 are reachable from the root, so all survive.
        assert_eq!(gc.oldgen.object_count(), 4);

        gc.roots.clear();
    }

    #[test]
    fn major_mark_skips_unmanaged_field_target() {
        // An old-gen object's `gc_ptr_offsets` field may point at a
        // `std::alloc`-backed (non-GC) block while some object fields are
        // still host allocations rather than managed-GC allocations.
        // The major-mark walker must skip such fields rather than
        // dereferencing memory before the unmanaged block as a
        // `GcHeader`. This test catches regression of the
        // `is_managed_heap_object` guard in `mark_object`.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Allocate parent in old-gen. Field pointer will be set to a
        // raw `Box::into_raw` block — what `std::alloc` does for an
        // unmigrated `wrappeditems` / `items` slot today.
        let parent = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
        let unmanaged = Box::into_raw(Box::new(0xCAFEBABEu64));
        unsafe {
            *(parent.0 as *mut GcRef) = GcRef(unmanaged as usize);
        }

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Major collection — without the guard this would crash on
        // `header_of(unmanaged)` reading invalid memory.
        gc.collect_full();

        // Parent survived because it is rooted; the unmanaged target
        // was correctly skipped.
        assert_eq!(gc.oldgen.object_count(), 1);

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(unmanaged));
        }
    }

    #[test]
    fn major_mark_skips_unmanaged_varsize_item_target() {
        // Same scenario, varsize variant: an `items_have_gc_ptrs = true`
        // array can hold unmanaged pointers while item storage is still
        // host-allocated. The mark
        // walker's variable-part loop must also guard with
        // `is_managed_heap_object`.
        let ptr_size = std::mem::size_of::<GcRef>();
        let length_offset = 0;
        let base_size = 8; // length field (Signed)
        let length = 2usize;

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(
            base_size,
            ptr_size,
            length_offset,
            true,
            Vec::new(),
        ));

        let total_payload = base_size + ptr_size * length;
        let parent = gc.alloc_in_oldgen(tid, GcHeader::SIZE + total_payload);
        unsafe {
            *(parent.0 as *mut usize) = length;
        }
        let unmanaged_a = Box::into_raw(Box::new(0xDEADBEEFu64));
        let unmanaged_b = Box::into_raw(Box::new(0xFEEDFACEu64));
        unsafe {
            *((parent.0 + base_size) as *mut GcRef) = GcRef(unmanaged_a as usize);
            *((parent.0 + base_size + ptr_size) as *mut GcRef) = GcRef(unmanaged_b as usize);
        }

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_full();
        assert_eq!(gc.oldgen.object_count(), 1);

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(unmanaged_a));
            drop(Box::from_raw(unmanaged_b));
        }
    }

    #[test]
    fn test_data_integrity_across_collections() {
        // Allocate objects with distinctive data, collect, verify data.
        let mut gc = test_gc(2048);
        // Type: 32 bytes payload, one GcRef at offset 0, then 24 bytes of data.
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(32, vec![0]));

        let child = gc.alloc_with_type(tid, 32);
        unsafe {
            *(child.0 as *mut GcRef) = GcRef::NULL;
            *((child.0 + 8) as *mut u64) = 0xAAAA_BBBB_CCCC_DDDD;
            *((child.0 + 16) as *mut u64) = 0x1111_2222_3333_4444;
            *((child.0 + 24) as *mut u64) = 0x5555_6666_7777_8888;
        }

        let parent = gc.alloc_with_type(tid, 32);
        unsafe {
            *(parent.0 as *mut GcRef) = child;
            *((parent.0 + 8) as *mut u64) = 0xCAFE_BABE_DEAD_BEEF;
        }

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // Verify parent data.
        let pdata = unsafe { *((root.0 + 8) as *const u64) };
        assert_eq!(pdata, 0xCAFE_BABE_DEAD_BEEF);

        // Verify child data.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!child_ref.is_null());
        let c1 = unsafe { *((child_ref.0 + 8) as *const u64) };
        let c2 = unsafe { *((child_ref.0 + 16) as *const u64) };
        let c3 = unsafe { *((child_ref.0 + 24) as *const u64) };
        assert_eq!(c1, 0xAAAA_BBBB_CCCC_DDDD);
        assert_eq!(c2, 0x1111_2222_3333_4444);
        assert_eq!(c3, 0x5555_6666_7777_8888);

        gc.roots.clear();
    }

    #[test]
    fn marking_minor_root_greys_white_old_object() {
        // incminimark.py:2128-2143 `_trace_drag_out1_marking_phase` checks
        // the root after `_trace_drag_out` even when it was already old.
        // This is how a stack root exposed after the major's initial root
        // snapshot re-enters the marking worklist at the next minor.
        let mut gc = test_gc(2048);
        let tid = gc.register_type(TypeInfo::simple(8));
        let old = gc.alloc_in_oldgen(tid, GcHeader::SIZE + 8);
        let hdr = unsafe { header_of(old.0) };
        assert!(!unsafe { (*hdr).has_flag(flags::VISITED) });

        gc.gc_state = GcState::Marking;
        let mut root = old;
        gc.drag_out_root(&mut root);

        assert_eq!(root, old);
        assert!(unsafe { (*hdr).has_flag(flags::VISITED) });
        assert!(gc.incr_state.gray_stack.is_empty());
        assert_eq!(gc.incr_state.more_gray_stack.pop(), Some(old.0));
    }

    // ── Card marking tests ──
    // incminimark.py card bytes layout: [card_bytes...][GcHeader][payload]

    /// Helper: register a varsize array type and allocate in old gen with
    /// inline card bytes. Returns (obj, array_length).
    fn alloc_card_array(gc: &mut MiniMarkGC, array_length: usize) -> (GcRef, usize) {
        let ptr_size = std::mem::size_of::<GcRef>();
        let arr_tid = gc.register_type(TypeInfo::varsize(8, ptr_size, 0, true, Vec::new()));
        let total_size = GcHeader::SIZE + 8 + ptr_size * array_length;
        let obj = gc.alloc_in_oldgen_with_cards(arr_tid, total_size, array_length, true);
        // Write the length field.
        unsafe {
            *(obj.0 as *mut usize) = array_length;
        }
        (obj, array_length)
    }

    #[test]
    fn test_card_marking_basic() {
        let mut gc = test_gc(4096);
        let (obj, _) = alloc_card_array(&mut gc, 512);
        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(flags::HAS_CARDS) });

        // Card-marking write barrier: mark card for index 5.
        gc.do_write_barrier_card(obj, 5, DEFAULT_CARD_PAGE_SHIFT);

        // The card at index 5 >> 7 = 0 should be dirty.
        assert!(
            gc.is_card_dirty(obj, 0),
            "card 0 should be dirty after writing index 5"
        );
        assert!(
            unsafe { (*hdr).has_flag(flags::CARDS_SET) },
            "CARDS_SET flag should be set"
        );

        // Mark another index in a different card.
        gc.do_write_barrier_card(obj, 200, DEFAULT_CARD_PAGE_SHIFT);
        let card_idx = 200 >> DEFAULT_CARD_PAGE_SHIFT;
        assert!(
            gc.is_card_dirty(obj, card_idx as usize),
            "card for index 200 should be dirty"
        );
    }

    #[test]
    fn test_card_marking_clear_after_collection() {
        let mut gc = test_gc(4096);
        let (obj, array_length) = alloc_card_array(&mut gc, 512);
        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(flags::HAS_CARDS) });

        // external_malloc is not zero-filled.  Initialize every item that
        // dirty-card tracing is allowed to read, as production allocation
        // rewrites do before publishing the array.
        let ptr_size = std::mem::size_of::<GcRef>();
        let items_start = obj.0 + 8;
        for i in 0..array_length {
            unsafe {
                *((items_start + i * ptr_size) as *mut GcRef) = GcRef::NULL;
            }
        }

        // Mark some cards.
        gc.do_write_barrier_card(obj, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 200, DEFAULT_CARD_PAGE_SHIFT);
        assert!(
            unsafe { (*hdr).has_flag(flags::CARDS_SET) },
            "CARDS_SET should be set before collection"
        );

        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Minor collection clears card bytes.
        gc.do_collect_nursery();

        let hdr = unsafe { header_of(root.0) };
        assert!(
            unsafe { !(*hdr).has_flag(flags::CARDS_SET) },
            "CARDS_SET flag should be cleared after collection"
        );
        assert!(
            !gc.is_card_dirty(root, 0),
            "card 0 should be cleared after collection"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_card_marking_dirty_cards_list() {
        let mut gc = test_gc(4096);
        let (obj, _) = alloc_card_array(&mut gc, 512);

        // Mark cards for indices in different card pages.
        gc.do_write_barrier_card(obj, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 128, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 256, DEFAULT_CARD_PAGE_SHIFT);

        let dirty = gc.dirty_cards(obj);
        assert_eq!(dirty, vec![0, 1, 2], "should have cards 0, 1, 2 dirty");
    }

    #[test]
    fn test_card_marking_fallback_without_has_cards() {
        // Object without HAS_CARDS should fall back to remembered set.
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        // Do NOT set HAS_CARDS — no card header space allocated.

        gc.do_write_barrier_card(obj, 5, DEFAULT_CARD_PAGE_SHIFT);

        // Should fall back to full remembered set.
        assert_eq!(gc.remembered_set.len(), 1, "should add to remembered set");
        assert!(
            gc.old_objects_with_cards_set.is_empty(),
            "should not add to old_objects_with_cards_set without HAS_CARDS"
        );
    }

    #[test]
    fn test_card_clear_individual() {
        let mut gc = test_gc(4096);
        let (obj1, _) = alloc_card_array(&mut gc, 512);
        let (obj2, _) = alloc_card_array(&mut gc, 512);

        gc.do_write_barrier_card(obj1, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj2, 0, DEFAULT_CARD_PAGE_SHIFT);

        // Clear only obj1's cards.
        gc.clear_cards(obj1.0);

        assert!(!gc.is_card_dirty(obj1, 0), "obj1 cards should be cleared");
        assert!(
            gc.is_card_dirty(obj2, 0),
            "obj2 cards should still be dirty"
        );
    }

    // ── SafepointMap tests ──

    #[test]
    fn test_safepoint_map_register_and_lookup() {
        let mut smap = SafepointMap::new();

        let mut gc_map_0 = crate::GcMap::new();
        gc_map_0.set_ref(0);
        gc_map_0.set_ref(3);

        let mut gc_map_1 = crate::GcMap::new();
        gc_map_1.set_ref(1);
        gc_map_1.set_ref(7);

        smap.add(100, gc_map_0);
        smap.add(200, gc_map_1);

        // Lookup existing entries.
        let found_0 = smap.lookup(100).unwrap();
        assert!(found_0.is_ref(0));
        assert!(found_0.is_ref(3));
        assert!(!found_0.is_ref(1));

        let found_1 = smap.lookup(200).unwrap();
        assert!(found_1.is_ref(1));
        assert!(found_1.is_ref(7));
        assert!(!found_1.is_ref(0));

        // Lookup non-existent offset returns None.
        assert!(smap.lookup(999).is_none());
    }

    #[test]
    fn test_safepoint_map_empty() {
        let smap = SafepointMap::new();
        assert!(smap.lookup(0).is_none());
        assert!(smap.entries.is_empty());
    }

    // ── CompiledCodeRegistry tests ──

    #[test]
    fn test_compiled_code_registry_register_and_find() {
        let mut registry = CompiledCodeRegistry::new();
        assert!(registry.is_empty());

        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(16, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 42,
        });

        assert_eq!(registry.len(), 1);

        // Address inside the region.
        let (region, offset) = registry.find_region(0x1010).unwrap();
        assert_eq!(region.loop_token, 42);
        assert_eq!(offset, 0x10);

        // Address at the start.
        let (region, offset) = registry.find_region(0x1000).unwrap();
        assert_eq!(region.loop_token, 42);
        assert_eq!(offset, 0);

        // Address outside the region.
        assert!(registry.find_region(0x900).is_none());
        assert!(registry.find_region(0x1100).is_none());
    }

    #[test]
    fn test_compiled_code_registry_multiple_regions() {
        let mut registry = CompiledCodeRegistry::new();

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 4,
            loop_token: 1,
        });
        registry.register(CompiledCodeRegion {
            code_start: 0x3000,
            code_size: 0x200,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 8,
            loop_token: 2,
        });
        registry.register(CompiledCodeRegion {
            code_start: 0x2000,
            code_size: 0x80,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 2,
            loop_token: 3,
        });

        assert_eq!(registry.len(), 3);

        // Each region should be findable.
        assert_eq!(registry.find_region(0x1050).unwrap().0.loop_token, 1);
        assert_eq!(registry.find_region(0x2040).unwrap().0.loop_token, 3);
        assert_eq!(registry.find_region(0x3100).unwrap().0.loop_token, 2);

        // Gap between regions returns None.
        assert!(registry.find_region(0x1200).is_none());
    }

    #[test]
    fn test_compiled_code_registry_unregister() {
        let mut registry = CompiledCodeRegistry::new();

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 4,
            loop_token: 10,
        });
        registry.register(CompiledCodeRegion {
            code_start: 0x2000,
            code_size: 0x100,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 4,
            loop_token: 20,
        });

        assert_eq!(registry.len(), 2);

        registry.unregister(10);
        assert_eq!(registry.len(), 1);
        assert!(registry.find_region(0x1050).is_none());
        assert_eq!(registry.find_region(0x2050).unwrap().0.loop_token, 20);
    }

    #[test]
    fn test_compiled_code_registry_safepoint_lookup_for_root_scanning() {
        let mut registry = CompiledCodeRegistry::new();

        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(0x20, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0x5000,
            code_size: 0x200,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 99,
        });

        // Simulate finding a return address and looking up the safepoint map.
        let return_addr = 0x5020;
        let (region, offset) = registry.find_region(return_addr).unwrap();
        let gc_map = region.safepoint_map.lookup(offset).unwrap();

        // Verify the GC map identifies the correct slots.
        assert!(gc_map.is_ref(0), "slot 0 should be a GC ref");
        assert!(!gc_map.is_ref(1), "slot 1 should not be a GC ref");
        assert!(gc_map.is_ref(2), "slot 2 should be a GC ref");
        assert!(!gc_map.is_ref(3), "slot 3 should not be a GC ref");
    }

    #[test]
    fn test_scan_frame_enumerates_gc_ref_slots() {
        let mut registry = CompiledCodeRegistry::new();

        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(0x10, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0xA000,
            code_size: 0x100,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 77,
        });

        // Allocate a fake frame on the stack.
        let frame: [usize; 4] = [111, 222, 333, 444];
        let frame_base = frame.as_ptr();

        let return_addr = 0xA010;
        let roots = unsafe { registry.scan_frame(return_addr, frame_base) };

        // Should find slots 0 and 2.
        assert_eq!(roots.len(), 2);
        unsafe {
            assert_eq!(*(roots[0] as *const usize), 111);
            assert_eq!(*(roots[1] as *const usize), 333);
        }
    }

    // ── Incremental marking tests ──

    #[test]
    fn test_incremental_marking_basic() {
        // Start an incremental cycle, run steps, and verify completion.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote some objects to old gen via minor collection.
        let obj1 = gc.alloc_with_type(tid, 16);
        let obj2 = gc.alloc_with_type(tid, 16);
        let mut root1 = obj1;
        let mut root2 = obj2;
        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root1.0));
        assert!(!gc.is_in_nursery(root2.0));

        // Manually start an incremental cycle.
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Run marking steps until complete.
        let mut steps = 0;
        while gc.is_incremental_marking() {
            gc.incremental_mark_step();
            steps += 1;
            if steps > 100 {
                panic!("incremental marking did not complete");
            }
        }

        // Marking should have processed the 2 root objects.
        assert!(gc.incremental_objects_marked() >= 2);

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_marking_piggyback() {
        // Verify that incremental marking progresses during nursery collections.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote several objects to old gen by rooting each and collecting.
        let mut roots_storage = vec![GcRef::NULL; 10];
        for r in roots_storage.iter_mut() {
            unsafe {
                gc.roots.add(r);
            }
        }
        for r in roots_storage.iter_mut() {
            let obj = gc.alloc_with_type(tid, 16);
            *r = obj;
        }
        gc.do_collect_nursery();
        for r in &roots_storage {
            assert!(!gc.is_in_nursery(r.0));
        }

        // Start an incremental cycle with a tiny budget so it takes
        // multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Each nursery collection should advance the marking.
        let marked_before = gc.incremental_objects_marked();
        gc.do_collect_nursery();
        // After one nursery collection with budget=1, we should have
        // marked at least one more object (if any remained).
        assert!(
            gc.incremental_objects_marked() > marked_before,
            "marking should advance during nursery collection"
        );

        gc.roots.clear();
    }

    #[test]
    fn minor_forwards_nursery_references_held_by_gray_objects() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let holder_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::simple(16));

        let holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe { *(holder.0 as *mut GcRef) = GcRef::NULL };
        let mut root = holder;
        unsafe { gc.roots.add(&mut root) };
        gc.do_collect_nursery();

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert_eq!(gc.incr_state.gray_stack, vec![root.0]);

        let young = gc.alloc_with_type(leaf_tid, 16);
        unsafe {
            *(young.0 as *mut u64) = 0xC011_EC70;
            // Model a collector-owned JITFRAME gcmap spill: it does not pass
            // through the ordinary mutator write barrier.
            *(root.0 as *mut GcRef) = young;
        }
        assert!(gc.remembered_set.contains(&root.0));

        gc.disable();
        gc.do_collect_nursery();
        gc.enable();

        let forwarded = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(forwarded.0));
        assert_eq!(unsafe { *(forwarded.0 as *const u64) }, 0xC011_EC70);
        gc.do_collect_full();
        gc.roots.clear();
    }

    #[test]
    fn test_minor_collection_can_take_multiple_major_steps_when_promotions_outpace_credit() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Build an old-gen chain large enough that a single budget=1 marking
        // step cannot finish it.
        let mut prev = GcRef::NULL;
        for _ in 0..6 {
            let obj = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
        }
        let mut root = prev;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(!gc.incr_state.gray_stack.is_empty());

        // Simulate a minor collection that promoted more than one step's worth
        // of objects so incminimark-style accounting demands extra progress.
        gc.bytes_made_old_since_cycle = gc.config.nursery_size;
        gc.threshold_bytes_made_old = 0;
        gc.run_major_progress_after_minor();

        assert!(
            gc.incremental_objects_marked() >= 2
                || gc.threshold_bytes_made_old > gc.config.nursery_size / 2,
            "major progress should take multiple steps when promoted bytes outpace credit"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_marking_budget() {
        // Each step should process at most `mark_budget_per_step` objects.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create a chain of 10 objects so marking has plenty of work.
        let mut prev = GcRef::NULL;
        let mut roots = Vec::new();
        for _ in 0..10 {
            let obj = gc.alloc_with_type(tid, ptr_size);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
            roots.push(prev);
        }

        // Root the head of the chain.
        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote all to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));

        // Start incremental cycle with a tiny byte budget so each step still
        // processes exactly one object.  Clear the previous minor's survivor
        // sample: incminimark normally raises the step budget to twice that
        // sample, which is covered independently below.
        gc.set_mark_budget(2);
        gc.nursery_surviving_size = 0;
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // First step: marks at most 1 object.
        let done = gc.incremental_mark_step();
        assert!(!done, "should not be done after marking only 1 out of 10");
        assert_eq!(gc.incremental_objects_marked(), 1);

        // Second step: marks 1 more.
        let done = gc.incremental_mark_step();
        assert!(!done, "should not be done after 2 total");
        assert_eq!(gc.incremental_objects_marked(), 2);

        gc.roots.clear();
    }

    #[test]
    fn incremental_marking_uses_twice_the_latest_minor_survivors_as_its_budget() {
        // incminimark.py:2453-2457: even an explicitly tiny increment must
        // keep up with a high-survival minor collection.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        let mut head = GcRef::NULL;
        for _ in 0..4 {
            let obj = gc.alloc_with_type(tid, ptr_size);
            unsafe { *(obj.0 as *mut GcRef) = head };
            head = obj;
        }
        unsafe { gc.roots.add(&mut head) };
        gc.do_collect_nursery();
        assert!(gc.nursery_surviving_size > 0);

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        gc.incremental_mark_step();

        assert_eq!(gc.incremental_objects_marked(), 4);
        gc.roots.clear();
    }

    #[test]
    fn incremental_marking_drains_the_separate_mutation_worklist() {
        // incminimark.py:2458-2470: an empty primary list means the ordinary
        // walk consumed less than half the step, so more_objects_to_trace is
        // swapped in and drained completely.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let modified = gc.alloc_in_oldgen(tid, GcHeader::SIZE + 16);
        unsafe { (*header_of(modified.0)).set_flag(flags::VISITED) };

        gc.gc_state = GcState::Marking;
        gc.incr_state.gray_stack.clear();
        gc.incr_state.more_gray_stack.push(modified.0);
        gc.nursery_surviving_size = 0;

        assert!(gc.incremental_mark_step());
        assert_eq!(gc.incremental_objects_marked(), 1);
        assert!(gc.incr_state.gray_stack.is_empty());
        assert!(gc.incr_state.more_gray_stack.is_empty());
    }

    #[test]
    fn test_incremental_marking_completes() {
        // A full incremental cycle (start -> repeated steps -> sweep)
        // produces the same result as a stop-the-world full collection:
        // unreachable old-gen objects are freed.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create reachable and unreachable old-gen objects.
        let reachable = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(reachable.0 as *mut GcRef) = GcRef::NULL;
        }
        let unreachable = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(unreachable.0 as *mut GcRef) = GcRef::NULL;
        }

        let mut root = reachable;
        let mut root2 = unreachable;
        unsafe {
            gc.roots.add(&mut root);
            gc.roots.add(&mut root2);
        }

        // Promote both to old gen.
        gc.do_collect_nursery();
        assert_eq!(gc.oldgen.object_count(), 2);

        // Unroot the unreachable one.
        gc.roots.remove(&mut root2);

        // Run incremental cycle with budget=1 to force multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();

        // Drive the cycle to completion.
        let mut iterations = 0;
        while gc.is_incremental_marking() {
            gc.incremental_mark_step();
            iterations += 1;
            if iterations > 100 {
                panic!("incremental marking did not complete");
            }
        }

        // Finish: sweep unreachable objects.
        gc.gc_step_until_scanning();

        // Only the reachable object should survive.
        assert_eq!(gc.oldgen.object_count(), 1);
        assert!(!root.is_null());

        gc.roots.clear();
    }

    #[test]
    fn incremental_sweep_spans_steps_and_preserves_interleaved_promotion() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let total_size = GcHeader::SIZE + 16;

        // More than two arena pages of white old objects ensure that the
        // one-page budget used by this tiny test nursery cannot finish sweep.
        for _ in 0..900 {
            gc.alloc_in_oldgen(tid, total_size);
        }

        gc.start_incremental_cycle();
        assert!(gc.incremental_mark_step());
        assert_eq!(gc.gc_state, GcState::Sweeping);
        gc.major_collection_step();
        assert_eq!(gc.gc_state, GcState::Sweeping);

        // Disable automatic major progress only while this minor promotes its
        // survivor. The promotion allocates on ArenaCollection's fresh active
        // page lists created by mass_free_prepare, not the frozen old_* lists.
        let young = gc.alloc_with_type(tid, 16);
        unsafe { *(young.0 as *mut u64) = 0x0517_EA5E };
        let mut root = young;
        unsafe { gc.roots.add(&mut root) };
        gc.disable();
        gc.do_collect_nursery();
        gc.enable();
        assert!(!gc.is_in_nursery(root.0));

        gc.gc_step_until_scanning();
        assert_eq!(gc.gc_state, GcState::Scanning);
        assert_eq!(gc.major_collections, 1);
        assert_eq!(gc.oldgen.object_count(), 1);
        assert_eq!(unsafe { *(root.0 as *const u64) }, 0x0517_EA5E);
        gc.roots.clear();
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "raw_malloc_might_sweep must be empty outside SWEEPING")]
    fn rawmalloc_sweep_candidates_require_sweeping_state() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let raw_size = gc.oldgen.small_request_threshold() + std::mem::size_of::<usize>();
        gc.alloc_in_oldgen(tid, raw_size);
        gc.oldgen.sweep_prepare();
        assert!(gc.oldgen.rawmalloc_sweep_pending());
        assert_eq!(gc.gc_state, GcState::Scanning);
        gc.debug_check_consistency();
    }

    #[test]
    fn stop_the_world_full_collection_drains_marking_and_sweeping() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let live = gc.alloc_in_oldgen(tid, GcHeader::SIZE + 16);
        let mut root = live;
        unsafe { gc.roots.add(&mut root) };

        let raw_size = gc.oldgen.small_request_threshold() + std::mem::size_of::<usize>();
        for _ in 0..20 {
            gc.alloc_in_oldgen(tid, raw_size);
        }
        let majors_before = gc.major_collections;

        gc.do_collect_full();

        assert_eq!(gc.gc_state, GcState::Scanning);
        assert!(!gc.oldgen.rawmalloc_sweep_pending());
        assert_eq!(gc.major_collections, majors_before + 1);
        assert_eq!(gc.oldgen.object_count(), 1);
        assert_eq!(root.0, live.0);
        gc.roots.clear();
    }

    // ── GC stress tests ──

    #[test]
    #[cfg(debug_assertions)]
    fn test_gc_stress_with_safepoint_scanning() {
        // Register a compiled code region with a safepoint map, then
        // allocate objects under pressure so nursery collections fire.
        // After collection, verify that roots discovered via scan_frame
        // point to valid, promoted objects.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(512); // small nursery to force frequent collections
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size * 2, vec![0, ptr_size]));

        // Build a compiled code registry with a safepoint map marking
        // frame slots 0 and 2 as GC references.
        let mut registry = CompiledCodeRegistry::new();
        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(0x50, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 1,
        });

        // Simulate a JIT frame: slots 0 and 2 hold GcRefs, slots 1 and 3
        // hold non-pointer data.
        let obj_a = gc.alloc_with_type(tid, ptr_size * 2);
        let obj_b = gc.alloc_with_type(tid, ptr_size * 2);
        unsafe {
            *(obj_a.0 as *mut GcRef) = GcRef::NULL;
            *((obj_a.0 + ptr_size) as *mut GcRef) = GcRef::NULL;
            *(obj_b.0 as *mut GcRef) = GcRef::NULL;
            *((obj_b.0 + ptr_size) as *mut GcRef) = GcRef::NULL;
        }

        let frame: [usize; 4] = [obj_a.0, 0xDEAD, obj_b.0, 0xBEEF];

        // Register frame slots as GC roots (simulating what the backend does
        // at a safepoint).
        let roots_from_frame = unsafe { registry.scan_frame(0x1050, frame.as_ptr()) };
        assert_eq!(roots_from_frame.len(), 2);

        // Register the scanned slots as roots with the GC.
        for root_ptr in &roots_from_frame {
            unsafe {
                gc.roots.add(*root_ptr);
            }
        }

        // Allocate many objects to force multiple nursery collections.
        for i in 0..200 {
            let filler = gc.alloc_with_type(tid, ptr_size * 2);
            unsafe {
                *(filler.0 as *mut u64) = i as u64;
            }
        }
        assert!(
            gc.minor_collections > 0,
            "should have triggered nursery collections"
        );

        // Read back the GcRefs from the frame slots (the GC may have updated
        // them when it promoted the objects).
        let ref_a = GcRef(frame[0]);
        let ref_b = GcRef(frame[2]);

        // The original nursery objects should have been forwarded.
        // The frame slots must now point to valid (non-nursery) addresses.
        assert!(!ref_a.is_null());
        assert!(!ref_b.is_null());
        assert!(
            !gc.is_in_nursery(ref_a.0),
            "object A should have been promoted out of nursery"
        );
        assert!(
            !gc.is_in_nursery(ref_b.0),
            "object B should have been promoted out of nursery"
        );

        // Verify non-GC slots are untouched.
        assert_eq!(frame[1], 0xDEAD);
        assert_eq!(frame[3], 0xBEEF);

        gc.roots.clear();
    }

    /// With `set_stress_collect(true)`, `alloc_with_type` forces a full
    /// collection on every allocation. A large nursery is used so that no
    /// collection would fire naturally across these allocations — the
    /// collection counter therefore isolates the flag's effect. A rooted
    /// nursery object must be forwarded out of the nursery and stay valid.
    #[test]
    #[cfg(feature = "gc_stress")]
    fn test_gc_stress_forces_collection_per_alloc() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1 << 20); // 1 MiB nursery: no natural collection here.
        gc.set_stress_collect(true);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        let obj = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj.0 as *mut GcRef) = GcRef::NULL;
        }
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root as *mut GcRef);
        }

        let before = gc.minor_collections;
        for _ in 0..5 {
            let _ = gc.alloc_with_type(tid, ptr_size);
        }
        assert!(
            gc.minor_collections >= before + 5,
            "stress_collect must force a collection on every allocation"
        );

        // The rooted object was forwarded; the root slot must now hold a
        // valid, non-nursery address.
        assert!(!root.is_null());
        assert!(
            !gc.is_in_nursery(root.0),
            "rooted object should have been promoted out of the nursery"
        );

        gc.roots.clear();
    }

    /// Isolation guarantee: with the `gc_stress` feature compiled in but
    /// `stress_collect` left off (the default), a large-nursery GC performs no
    /// forced collection — so suites that assert `minor_collections == 0` are
    /// unaffected even when the feature is enabled workspace-wide.
    #[test]
    #[cfg(feature = "gc_stress")]
    fn test_gc_stress_off_by_default() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1 << 20);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        for _ in 0..5 {
            let _ = gc.alloc_with_type(tid, ptr_size);
        }
        assert_eq!(
            gc.minor_collections, 0,
            "stress_collect defaults off: no collection without opt-in"
        );
    }

    #[test]
    fn test_incremental_gc_under_allocation_pressure() {
        // Allocate many objects forming a linked list, promote to old gen,
        // then run an incremental major cycle with a tiny budget while
        // continuing to allocate. Verify data integrity throughout.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        // Object layout: [next: GcRef][data: u64] = 16 bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size + 8, vec![0]));

        // Build a linked list of 20 objects.
        let mut prev = GcRef::NULL;
        let mut all_roots: Vec<GcRef> = Vec::new();
        for i in 0..20u64 {
            let obj = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
                *((obj.0 + ptr_size) as *mut u64) = 0xA000 + i;
            }
            prev = obj;
            all_roots.push(obj);
        }

        // Root only the head of the list.
        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote the whole list to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));

        // Start an incremental cycle with budget = 2 so it takes many steps.
        gc.set_mark_budget(2);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Interleave incremental marking steps with new allocations.
        let mut step_count = 0;
        while gc.is_incremental_marking() {
            // Allocate a few new (short-lived) objects to maintain pressure.
            for _ in 0..5 {
                let tmp = gc.alloc_with_type(tid, ptr_size + 8);
                unsafe {
                    *(tmp.0 as *mut GcRef) = GcRef::NULL;
                    *((tmp.0 + ptr_size) as *mut u64) = 0xFFFF;
                }
            }
            let done = gc.incremental_mark_step();
            step_count += 1;
            if done {
                break;
            }
            assert!(step_count < 200, "incremental marking should converge");
        }

        // Complete the cycle.
        gc.gc_step_until_scanning();
        assert!(gc.major_collections > 0);

        // Walk the list from the head and verify all data values.
        let mut cursor = head;
        let mut count = 0;
        while !cursor.is_null() {
            let data = unsafe { *((cursor.0 + ptr_size) as *const u64) };
            // data should be 0xA000 + (19 - count) because the list was
            // built in reverse.
            assert_eq!(
                data,
                0xA000 + (19 - count) as u64,
                "data corruption detected at node {count}"
            );
            cursor = unsafe { *(cursor.0 as *const GcRef) };
            count += 1;
        }
        assert_eq!(count, 20, "entire list should be reachable");

        gc.roots.clear();
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_card_marking_under_write_pressure() {
        // Allocate a large array in old gen, write GC refs into many slots,
        // and verify that card marking accurately tracks the dirty ranges
        // so that nursery objects stored in the array survive collection.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(2048);

        // Element type: simple 16-byte object.
        let elem_tid = gc.register_type(TypeInfo::simple(16));

        // Array type: varsize, base_size = 8 (length field at offset 0),
        // each item is a GcRef (item_size = ptr_size), items have GC ptrs.
        let arr_tid = gc.register_type(TypeInfo::varsize(
            8,          // base_size (length field)
            ptr_size,   // item_size
            0,          // length_offset
            true,       // items_have_gc_ptrs
            Vec::new(), // no fixed GC ptr fields
        ));

        let array_length = 512usize;
        let total_payload = 8 + ptr_size * array_length;

        // Allocate with inline card bytes (incminimark.py:1027 parity).
        let arr = gc.alloc_in_oldgen_with_cards(
            arr_tid,
            GcHeader::SIZE + total_payload,
            array_length,
            true,
        );

        // Write the length field.
        unsafe {
            *(arr.0 as *mut usize) = array_length;
        }

        let hdr = unsafe { header_of(arr.0) };
        assert!(unsafe { (*hdr).has_flag(flags::HAS_CARDS) });

        // Initialize all array slots to NULL.
        let items_start = arr.0 + 8;
        for i in 0..array_length {
            unsafe {
                *((items_start + i * ptr_size) as *mut GcRef) = GcRef::NULL;
            }
        }

        // Write nursery objects into scattered array positions and trigger
        // card-marking write barriers.
        let write_indices: Vec<usize> = vec![0, 1, 5, 64, 127, 128, 200, 255, 256, 400, 511];
        let mut expected_cards: Vec<usize> = Vec::new();
        let mut nursery_objs: Vec<(usize, GcRef)> = Vec::new();

        for &idx in &write_indices {
            let obj = gc.alloc_with_type(elem_tid, 16);
            // Write a distinctive marker.
            unsafe {
                *(obj.0 as *mut u64) = 0xCAFE_0000 + idx as u64;
            }
            // Store into the array.
            unsafe {
                *((items_start + idx * ptr_size) as *mut GcRef) = obj;
            }
            // Write barrier with card marking.
            gc.do_write_barrier_card(arr, idx, DEFAULT_CARD_PAGE_SHIFT);
            let card = idx >> DEFAULT_CARD_PAGE_SHIFT;
            if !expected_cards.contains(&card) {
                expected_cards.push(card);
            }
            nursery_objs.push((idx, obj));
        }

        // Verify the correct cards are dirty.
        let mut dirty_set: Vec<usize> = gc.dirty_cards(arr);
        dirty_set.sort();
        expected_cards.sort();
        assert_eq!(
            dirty_set, expected_cards,
            "dirty card set should match expected cards"
        );

        // Root the array so the collection traces it via card scanning.
        let mut root = arr;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger nursery collection — this should use card scanning
        // to find and promote the nursery objects stored in the array.
        gc.do_collect_nursery();

        // After collection, all stored objects should be promoted and
        // their data should be intact.
        for &(idx, _orig) in &nursery_objs {
            let slot_ref = unsafe { *((items_start + idx * ptr_size) as *const GcRef) };
            assert!(
                !slot_ref.is_null(),
                "array slot {idx} should not be null after collection"
            );
            assert!(
                !gc.is_in_nursery(slot_ref.0),
                "array slot {idx} should be promoted to old gen"
            );
            let marker = unsafe { *(slot_ref.0 as *const u64) };
            assert_eq!(
                marker,
                0xCAFE_0000 + idx as u64,
                "data in array slot {idx} should be preserved"
            );
        }

        // Cards should be cleared after collection.
        let hdr = unsafe { header_of(root.0) };
        assert!(
            unsafe { !(*hdr).has_flag(flags::CARDS_SET) },
            "CARDS_SET should be cleared after collection"
        );
        assert!(
            gc.old_objects_with_cards_set.is_empty(),
            "old_objects_with_cards_set should be empty after collection"
        );
        assert!(
            unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) },
            "TRACK_YOUNG_PTRS should be re-set after collection"
        );

        // Now do a second round of writes to verify card marking works
        // again after collection.
        let more_indices = vec![10, 300, 450];
        for &idx in &more_indices {
            let obj = gc.alloc_with_type(elem_tid, 16);
            unsafe {
                *(obj.0 as *mut u64) = 0xBEEF_0000 + idx as u64;
                *((items_start + idx * ptr_size) as *mut GcRef) = obj;
            }
            gc.do_write_barrier_card(arr, idx, DEFAULT_CARD_PAGE_SHIFT);
        }

        gc.do_collect_nursery();

        for &idx in &more_indices {
            let slot_ref = unsafe { *((items_start + idx * ptr_size) as *const GcRef) };
            assert!(!slot_ref.is_null());
            assert!(!gc.is_in_nursery(slot_ref.0));
            let marker = unsafe { *(slot_ref.0 as *const u64) };
            assert_eq!(marker, 0xBEEF_0000 + idx as u64);
        }

        gc.roots.clear();
    }

    // ── Write barrier + incremental marking interaction tests ──

    #[test]
    fn test_write_barrier_during_incremental_marking() {
        // Verify that write barriers fired during an active incremental
        // marking cycle correctly add old-gen objects to the remembered set.
        // The remembered set is processed at the next minor collection,
        // ensuring mutated old-gen objects are re-scanned.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Object layout: [field: GcRef] = ptr_size bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create 3 old-gen objects (A, B, C) with no pointers initially.
        let obj_a = gc.alloc_with_type(tid, ptr_size);
        let obj_b = gc.alloc_with_type(tid, ptr_size);
        let obj_c = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_a.0 as *mut GcRef) = GcRef::NULL;
            *(obj_b.0 as *mut GcRef) = GcRef::NULL;
            *(obj_c.0 as *mut GcRef) = GcRef::NULL;
        }

        let mut root_a = obj_a;
        let mut root_b = obj_b;
        let mut root_c = obj_c;
        unsafe {
            gc.roots.add(&mut root_a);
            gc.roots.add(&mut root_b);
            gc.roots.add(&mut root_c);
        }

        // Promote all to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_a.0));
        assert!(!gc.is_in_nursery(root_b.0));
        assert!(!gc.is_in_nursery(root_c.0));

        // All old-gen objects should have TRACK_YOUNG_PTRS set.
        assert!(unsafe { (*header_of(root_a.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert!(unsafe { (*header_of(root_b.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert!(unsafe { (*header_of(root_c.0)).has_flag(flags::TRACK_YOUNG_PTRS) });

        // Start an incremental marking cycle with budget=1 so it stays
        // active across multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // `seed_major_root` arms all newly gray roots for the next minor so
        // collector-owned JITFRAME spills cannot strand nursery references.
        // Reset that one-time collector setup here to isolate the ordinary
        // mutator write-barrier behavior this test covers.
        gc.remembered_set.clear();
        unsafe {
            (*header_of(root_a.0)).set_flag(flags::TRACK_YOUNG_PTRS);
            (*header_of(root_b.0)).set_flag(flags::TRACK_YOUNG_PTRS);
            (*header_of(root_c.0)).set_flag(flags::TRACK_YOUNG_PTRS);
        }

        // During marking, perform write barriers on A and B.
        gc.do_write_barrier(root_a);
        gc.do_write_barrier(root_b);

        // A and B should be in the remembered set.
        assert!(
            gc.remembered_set.contains(&root_a.0),
            "write barrier should add A to remembered set during marking"
        );
        assert!(
            gc.remembered_set.contains(&root_b.0),
            "write barrier should add B to remembered set during marking"
        );
        // C was not written to, so it shouldn't be in remembered set.
        assert!(
            !gc.remembered_set.contains(&root_c.0),
            "C should not be in remembered set"
        );

        // TRACK_YOUNG_PTRS should be cleared on A and B.
        assert!(!unsafe { (*header_of(root_a.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert!(!unsafe { (*header_of(root_b.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
        // C still has it.
        assert!(unsafe { (*header_of(root_c.0)).has_flag(flags::TRACK_YOUNG_PTRS) });

        // Drive the incremental cycle to completion via nursery collections.
        for _ in 0..50 {
            if !gc.is_incremental_marking() {
                break;
            }
            gc.do_collect_nursery();
        }

        // No objects should be lost — all 3 are still rooted and should
        // survive the sweep.
        gc.do_collect_full();
        assert_eq!(
            gc.oldgen.object_count(),
            3,
            "all 3 rooted objects should survive full collection"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_mutation_during_incremental_preserves_reachability() {
        // During an incremental marking cycle, mutate an old-gen object to
        // point to a newly promoted object (D). The write barrier ensures D
        // is reachable through the remembered set, so D survives the sweep.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Object layout: [next: GcRef] = ptr_size bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Build a chain A→B→C in the nursery.
        let obj_c = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_c.0 as *mut GcRef) = GcRef::NULL;
        }
        let obj_b = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_b.0 as *mut GcRef) = obj_c;
        }
        let obj_a = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_a.0 as *mut GcRef) = obj_b;
        }

        let mut root_a = obj_a;
        unsafe {
            gc.roots.add(&mut root_a);
        }

        // Promote A→B→C to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_a.0));
        let promoted_b = unsafe { *(root_a.0 as *const GcRef) };
        let promoted_c = unsafe { *(promoted_b.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(promoted_b.0));
        assert!(!gc.is_in_nursery(promoted_c.0));

        // Start incremental marking with a small budget.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Do one marking step — only partially through the graph.
        gc.incremental_mark_step();

        // Now allocate D in the nursery.
        let obj_d = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_d.0 as *mut GcRef) = GcRef::NULL;
        }

        // Promote D to old gen by rooting it temporarily.
        let mut root_d = obj_d;
        unsafe {
            gc.roots.add(&mut root_d);
        }
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_d.0));

        // Mutate A to point to D instead of B (A→D).
        // The write barrier ensures A is in the remembered set.
        gc.do_write_barrier(root_a);
        unsafe {
            *(root_a.0 as *mut GcRef) = root_d;
        }

        // Remove D's direct root — D is now only reachable via A→D.
        gc.roots.remove(&mut root_d);

        // Complete the incremental cycle: drive marking to completion
        // and then sweep.
        gc.do_collect_full();

        // A and D should survive (A is rooted, D reachable via A→D).
        // B and C may or may not survive depending on marking order,
        // but D MUST survive.
        let d_addr = root_d.0;
        let a_field = unsafe { *(root_a.0 as *const GcRef) };
        assert_eq!(
            a_field.0, d_addr,
            "A should still point to D after collection"
        );

        // Verify D is actually alive by checking it's still in old gen.
        assert!(
            !a_field.is_null(),
            "D must survive collection — it's reachable via A"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_nursery_alloc_during_incremental_marking() {
        // Allocate new nursery objects between incremental marking steps.
        // Trigger nursery collections that piggyback marking steps.
        // Verify that newly allocated objects are correctly promoted and
        // that the original old-gen object graph remains intact.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024); // small nursery to force frequent collections
        // Object layout: [next: GcRef][data: u64] = ptr_size + 8 bytes
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size + 8, vec![0]));

        // Create initial old-gen objects (a small chain of 5).
        let mut prev = GcRef::NULL;
        for i in 0..5u64 {
            let obj = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
                *((obj.0 + ptr_size) as *mut u64) = 0xBB00 + i;
            }
            prev = obj;
        }

        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote the chain to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));
        let old_count_after_promote = gc.oldgen.object_count();
        assert_eq!(old_count_after_promote, 5);

        // Start incremental marking with budget=1.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        let minor_before = gc.minor_collections;

        // Allocate many nursery objects between marking steps. The small
        // nursery (1024 bytes) forces nursery collections that piggyback
        // incremental marking steps.
        for i in 0..100u64 {
            let tmp = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(tmp.0 as *mut GcRef) = GcRef::NULL;
                *((tmp.0 + ptr_size) as *mut u64) = 0xDD00 + i;
            }
        }

        let minor_during = gc.minor_collections - minor_before;
        assert!(
            minor_during >= 2,
            "should have triggered multiple nursery collections, got {minor_during}"
        );

        // The incremental marking should have advanced via piggybacking.
        assert!(
            gc.incremental_objects_marked() > 0,
            "piggybacked marking should have processed some objects"
        );

        // Drive any remaining incremental marking to completion and sweep.
        // Use do_collect_full which correctly handles an in-progress cycle.
        gc.do_collect_full();
        assert!(gc.major_collections > 0);

        // Verify the original chain is intact — the 5 rooted objects
        // survived the incremental major cycle.
        let mut cursor = head;
        let mut count = 0;
        while !cursor.is_null() {
            let data = unsafe { *((cursor.0 + ptr_size) as *const u64) };
            assert_eq!(
                data,
                0xBB00 + (4 - count) as u64,
                "original chain data corrupted at node {count}"
            );
            cursor = unsafe { *(cursor.0 as *const GcRef) };
            count += 1;
        }
        assert_eq!(count, 5, "entire chain should be reachable after cycle");

        gc.roots.clear();
    }

    #[test]
    fn full_collection_finishes_active_cycle_then_runs_fresh_cycle() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut rooted = gc.alloc_with_type(tid, 16);
        unsafe { gc.roots.add(&mut rooted) };
        gc.do_collect_nursery();

        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());
        let major_before = gc.major_collections;

        gc.do_collect_full();

        assert_eq!(gc.major_collections, major_before + 2);
        assert!(gc.oldgen.contains(rooted.0));
        gc.roots.clear();
    }

    #[test]
    fn full_collection_forwards_nursery_references_held_by_gray_objects() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let holder_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::simple(16));

        let holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe { *(holder.0 as *mut GcRef) = GcRef::NULL };
        let mut root = holder;
        unsafe { gc.roots.add(&mut root) };
        gc.do_collect_nursery();

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert_eq!(gc.incr_state.gray_stack, vec![root.0]);

        let young = gc.alloc_with_type(leaf_tid, 16);
        unsafe {
            *(young.0 as *mut u64) = 0xF011_C011_EC70;
            *(root.0 as *mut GcRef) = young;
        }

        // incminimark.py `minor_and_major_collection` finishes the active
        // cycle through `gc_step_until`, whose leading minor forwards this
        // edge before the gray holder is consumed.
        gc.do_collect_full();

        let forwarded = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(forwarded.0));
        assert_eq!(unsafe { *(forwarded.0 as *const u64) }, 0xF011_C011_EC70);
        assert!(gc.oldgen.contains(forwarded.0));
        gc.roots.clear();
    }

    // ── JIT integration hook tests ──

    #[test]
    fn test_jit_remember_young_pointer() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // Allocate an old-gen object.
        let obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        assert!(!gc.is_in_nursery(obj.0));

        // Initially TRACK_YOUNG_PTRS is set.
        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) });

        // JIT fast-path barrier: clears flag and adds to remembered set.
        gc.jit_remember_young_pointer(obj);

        assert!(unsafe { !(*hdr).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert_eq!(gc.remembered_set_len(), 1);

        // Calling again adds a second entry (JIT fast-path does not
        // deduplicate; the collector handles this during minor collection).
        gc.jit_remember_young_pointer(obj);
        assert_eq!(gc.remembered_set_len(), 2);
    }

    #[test]
    fn test_jit_remember_young_pointer_survives_collection() {
        // Verify that the remembered-set entry from jit_remember_young_pointer
        // causes a young object to survive minor collection.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create an old-gen parent and a young child.
        let parent = gc.alloc_in_oldgen(tid, GcHeader::SIZE + std::mem::size_of::<GcRef>());
        let child = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(child.0 as *mut u64) = 0xABCD_1234;
            *(parent.0 as *mut GcRef) = child;
        }

        // Use the JIT hook instead of do_write_barrier.
        gc.jit_remember_young_pointer(parent);

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // The child should have been promoted.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());
        let val = unsafe { *(child_ref.0 as *const u64) };
        assert_eq!(val, 0xABCD_1234);

        gc.roots.clear();
    }

    #[test]
    fn test_can_optimize_cond_call() {
        let gc = test_gc(4096);
        assert!(gc.can_optimize_cond_call());
    }

    /// env.py:149-210 `get_L2cache_linux2_cpuinfo`: use the smallest valid
    /// per-CPU K/k value, ignore another unit and malformed lines, and retain
    /// the upstream `_findend('\n' + label)` first-line behavior.
    #[test]
    fn linux_cpuinfo_cache_probe_matches_upstream_parser() {
        let data = b"cache size : 1 KB\n\
processor : 0\n\
cache size : 32768 KB\n\
cache size : 16 MB\n\
cache size = 8 KB\n\
processor : 1\n\
cache size\t: 8192 kB\n";
        assert_eq!(l2cache_from_cpuinfo(data, b"cache size"), 8 * 1024 * 1024);
        assert_eq!(l2cache_from_cpuinfo(data, b"L2 cache"), -1);
    }

    /// env.py:287-356 `get_L2cache_linux2_system_cpu_index`: L2 and L3 are
    /// minimized independently across CPUs and then added.
    #[test]
    fn linux_sysfs_cache_probe_sums_smallest_l2_and_l3() {
        static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);
        let serial = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "majit-gc-cache-fixture-{}-{serial}",
            std::process::id()
        ));

        let entries = [
            (0, 0, "1\n", "64K\n"),
            (0, 1, "2\n", "2048K\n"),
            (0, 2, "3\n", "8192K\n"),
            (1, 0, "2\n", "1024K\n"),
            (1, 1, "3\n", "4096K\n"),
        ];
        for (cpu, index, level, size) in entries {
            let dir = root.join(format!("cpu{cpu}/cache/index{index}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("level"), level).unwrap();
            std::fs::write(dir.join("size"), size).unwrap();
        }

        let result = get_l2cache_linux_system_cpu_index(root.to_str().unwrap());
        assert_eq!(result, 5 * 1024 * 1024);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn linux_sysfs_cache_size_requires_leading_decimal_kilobytes() {
        assert_eq!(parse_sysfs_cache_size(b"2048K\n"), Some(2 * 1024 * 1024));
        assert_eq!(parse_sysfs_cache_size(b"512k\n"), Some(512 * 1024));
        assert_eq!(parse_sysfs_cache_size(b"2M\n"), None);
        assert_eq!(parse_sysfs_cache_size(b" K\n"), None);
    }

    #[test]
    fn test_estimate_best_nursery_size() {
        // env.py:443-456 — half the L2 cache when it exceeds 8MB, else the
        // 4MB unknown-cache fallback. Either way it must never return a size
        // below the fallback, so the major-collection floor (nursery*8) is
        // always well-defined.
        let est = estimate_best_nursery_size();
        assert!(est >= DEFAULT_NURSERY_SIZE, "estimate {est} below fallback");
        // best_nursery_size_for_l2cache mirrors env.py's strict `> 8MB` test.
        assert_eq!(best_nursery_size_for_l2cache(-1), DEFAULT_NURSERY_SIZE);
        assert_eq!(
            best_nursery_size_for_l2cache(8 * 1024 * 1024),
            DEFAULT_NURSERY_SIZE
        );
        assert_eq!(
            best_nursery_size_for_l2cache(32 * 1024 * 1024),
            16 * 1024 * 1024
        );
    }

    #[test]
    fn test_can_optimize_cond_call_via_trait() {
        let gc = test_gc(4096);
        let alloc: &dyn GcAllocator = &gc;
        assert!(alloc.can_optimize_cond_call());
    }

    #[test]
    fn test_gc_step_no_work_when_old_gen_small() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // With an almost-empty old gen, gc_step should do nothing.
        assert!(!gc.gc_step());
        assert!(!gc.is_incremental_marking());
    }

    #[test]
    fn test_gc_step_respects_enabled_flag() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));
        gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        gc.next_major_collection_threshold = 0.0;
        assert!(gc.threshold_reached(0));

        gc.disable();
        assert!(!gc.gc_step());
        assert_eq!(gc.gc_state, GcState::Scanning);

        gc.enable();
        assert!(gc.gc_step());
    }

    #[test]
    fn test_gc_step_triggers_incremental() {
        let mut gc = test_gc(256);
        gc.register_type(TypeInfo::simple(16));

        // Force a full collection to establish the next-major threshold baseline.
        let obj = gc.alloc_with_type(0, 16);
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }
        gc.collect_full();

        // Fill old gen to trigger the major-cycle ratio threshold.
        // Allocate many objects directly in old gen.
        for _ in 0..200 {
            gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        }

        // gc_step should now start an incremental cycle and do work.
        let did_work = gc.gc_step();
        assert!(did_work);

        gc.roots.clear();
    }

    #[test]
    fn test_gc_step_advances_marking() {
        // Nursery must be large enough to hold all 20 chain links without an
        // auto-collect mid-loop; otherwise intermediate links get dropped and
        // `prev` ends up chaining through stale nursery addresses (reset-zero
        // bytes parse as tid=0 / `simple(16)`, which causes `copy_nursery_object`
        // to over-read past the nursery end).
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Build a chain of old-gen objects so there's marking work to do.
        let mut prev = GcRef::NULL;
        let ptr_tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));
        for _ in 0..20 {
            let obj = gc.alloc_with_type(ptr_tid, std::mem::size_of::<GcRef>());
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
        }

        let mut root = prev;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Promote everything to old gen.
        gc.collect_nursery();

        // Add enough old-gen objects to cross the next-major threshold. The
        // threshold is floored at `min_heap_size = nursery_size * 8` = 32KB
        // for this 4096-byte nursery (incminimark.py:488,562), so the old gen
        // must exceed 32KB before `gc_step` will start a cycle — a few hundred
        // small objects is intentionally not enough under the parity-correct
        // floor.
        for _ in 0..4000 {
            gc.alloc_in_oldgen(tid, GcHeader::SIZE + 16);
        }

        // First step: start cycle.
        let work1 = gc.gc_step();
        assert!(work1);
        let marked_after_1 = gc.incremental_objects_marked();

        // Second step: advance marking further.
        if gc.is_incremental_marking() {
            let work2 = gc.gc_step();
            assert!(work2);
            assert!(gc.incremental_objects_marked() >= marked_after_1);
        }

        gc.roots.clear();
    }

    // ── Pin / Unpin / jit_free tests ──

    #[test]
    fn test_pin_prevents_nursery_move() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write a marker value.
        unsafe {
            *(obj.0 as *mut u64) = 0xCAFE_BABE;
        }

        // Pin the object and root it.
        assert!(gc.pin(obj));
        assert!(gc.is_pinned(obj));

        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger minor collection.
        gc.do_collect_nursery();

        // The root should still point to the same nursery address.
        assert_eq!(root.0, obj.0);
        assert!(gc.is_in_nursery(root.0));

        // Data should be intact.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xCAFE_BABE);

        gc.roots.clear();
    }

    #[test]
    fn test_can_move_nursery_pin_and_bounds() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // rgc.py:229 — a young nursery object can still move.
        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));
        assert!(gc.can_move(obj));

        // incminimark.py:1117-1119 answers by nursery membership even while
        // the object's PINNED flag temporarily prevents an actual move.
        assert!(gc.pin(obj));
        assert!(gc.can_move(obj));
        gc.unpin(obj);
        assert!(gc.can_move(obj));

        // Null and out-of-nursery addresses never move (rgc.py:231
        // "with non-moving GCs, it is always False").
        assert!(!gc.can_move(GcRef(0)));
        assert!(!gc.can_move(GcRef(0xdead_0000)));
    }

    #[test]
    fn test_unpin_allows_move() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write a marker.
        unsafe {
            *(obj.0 as *mut u64) = 0xDEAD_BEEF;
        }

        // Pin, then unpin.
        assert!(gc.pin(obj));
        gc.unpin(obj);
        assert!(!gc.is_pinned(obj));

        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Collection should now move the object to old gen.
        gc.do_collect_nursery();

        // The root should now point to old gen (different address).
        assert!(!gc.is_in_nursery(root.0));
        assert_ne!(root.0, obj.0);

        // Data should be preserved after the move.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xDEAD_BEEF);

        gc.roots.clear();
    }

    #[test]
    fn test_is_pinned_query() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);

        // Not pinned by default.
        assert!(!gc.is_pinned(obj));

        // Pin it.
        assert!(gc.pin(obj));
        assert!(gc.is_pinned(obj));

        // incminimark.py:1129-1134 rejects a second pin rather than keeping a
        // reference count whose first unpin could invalidate the second pin.
        assert!(!gc.pin(obj));

        // Null cannot be pinned.
        assert!(!gc.pin(GcRef(0)));
        assert!(!gc.is_pinned(GcRef(0)));

        // Unpin.
        gc.unpin(obj);
        assert!(!gc.is_pinned(obj));
    }

    #[test]
    fn test_pin_enforces_capacity_and_cannot_pin_type_query() {
        unsafe fn noop_destructor(_obj_addr: usize) {}
        fn trigger() {}

        let mut gc = test_gc(4096);
        gc.max_number_of_pinned_objects = 1;
        let plain_tid = gc.register_type(TypeInfo::simple(16));
        let gcptr_tid = gc.register_type(TypeInfo::with_gc_ptrs(16, vec![0]));
        let weakref_tid = gc.register_type(TypeInfo::weakref());
        let destructor_tid = gc.register_type(TypeInfo::with_destructor(16, noop_destructor));

        let first = gc.alloc_with_type(plain_tid, 16);
        let second = gc.alloc_with_type(plain_tid, 16);
        assert!(gc.pin(first));
        assert_eq!(gc.pinned_objects_in_nursery, 1);
        // incminimark.py:1122-1123 checks the configured count first.
        assert!(!gc.pin(second));
        gc.unpin(first);
        assert_eq!(gc.pinned_objects_in_nursery, 0);

        // gctypelayout.py:89-92 `q_cannot_pin`.
        let with_gcptr = gc.alloc_with_type(gcptr_tid, 16);
        let weakref = gc.alloc_with_type(weakref_tid, crate::weakref::SIZEOF_WEAKREF);
        let with_destructor = gc.alloc_with_type(destructor_tid, 16);
        assert!(!gc.pin(with_gcptr));
        assert!(!gc.pin(weakref));
        assert!(!gc.pin(with_destructor));

        // Pyre's finalizer queue is registered dynamically rather than
        // encoded as RPython `customdata`, so the header bit supplies the same
        // rejection once registration has occurred.
        let finalizable = gc.alloc_with_type(plain_tid, 16);
        GcAllocator::register_finalizer(&mut gc, 0, finalizable, trigger);
        assert!(!gc.pin(finalizable));
    }

    #[test]
    fn test_unreachable_pin_is_not_an_implicit_root() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(tid, 16);
        assert!(gc.pin(obj));

        // incminimark.py:1777-1785: the survivor stack is populated only by
        // traced references. With no such reference, the pinned object dies.
        gc.do_collect_nursery();
        assert!(!gc.is_pinned(obj));
        assert_eq!(gc.pinned_objects_in_nursery, 0);
    }

    #[test]
    fn test_old_parent_is_retraced_while_it_points_to_a_pin() {
        let mut gc = test_gc(4096);
        let child_tid = gc.register_type(TypeInfo::simple(16));
        let parent_tid = gc.register_type(TypeInfo::with_gc_ptrs(16, vec![0]));
        let child = gc.alloc_with_type(child_tid, 16);
        let mut parent = gc.alloc_with_type(parent_tid, 16);
        unsafe { *(parent.0 as *mut GcRef) = child };
        assert!(gc.pin(child));
        unsafe { gc.roots.add(&mut parent) };

        // The parent promotes, but its pinned child stays at the same nursery
        // address and records the old parent for the following minor.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(parent.0));
        assert!(gc.is_pinned(child));
        assert_eq!(unsafe { *(parent.0 as *const GcRef) }, child);
        assert_eq!(gc.old_objects_pointing_to_pinned, vec![parent.0]);

        // No remembered-set entry is needed: the dedicated PyPy parent list
        // revisits the edge and keeps the child alive for another cycle.
        gc.do_collect_nursery();
        assert!(gc.is_pinned(child));
        assert_eq!(gc.old_objects_pointing_to_pinned, vec![parent.0]);

        // Once that edge disappears, the next minor does not rediscover the
        // child and pinning alone no longer preserves it.
        unsafe { *(parent.0 as *mut GcRef) = GcRef::NULL };
        gc.do_collect_nursery();
        assert!(!gc.is_pinned(child));
        assert_eq!(gc.pinned_objects_in_nursery, 0);
        assert!(gc.old_objects_pointing_to_pinned.is_empty());
        gc.roots.clear();
    }

    /// The sibling above only ever discovers the parent *after* the list is
    /// swapped out. Phase 1c traces an old-generation jitframe directly, with
    /// itself as the holder, and that runs earlier — so a parent found there
    /// would be re-flagged before the drained copy is visited, never make it
    /// into the fresh list, and keep `PINNED` set for good.
    #[test]
    fn test_old_parent_found_before_the_swap_stays_in_the_list() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock().unwrap();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let child_tid = gc.register_type(TypeInfo::simple(16));
        let parent_tid = gc.register_type(TypeInfo::with_gc_ptrs(16, vec![0]));
        let child = gc.alloc_with_type(child_tid, 16);
        let mut parent = gc.alloc_with_type(parent_tid, 16);
        unsafe { *(parent.0 as *mut GcRef) = child };
        assert!(gc.pin(child));
        unsafe { gc.roots.add(&mut parent) };

        gc.do_collect_nursery();
        assert!(gc.oldgen.contains(parent.0));
        assert!(gc.is_pinned(child));
        assert_eq!(gc.old_objects_pointing_to_pinned, vec![parent.0]);

        // Publish the promoted parent as a jitframe root so the minor below
        // reaches it through the pre-swap old-generation arm as well.
        crate::shadow_stack::push_jf(parent);
        gc.do_collect_nursery();
        assert_eq!(gc.old_objects_pointing_to_pinned, vec![parent.0]);
        assert!(gc.is_pinned(child));

        // The edge is still the child's only reference, so losing the parent
        // record above would let this minor reclaim it.
        gc.do_collect_nursery();
        assert!(gc.is_pinned(child));
        assert_eq!(unsafe { *(parent.0 as *const GcRef) }, child);

        crate::shadow_stack::pop_jf_to(0);
        gc.roots.clear();
    }

    #[test]
    fn test_jit_free_unregisters_code() {
        let mut gc = test_gc(4096);

        let smap = SafepointMap::new();
        gc.compiled_code_registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 256,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 1,
        });

        let smap2 = SafepointMap::new();
        gc.compiled_code_registry.register(CompiledCodeRegion {
            code_start: 0x2000,
            code_size: 512,
            safepoint_map: smap2,
            frame_size_slots: 8,
            loop_token: 2,
        });

        assert_eq!(gc.compiled_code_registry.len(), 2);

        // Free the first region.
        gc.jit_free(0x1000, 256);

        assert_eq!(gc.compiled_code_registry.len(), 1);
        assert!(gc.compiled_code_registry.find_region(0x1050).is_none());
        assert!(gc.compiled_code_registry.find_region(0x2050).is_some());

        // Free the second region.
        gc.jit_free(0x2000, 512);
        assert_eq!(gc.compiled_code_registry.len(), 0);
    }

    #[test]
    fn test_incremental_cycle_root_walk_skips_non_gc_roots() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let mut rooted = obj;
        let mut external = GcRef(Box::into_raw(Box::new(0xDEADBEEFu64)) as usize);
        unsafe {
            gc.roots.add(&mut rooted);
            gc.roots.add(&mut external);
        }

        gc.do_collect_nursery();
        gc.start_incremental_cycle();
        while !gc.gc_step() {}

        assert!(!rooted.is_null());
        assert!(gc.oldgen.contains(rooted.0));

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(external.0 as *mut u64));
        }
    }

    #[test]
    fn test_full_collection_root_walk_skips_non_gc_roots() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let mut rooted = obj;
        let mut external = GcRef(Box::into_raw(Box::new(0xFEEDFACEu64)) as usize);
        unsafe {
            gc.roots.add(&mut rooted);
            gc.roots.add(&mut external);
        }

        gc.do_collect_nursery();
        gc.do_collect_full();

        assert!(!rooted.is_null());
        assert!(gc.oldgen.contains(rooted.0));

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(external.0 as *mut u64));
        }
    }

    /// A minor collection rewrites the SLOT, so reading through the position
    /// yields the promoted address.
    ///
    /// `walk_stack_root` (`shadowstack.py:44-70`) hands the collector the slot
    /// itself — `invoke(..., addr)` where `addr` is the stack address, not the
    /// value — which is what lets the forwarded address be stored back into it.
    /// Every holder that addresses its root by position therefore sees the move
    /// without being told about it.
    #[test]
    fn minor_collection_rewrites_an_owner_root_slot() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock().unwrap();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        assert!(gc.is_in_nursery(obj.0));
        unsafe {
            *(obj.0 as *mut u64) = 0xD00D_D00D;
        }
        let root = crate::shadow_stack::OwnerRootGuard::new(obj);

        gc.do_collect_nursery();

        let moved = root.get();
        assert_ne!(
            moved, obj,
            "a surviving nursery object is promoted, so this run cannot tell a \
             rewritten slot from an unmoved object"
        );
        assert!(gc.oldgen.contains(moved.0));
        assert_eq!(unsafe { *(moved.0 as *const u64) }, 0xD00D_D00D);
    }

    /// The HOLDER may move. Addressing a root by position is what makes that
    /// true, and it is the whole difference from registering the address of the
    /// field that holds the pointer: a registered field address is only valid
    /// while its owner stays put, so an owner that can be returned by value has
    /// to be pinned behind a heap allocation first.
    #[test]
    fn an_owner_root_survives_its_holder_being_moved() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock().unwrap();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        assert!(gc.is_in_nursery(obj.0));
        unsafe {
            *(obj.0 as *mut u64) = 0xFEED_FEED;
        }

        // Three distinct addresses for the same guard, one of them the heap,
        // with a collection between the last move and the read.
        let root = crate::shadow_stack::OwnerRootGuard::new(obj);
        let moved_once = root;
        let mut holders = vec![moved_once];
        gc.do_collect_nursery();

        let moved = holders.pop().unwrap().get();
        assert_ne!(moved, obj);
        assert!(gc.oldgen.contains(moved.0));
        assert_eq!(unsafe { *(moved.0 as *const u64) }, 0xFEED_FEED);
    }

    #[test]
    fn test_incremental_cycle_marks_jitframe_shadow_stack_roots() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock().unwrap();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let depth = crate::shadow_stack::jf_depth();
        crate::shadow_stack::push_jf(obj);

        gc.do_collect_nursery();
        let rooted = crate::shadow_stack::jf_top_ptr();
        assert!(gc.oldgen.contains(rooted.0));

        gc.start_incremental_cycle();
        gc.gc_step_until_scanning();

        assert!(gc.oldgen.contains(rooted.0));
        crate::shadow_stack::pop_jf_to(depth);
    }

    #[test]
    fn test_full_collection_marks_jitframe_shadow_stack_roots() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock().unwrap();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let depth = crate::shadow_stack::jf_depth();
        crate::shadow_stack::push_jf(obj);

        gc.do_collect_nursery();
        let rooted = crate::shadow_stack::jf_top_ptr();
        assert!(gc.oldgen.contains(rooted.0));

        gc.do_collect_full();

        assert!(gc.oldgen.contains(rooted.0));
        crate::shadow_stack::pop_jf_to(depth);
    }

    // Note: a `test_minor_root_walk_skips_interior_nursery_pointers`
    // test used to live here. It was predicated on a majit-local
    // extension where the nursery tracked exact object-start addresses
    // and `is_nursery_object_start` filtered out interior pointers.
    //
    // incminimark.py:1208 is_in_nursery is a pure range check
    //     return self.nursery <= addr < self.nursery + self.nursery_size
    // RPython guarantees that GC roots are exact object pointers (the
    // shadow stack only ever carries exact pointers produced by the
    // compiler), so interior-pointer filtering is not part of the GC
    // contract. The test disagreed with that contract and was removed
    // to keep majit-gc structurally aligned with RPython.

    /// incminimark.py:3068-3079 dead-target branch. A WEAKREF whose
    /// target is a nursery object with no GC root must have its
    /// `weakptr` slot invalidated to 0 after the minor cycle.
    #[test]
    fn test_minor_invalidate_young_weakref_dead_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        // Allocate the target first, then a WEAKREF whose payload's
        // `weakptr` slot points to that target. Neither object is
        // rooted in `gc.roots`; only the WEAKREF itself is rooted so
        // the minor cycle keeps the WEAKREF alive long enough to
        // observe the post-invalidation slot.
        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        let mut wref_root = wref;
        unsafe { gc.roots.add(&mut wref_root) };

        gc.do_collect_nursery();

        // The WEAKREF survived → forwarded into old-gen.
        assert!(gc.oldgen.contains(wref_root.0));
        // Target had no root and was not reachable through the WEAKREF
        // (the collector does not trace weakptr), so it died — the
        // slot must read null.
        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert!(after.is_null(), "dead-target weakptr should read null");

        gc.roots.clear();
    }

    /// incminimark.py:3071-3074 live-target branch. A WEAKREF whose
    /// target survives the minor cycle (because something else
    /// rooted it) must have its `weakptr` slot rewritten to the
    /// target's new old-gen address.
    #[test]
    fn test_minor_invalidate_young_weakref_live_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        // Root both the target and the weakref so both forward.
        let mut target_root = target;
        let mut wref_root = wref;
        unsafe {
            gc.roots.add(&mut target_root);
            gc.roots.add(&mut wref_root);
        }

        gc.do_collect_nursery();

        // Both forwarded out.
        assert!(gc.oldgen.contains(target_root.0));
        assert!(gc.oldgen.contains(wref_root.0));
        // The weakref's slot now reads the target's forwarded address.
        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert_eq!(after.0, target_root.0);

        gc.roots.clear();
    }

    /// incminimark.py:3065-3066 "weakref itself dies" branch. The
    /// WEAKREF object has no root → the cycle treats the entry as a
    /// no-op (no panic, no UB) and the bookkeeping list ends up empty.
    #[test]
    fn test_minor_invalidate_young_weakref_self_dies() {
        let mut gc = test_gc(4096);
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let _wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        // No root on the WEAKREF itself.
        assert_eq!(gc.young_objects_with_weakrefs.len(), 1);

        gc.do_collect_nursery();

        // The cycle drained the list and didn't push the dead weakref
        // into the old-side queue.
        assert!(gc.young_objects_with_weakrefs.is_empty());
        assert!(gc.old_objects_with_weakrefs.is_empty());
    }

    /// incminimark.py:3116-3122 dying-target branch in
    /// `invalidate_old_weakrefs`. After a full GC where the WEAKREF's
    /// target is no longer rooted, the weakptr slot reads null and
    /// the bookkeeping list drops the weakref.
    #[test]
    fn test_major_invalidate_old_weakref_dead_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        // Root all three (target, wref) and pre-promote them to old
        // gen via a minor collection so the major cycle sees both as
        // oldgen-resident.
        let mut target_root = target;
        let mut wref_root = wref;
        unsafe {
            gc.roots.add(&mut target_root);
            gc.roots.add(&mut wref_root);
        }
        gc.do_collect_nursery();
        assert!(gc.oldgen.contains(target_root.0));
        assert!(gc.oldgen.contains(wref_root.0));
        assert_eq!(gc.old_objects_with_weakrefs.len(), 1);

        // Drop the target root; keep wref rooted. Full GC should
        // sweep the target and null out wref's weakptr slot.
        gc.roots.remove(&mut target_root);
        gc.do_collect_full();

        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert!(after.is_null(), "dead-target old weakptr should read null");
        // Weakref survived but its entry was popped from the list
        // because its (now-null) slot has nothing left to clear.
        assert!(gc.old_objects_with_weakrefs.is_empty());

        gc.roots.clear();
    }

    /// incminimark.py:3120-3121 live-target keep branch. When both
    /// the WEAKREF and its target survive the major mark, the
    /// weakref carries over into the next cycle's bookkeeping list.
    #[test]
    fn test_major_invalidate_old_weakref_live_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        let mut target_root = target;
        let mut wref_root = wref;
        unsafe {
            gc.roots.add(&mut target_root);
            gc.roots.add(&mut wref_root);
        }

        gc.do_collect_full();

        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert_eq!(after.0, target_root.0);
        assert_eq!(gc.old_objects_with_weakrefs.len(), 1);

        gc.roots.clear();
    }

    /// A non-moving major (`do_collect_oldgen_nonmoving`) sweeps dead old-gen
    /// objects while a populated nursery is left byte-for-byte intact, marks
    /// through nursery objects to keep `old -> nursery -> old` survivors, and
    /// clears `VISITED` off every greyed nursery object as the strictly-last
    /// step — otherwise a later minor promotion would memcpy a stale VISITED
    /// bit into the promoted copy.
    #[test]
    fn nonmoving_major_marks_through_nursery_and_clears_visited() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // q: an old-gen leaf reachable ONLY through a nursery object.
        let q = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(q.0 as *mut GcRef) = GcRef(0) };
        // n: nursery object pointing at q (nursery -> old edge).
        let n = gc.alloc_with_type(tid, ptr_size);
        assert!(gc.is_in_nursery(n.0));
        unsafe { *(n.0 as *mut GcRef) = q };
        // o: old-gen root pointing at n (old -> nursery edge).
        let o = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(o.0 as *mut GcRef) = n };
        // d: unreachable old-gen object that must be swept.
        let d = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(d.0 as *mut GcRef) = GcRef(0) };

        let mut root = o;
        unsafe { gc.roots.add(&mut root) };
        assert_eq!(gc.oldgen.object_count(), 3); // o, q, d

        gc.do_collect_oldgen_nonmoving();

        // o and q survive (o -> n -> q); d swept. The nursery is untouched.
        assert_eq!(gc.oldgen.object_count(), 2);
        assert!(gc.is_in_nursery(n.0), "nursery object must not move/free");
        assert_eq!(unsafe { *(n.0 as *const GcRef) }, q, "n -> q edge intact");

        // The must-fix: no greyed nursery object retains VISITED.
        let n_hdr = unsafe { header_of(n.0) };
        assert!(
            unsafe { !(*n_hdr).has_flag(flags::VISITED) },
            "stale nursery VISITED would memcpy into the next minor's promotion"
        );
        // q (old-gen survivor) had VISITED cleared by the oldgen sweep.
        let q_hdr = unsafe { header_of(q.0) };
        assert!(unsafe { !(*q_hdr).has_flag(flags::VISITED) });

        gc.roots.clear();
    }

    /// A non-moving major must run `invalidate_old_weakrefs` (reads the
    /// target's VISITED) BEFORE the nursery-VISITED clear, so an old weakref
    /// whose target is a live nursery object is kept, not spuriously nulled.
    #[test]
    fn nonmoving_major_keeps_live_nursery_weakref() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        // n: nursery-resident weakref target, kept live by a root.
        let n = gc.alloc_with_type(target_tid, 16);
        assert!(gc.is_in_nursery(n.0));
        // w: weakref born directly in the old generation, pointing at n.
        // `alloc_in_oldgen` records born-old weakrefs onto
        // `old_objects_with_weakrefs`, so no manual registration is needed.
        let w = gc.alloc_in_oldgen(wref_tid, GcHeader::SIZE + crate::weakref::SIZEOF_WEAKREF);
        unsafe { *((w.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = n };

        let mut w_root = w;
        let mut n_root = n;
        unsafe {
            gc.roots.add(&mut w_root);
            gc.roots.add(&mut n_root);
        }

        gc.do_collect_oldgen_nonmoving();

        // n was marked-through (live), so the weakref slot is NOT nulled:
        // invalidate_old_weakrefs read n's VISITED before the clear pass ran.
        let after =
            unsafe { crate::weakref::ll_weakref_deref(w_root.0 as *const crate::weakref::Weakref) };
        assert_eq!(
            after.0, n_root.0,
            "live nursery weakref target must survive"
        );
        assert_eq!(gc.old_objects_with_weakrefs.len(), 1);
        assert!(gc.is_in_nursery(n_root.0));
        let n_hdr = unsafe { header_of(n_root.0) };
        assert!(unsafe { !(*n_hdr).has_flag(flags::VISITED) });

        gc.roots.clear();
    }

    #[test]
    fn nonmoving_major_invalidates_live_nursery_weakref_to_dead_old_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_in_oldgen(target_tid, GcHeader::SIZE + 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }
        let mut wref_root = wref;
        unsafe { gc.roots.add(&mut wref_root) };

        gc.do_collect_oldgen_nonmoving();

        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert_eq!(after.0, 0);
        assert_eq!(gc.young_objects_with_weakrefs, vec![wref.0]);
        gc.roots.clear();
    }

    #[test]
    fn nonmoving_major_keeps_live_nursery_weakref_to_live_old_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_in_oldgen(target_tid, GcHeader::SIZE + 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }
        let mut target_root = target;
        let mut wref_root = wref;
        unsafe {
            gc.roots.add(&mut target_root);
            gc.roots.add(&mut wref_root);
        }

        gc.do_collect_oldgen_nonmoving();

        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert_eq!(after.0, target_root.0);
        assert_eq!(gc.young_objects_with_weakrefs, vec![wref.0]);
        gc.roots.clear();
    }

    /// incminimark.py:3112 "weakref itself not marked" branch.
    /// Dropping the wref root before the major cycle drops the entry
    /// silently — no slot mutation, no panic.
    #[test]
    fn test_major_invalidate_old_weakref_self_dies() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        let mut target_root = target;
        let mut wref_root = wref;
        unsafe {
            gc.roots.add(&mut target_root);
            gc.roots.add(&mut wref_root);
        }
        gc.do_collect_nursery();
        assert_eq!(gc.old_objects_with_weakrefs.len(), 1);

        gc.roots.remove(&mut wref_root);
        gc.do_collect_full();

        assert!(gc.old_objects_with_weakrefs.is_empty());
        gc.roots.clear();
    }

    #[test]
    fn finalizer_queue_keeps_dead_object_until_it_is_popped() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static TRIGGERS: AtomicUsize = AtomicUsize::new(0);
        fn trigger() {
            TRIGGERS.fetch_add(1, Ordering::Relaxed);
        }

        TRIGGERS.store(0, Ordering::Relaxed);
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_in_oldgen(tid, GcHeader::SIZE + 16);
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };
        GcAllocator::register_finalizer(&mut gc, 0, obj, trigger);

        // A live registered object stays on the registration deque.
        gc.do_collect_oldgen_nonmoving();
        assert!(GcAllocator::finalizer_next_dead(&mut gc, 0).is_none());
        assert!(gc.oldgen.contains(obj.0));

        // Once unreachable it moves to the death queue and is kept alive for
        // app-level `__del__` execution.
        gc.roots.remove(&mut root);
        gc.do_collect_oldgen_nonmoving();
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 1);
        assert!(gc.oldgen.contains(obj.0));
        assert_eq!(GcAllocator::finalizer_next_dead(&mut gc, 0), Some(obj));

        // Popping removes the queue root; the following major reclaims it.
        gc.do_collect_oldgen_nonmoving();
        // ArenaCollection::contains is intentionally an arena-range query and
        // can stay true for a freed block while the current arena is retained.
        // The allocator's exact live count verifies reclamation here.
        assert_eq!(gc.oldgen.object_count(), 0);
    }

    /// incminimark.py:2492,2570-2573,3011-3020: objects marked solely to
    /// make a finalizer runnable remain allocated after the sweep, but their
    /// bytes are not survivor growth for the next-major threshold.  Counting
    /// them made a workload of mostly-finalizable garbage raise its threshold
    /// forever (the upstream issue #2590 invariant).
    #[test]
    fn finalizer_kept_graph_is_excluded_from_next_major_threshold() {
        fn trigger() {}

        let mut gc = test_gc(4096);
        let ptr_size = std::mem::size_of::<GcRef>();
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let allocation_size = OldGen::allocation_size(GcHeader::SIZE + ptr_size);

        // Remove the production threshold floor and growth cap so this test
        // observes the survivor baseline itself.  One ordinary rooted object
        // remains real live data; the unreachable finalizer object and its
        // child survive only for finalization ordering.
        gc.min_heap_size = 0.0;
        gc.next_major_collection_initial = 1_000_000_000.0;
        gc.next_major_collection_threshold = 1_000_000_000.0;
        gc.max_delta = f64::MAX;

        let live = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
        let child = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
        let finalizable = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
        unsafe {
            *(live.0 as *mut GcRef) = GcRef::NULL;
            *(child.0 as *mut GcRef) = GcRef::NULL;
            *(finalizable.0 as *mut GcRef) = child;
        }
        let mut live_root = live;
        unsafe { gc.roots.add(&mut live_root) };
        GcAllocator::register_finalizer(&mut gc, 0, finalizable, trigger);

        gc.do_collect_oldgen_nonmoving();

        assert_eq!(gc.kept_alive_by_finalizer, 2 * allocation_size);
        assert_eq!(gc.get_total_memory_used(), 3 * allocation_size);
        let expected = allocation_size as f64 * gc.major_collection_threshold;
        assert_eq!(gc.next_major_collection_threshold, expected);
        assert_eq!(
            GcAllocator::finalizer_next_dead(&mut gc, 0),
            Some(finalizable)
        );
    }

    /// `do_collect_oldgen_nonmoving` is a pyre adaptation: unlike upstream's
    /// major seam it can order a finalizer graph while nursery objects still
    /// exist. Nursery bytes are absent from `get_total_memory_used`, so they
    /// must also be absent from the amount subtracted from that metric.
    #[test]
    fn finalizer_threshold_accounting_excludes_live_nursery_children() {
        fn trigger() {}

        let mut gc = test_gc(4096);
        let ptr_size = std::mem::size_of::<GcRef>();
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let allocation_size = OldGen::allocation_size(GcHeader::SIZE + ptr_size);
        gc.min_heap_size = 0.0;
        gc.next_major_collection_initial = 1_000_000_000.0;
        gc.next_major_collection_threshold = 1_000_000_000.0;
        gc.max_delta = f64::MAX;

        let live = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
        let finalizable = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
        let young_child = gc.alloc_with_type(tid, ptr_size);
        assert!(gc.is_in_nursery(young_child.0));
        unsafe {
            *(live.0 as *mut GcRef) = GcRef::NULL;
            *(young_child.0 as *mut GcRef) = GcRef::NULL;
            *(finalizable.0 as *mut GcRef) = young_child;
        }
        gc.do_write_barrier(finalizable);
        let mut live_root = live;
        unsafe { gc.roots.add(&mut live_root) };
        GcAllocator::register_finalizer(&mut gc, 0, finalizable, trigger);

        gc.do_collect_oldgen_nonmoving();

        assert_eq!(gc.kept_alive_by_finalizer, allocation_size);
        assert_eq!(gc.get_total_memory_used(), 2 * allocation_size);
        let expected = allocation_size as f64 * gc.major_collection_threshold;
        assert_eq!(gc.next_major_collection_threshold, expected);
        assert!(gc.is_in_nursery(young_child.0));
    }

    /// `rgc.py:648-649` contracts `register_finalizer` to run at most once per
    /// object. Registering twice used to leave two deque entries: the first
    /// major delivers the object and re-appends the survivor through
    /// `new_with_finalizer` (incminimark.py:2944-2946), so the next major
    /// delivers it again and the app-level `__del__` runs a second time.
    #[test]
    fn register_finalizer_delivers_an_object_once_however_often_it_is_registered() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static TRIGGERS: AtomicUsize = AtomicUsize::new(0);
        fn trigger() {
            TRIGGERS.fetch_add(1, Ordering::Relaxed);
        }

        TRIGGERS.store(0, Ordering::Relaxed);
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_in_oldgen(tid, GcHeader::SIZE + 16);
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };

        // The construction site registers, then a later out-of-band caller
        // asks again without knowing that.
        GcAllocator::register_finalizer(&mut gc, 0, obj, trigger);
        GcAllocator::register_finalizer(&mut gc, 0, obj, trigger);
        assert_eq!(gc.old_objects_with_finalizers.len(), 1);

        gc.roots.remove(&mut root);
        gc.do_collect_oldgen_nonmoving();
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 1);
        assert_eq!(GcAllocator::finalizer_next_dead(&mut gc, 0), Some(obj));
        assert_eq!(GcAllocator::finalizer_next_dead(&mut gc, 0), None);

        // No survivor was carried into the next cycle, so the object is not
        // delivered a second time.
        gc.do_collect_oldgen_nonmoving();
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 1);
        assert_eq!(GcAllocator::finalizer_next_dead(&mut gc, 0), None);
    }

    // ── rawrefcount (incminimark.py:3157-3409) ──────────────────────

    /// A mirror block lives outside the GC heap at a fixed address, which is
    /// the whole point of the link.  Leaked deliberately: the collector holds
    /// the address in its lists for as long as the test's collector lives.
    fn test_mirror(refcnt: isize) -> usize {
        Box::into_raw(Box::new(rawrefcount::PyObjHeader {
            ob_refcnt: refcnt,
            ob_link: 0,
        })) as usize
    }

    fn mirror_refcnt(pyobject: usize) -> isize {
        unsafe { (*rawrefcount::pyobj(pyobject)).ob_refcnt }
    }

    fn mirror_link(pyobject: usize) -> usize {
        unsafe { (*rawrefcount::pyobj(pyobject)).ob_link }
    }

    thread_local! {
        /// Per-thread so the count belongs to one test: the trigger is a bare
        /// `fn`, and a `static` would be shared with every test the harness
        /// runs concurrently.  The collector calls it on the calling thread.
        static RRC_TRIGGER_FIRED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn rrc_test_trigger() {
        RRC_TRIGGER_FIRED.with(|fired| fired.set(fired.get() + 1));
    }

    fn rrc_test_gc() -> MiniMarkGC {
        let mut gc = test_gc(1 << 16);
        gc.rawrefcount_init(rrc_test_trigger);
        assert!(gc.rawrefcount_enabled());
        gc
    }

    /// incminimark.py:3259-3270 and :3284-3318: a count above the link share is
    /// a reference the C side holds, and it keeps the linked object alive
    /// across a minor with no interpreter root at all.  The link then follows
    /// the object to its new address, and the identity table is re-keyed there.
    #[test]
    fn a_c_referenced_mirror_roots_its_object_across_a_minor() {
        let mut gc = rrc_test_gc();
        let tid = gc.register_type(TypeInfo::object(64));
        let object = gc.alloc_with_type(tid, 64);
        assert!(gc.is_in_nursery(object.0), "premise: the object is young");

        let mirror = test_mirror(rawrefcount::REFCNT_FROM_PYRE + 1);
        gc.rawrefcount_create_link_pyre(object.0, mirror);
        assert_eq!(gc.rawrefcount_from_obj(object.0), mirror);

        gc.do_collect_nursery();

        let moved = mirror_link(mirror);
        assert_ne!(moved, 0, "the mirror was unlinked");
        assert_ne!(moved, object.0, "premise: a surviving nursery object moves");
        assert!(!gc.is_in_nursery(moved));
        assert_eq!(gc.rawrefcount_from_obj(moved), mirror);
        assert_eq!(gc.rawrefcount_to_obj(mirror), moved);
        assert_eq!(gc.rawrefcount_next_dead(), 0, "nothing died");
    }

    /// incminimark.py:3264-3265 and :3320-3354: a count of exactly the link
    /// share means nothing but the link references the mirror, so the object
    /// may die.  The mirror is then unlinked, queued, and left at 1 so the
    /// drain's own release is what frees it.
    #[test]
    fn a_mirror_at_the_link_share_lets_its_object_die_and_is_queued() {
        let mut gc = rrc_test_gc();
        let tid = gc.register_type(TypeInfo::object(64));
        let object = gc.alloc_with_type(tid, 64);
        let mirror = test_mirror(rawrefcount::REFCNT_FROM_PYRE);
        gc.rawrefcount_create_link_pyre(object.0, mirror);

        RRC_TRIGGER_FIRED.with(|fired| fired.set(0));
        gc.do_collect_nursery();

        assert_eq!(mirror_link(mirror), 0, "a dead mirror keeps no link");
        assert_eq!(
            mirror_refcnt(mirror),
            1,
            "incminimark.py:3352 leaves a queued mirror at 1"
        );
        assert_eq!(gc.rawrefcount_next_dead(), mirror);
        assert_eq!(gc.rawrefcount_next_dead(), 0, "the queue drains once");
        assert_eq!(
            RRC_TRIGGER_FIRED.with(|fired| fired.get()),
            1,
            "incminimark.py:3248-3250 schedules the drain for a non-empty queue"
        );
    }

    /// incminimark.py:3359-3375 and :3397-3409: the same two outcomes, decided
    /// by a major collection over the old list.
    #[test]
    fn a_major_frees_only_the_mirrors_whose_object_died() {
        let mut gc = rrc_test_gc();
        let tid = gc.register_type(TypeInfo::object(64));

        let mut kept = gc.alloc_in_oldgen(tid, GcHeader::SIZE + 64);
        let dying = gc.alloc_in_oldgen(tid, GcHeader::SIZE + 64);
        assert!(!gc.is_in_nursery(kept.0) && !gc.is_in_nursery(dying.0));

        let kept_mirror = test_mirror(rawrefcount::REFCNT_FROM_PYRE + 1);
        let dying_mirror = test_mirror(rawrefcount::REFCNT_FROM_PYRE);
        gc.rawrefcount_create_link_pyre(kept.0, kept_mirror);
        gc.rawrefcount_create_link_pyre(dying.0, dying_mirror);
        unsafe { gc.roots.add(&mut kept) };

        gc.do_collect_full();

        assert_eq!(mirror_link(kept_mirror), kept.0);
        assert_eq!(gc.rawrefcount_from_obj(kept.0), kept_mirror);
        assert_eq!(mirror_link(dying_mirror), 0);
        assert_eq!(mirror_refcnt(dying_mirror), 1);
        assert_eq!(gc.rawrefcount_next_dead(), dying_mirror);
        assert_eq!(gc.rawrefcount_next_dead(), 0);

        gc.roots.remove(&mut kept);
    }

    /// A pinned object survives a minor without moving, so there is no
    /// forwarding pointer for its mirror's link to follow — and reading only
    /// the forwarding bit, as incminimark.py:3287-3299 does, would report it
    /// dead and free a mirror the C side still holds.
    #[test]
    fn a_pinned_object_keeps_its_mirror_across_a_minor() {
        let mut gc = rrc_test_gc();
        // `pin` refuses anything carrying GC pointers, so this is the shape a
        // pinnable object has: a leaf byte block.
        let tid = gc.register_type(TypeInfo::simple(64));
        let mut object = gc.alloc_with_type(tid, 64);
        assert!(gc.pin(object), "premise: this object can be pinned");
        // Pinning does not root: something must still reach it this collection.
        unsafe { gc.roots.add(&mut object) };

        let mirror = test_mirror(rawrefcount::REFCNT_FROM_PYRE + 1);
        gc.rawrefcount_create_link_pyre(object.0, mirror);

        gc.do_collect_nursery();

        assert_eq!(
            mirror_link(mirror),
            object.0,
            "a pinned object does not move, so the link must not change"
        );
        assert!(gc.is_in_nursery(mirror_link(mirror)));
        assert_eq!(gc.rawrefcount_from_obj(object.0), mirror);
        assert_eq!(gc.rawrefcount_next_dead(), 0, "the mirror was freed");

        // Still tracked as young, so the next minor finds it again.
        gc.do_collect_nursery();
        assert_eq!(gc.rawrefcount_from_obj(object.0), mirror);
        assert_eq!(gc.rawrefcount_next_dead(), 0);

        gc.roots.remove(&mut object);
    }

    /// incminimark.py:3223-3227: the deallocating marker replaces the link, so
    /// a re-entrant lookup during a deallocator cannot hand back an address the
    /// collector has already reclaimed.
    #[test]
    fn mark_deallocating_replaces_the_link_with_the_sentinel() {
        let mut gc = rrc_test_gc();
        let tid = gc.register_type(TypeInfo::object(64));
        let object = gc.alloc_in_oldgen(tid, GcHeader::SIZE + 64);
        let mirror = test_mirror(rawrefcount::REFCNT_FROM_PYRE + 1);
        gc.rawrefcount_create_link_pyre(object.0, mirror);

        let marker = 0x0DEA_DFFF_usize;
        gc.rawrefcount_mark_deallocating(marker, mirror);
        assert_eq!(gc.rawrefcount_to_obj(mirror), marker);
    }

    /// A non-moving major runs no leading minor, so the young P list is the
    /// only thing that can report a C-referenced nursery object — and an old
    /// object reachable only through one would otherwise be swept.
    ///
    /// Differential, because "is this address still live?" has no sound direct
    /// oracle once the arena has been swept: the same heap is built twice and
    /// differs only in whether the mirror carries a C reference, so the
    /// surviving bytes are attributable to nothing else.
    #[test]
    fn a_nonmoving_major_keeps_an_old_object_reachable_only_through_a_c_root() {
        fn survivors_with(mirror_refcnt: isize) -> usize {
            let word = std::mem::size_of::<usize>();
            let mut gc = rrc_test_gc();
            let leaf = gc.register_type(TypeInfo::object(word));
            let holder = gc.register_type(TypeInfo::object_with_gc_ptrs(word, vec![0]));

            let old = gc.alloc_in_oldgen(leaf, GcHeader::SIZE + word);
            let young = gc.alloc_with_type(holder, word);
            assert!(gc.is_in_nursery(young.0) && !gc.is_in_nursery(old.0));
            // The old object is reachable from the nursery object and from
            // nowhere else; the nursery object is reachable from the mirror
            // and from nowhere else.
            unsafe { *(young.0 as *mut GcRef) = old };
            gc.rawrefcount_create_link_pyre(young.0, test_mirror(mirror_refcnt));

            gc.do_collect_oldgen_nonmoving();
            assert_eq!(gc.rawrefcount_next_dead(), 0, "the nursery is not swept");
            gc.get_total_memory_used()
        }

        let referenced = survivors_with(rawrefcount::REFCNT_FROM_PYRE + 1);
        let unreferenced = survivors_with(rawrefcount::REFCNT_FROM_PYRE);
        assert!(
            referenced > unreferenced,
            "the old object was swept out from under a live C reference \
             ({referenced} bytes held with a C reference, {unreferenced} without)"
        );
    }
}
