//! `rpython/rlib/debug.py` parity — PYPYLOG-style debug scope and print
//! API for the metainterp.
//!
//! PyPy structures runtime tracing through `debug_start(category)` /
//! `debug_stop(category)` brackets with intervening `debug_print(...)`
//! lines; the wire format is
//!
//! ```text
//! [<ts>] {<category>
//! <messages>
//! [<ts>] <category>}
//! ```
//!
//! Tooling (`pypy/tool/logparser.py`) parses this format directly.  Pyre
//! emits the same wire shape so log captures cross-tool with PyPy when
//! the `MAJIT_LOG` env var is set (pyre's `PYPYLOG=…:-` analog).
//!
//! Conversion is intentionally incremental: this module is the API and
//! the structurally-explicit `debug_start/stop` pairs in `memmgr` use
//! it today.  The bulk of inline `eprintln!("[<cat>] …")` sites — 70+
//! across metainterp/optimizeopt/backends — retain prefix-style output
//! for backward compatibility and will be migrated in a follow-up pass.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

/// Wall-clock origin used as the PyPy `read_timestamp()` analog.
fn ts_origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

/// `rlib/rtimer.py read_timestamp()` analog — monotonic nanosecond
/// counter rendered as PyPy's hex `[ts]` prefix.
fn read_timestamp() -> u128 {
    ts_origin().elapsed().as_nanos()
}

thread_local! {
    /// Per-thread category stack mirroring PyPy's `_log` debug log
    /// (`rlib/debug.py:24-30`).  Push on `debug_start`, pop on
    /// `debug_stop`.  The stack is only consulted by `have_debug_prints_for`;
    /// the on-wire output works without it.
    static CATEGORY_STACK: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// `rlib/debug.py:163-166 have_debug_prints()` — true when log output
/// is enabled at all.  Pyre keys this off `MAJIT_LOG` (cached on first
/// read, matching `crate::majit_log_enabled`).
pub fn have_debug_prints() -> bool {
    crate::majit_log_enabled()
}

/// `rlib/debug.py:168-172 have_debug_prints_for(prefix)` — true when at
/// least one active category in the current `debug_start` stack starts
/// with `prefix`, gated by [`have_debug_prints`].  PyPy uses this to
/// short-circuit message construction when a section is not active.
pub fn have_debug_prints_for(prefix: &str) -> bool {
    if !have_debug_prints() {
        return false;
    }
    CATEGORY_STACK.with(|stack| stack.borrow().iter().any(|cat| cat.starts_with(prefix)))
}

/// `rlib/debug.py:101-108 debug_start(category)` — open a logging
/// section.  Emits `[<ts>] {<category>` on stderr when the log is
/// enabled and pushes `category` onto the thread-local stack.  No-op
/// when [`have_debug_prints`] is false (the stack is also untouched,
/// matching PyPy where `_log` is None).
pub fn debug_start(category: &'static str) {
    if !have_debug_prints() {
        return;
    }
    eprintln!("[{:x}] {{{}", read_timestamp(), category);
    CATEGORY_STACK.with(|stack| stack.borrow_mut().push(category));
}

/// `rlib/debug.py:111-116 debug_stop(category)` — close the matching
/// section opened by [`debug_start`].  Emits `[<ts>] <category>}` and
/// pops the stack top.  Mismatched stops are silently ignored to mirror
/// PyPy's `DebugLog.debug_stop` (which scans backwards for the most
/// recent matching start and is forgiving on unbalanced calls).
pub fn debug_stop(category: &'static str) {
    if !have_debug_prints() {
        return;
    }
    CATEGORY_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        if s.last() == Some(&category) {
            s.pop();
        }
    });
    eprintln!("[{:x}] {}}}", read_timestamp(), category);
}

/// `rlib/debug.py:69-74 debug_print(*args)` — emit a single line inside
/// the currently-open section.  No-op when the log is disabled.  Pyre
/// callers format the message themselves and pass the result here.
pub fn debug_print(msg: &str) {
    if !have_debug_prints() {
        return;
    }
    eprintln!("{}", msg);
}

/// RAII scope guard returned by [`scope`]: panics still drop through
/// `Drop` so `debug_stop` always pairs with the opening `debug_start`.
#[must_use = "drop the guard to fire the matching debug_stop"]
pub struct DebugScope {
    category: &'static str,
}

impl Drop for DebugScope {
    fn drop(&mut self) {
        debug_stop(self.category);
    }
}

/// Convenience: open a `debug_start(category)` scope returning a guard
/// that fires the matching `debug_stop` on drop.  Mirrors PyPy's
/// typical `debug_start … try: … finally: debug_stop` pattern.
pub fn scope(category: &'static str) -> DebugScope {
    debug_start(category);
    DebugScope { category }
}
