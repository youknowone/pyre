//! `rpython/rlib/debug.py` parity — PYPYLOG-style debug scope and print
//! API shared across the metainterp, optimizer, and backends.
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
//! Single-event sites — the common case in Pyre's metainterp/optimizeopt/
//! backends — use [`log_one`], which opens a single-line section,
//! emits the body, and closes the section in one call.  Multi-message
//! pairs use [`scope`] (RAII) wrapping repeated [`debug_print`] calls.
//!
//! # Known divergences from `rpython/rlib/debug.py` / PYPYLOG
//!
//! The wire format matches, but four behaviours are deliberately
//! reduced versions of upstream — each marked here so callers know
//! what to expect:
//!
//! 1. **No category-prefix filter.**  PyPy's
//!    `PYPYLOG=jit-tracing,jit-backend:filename` parses a comma list of
//!    accepted category prefixes; only matching `debug_start` sections
//!    are emitted.  Pyre's [`majit_log_enabled`] is a single all-or-
//!    nothing switch — any value of `MAJIT_LOG` turns *every* category
//!    on.  [`have_debug_prints_for`] preserves the upstream signature
//!    so a future category parser can drop in without touching
//!    callers.
//!
//! 2. **No translated/untranslated split.**  RPython's
//!    `_log_capture` versus `_log` distinction (`rlib/debug.py:24-30`,
//!    `60-67`) lets untranslated tests assert against captured
//!    sections without writing to stderr.  Pyre always writes to
//!    `stderr` via `eprintln!` — tests that need to assert on log
//!    output redirect `stderr` at the OS level.
//!
//! 3. **Strict `debug_stop` nesting.**  RPython's
//!    `DebugLog.debug_stop` (`rpython/rlib/debug.py:30`) raises on
//!    mismatch; Pyre [`debug_stop`] panics with the same intent.  This
//!    is intentional, but it does mean a mid-stack panic propagates
//!    through any `debug_start`/`debug_stop` pair that was already
//!    opened (the RAII [`scope`] guard absorbs this by closing in
//!    `Drop`; bare `debug_start`/`debug_stop` callers must use
//!    `try/finally`-equivalent unwind discipline).
//!
//! 4. **Single-file output, no `:filename` sink.**  PyPy's
//!    `PYPYLOG=…:my.log` redirects to a file; pyre always uses
//!    `stderr`.  External tools that consume PYPYLOG-formatted output
//!    can capture pyre's `stderr` directly — the wire format is
//!    identical.
//!
//! The first three are revisit candidates if pyre gains category-
//! scoped log filtering or a per-test capture sink.  Until then
//! `MAJIT_LOG=1` is the only knob.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

/// Whether `MAJIT_LOG` is set, cached at first access.  Mirrors PyPy's
/// `PYPYLOG` env-var check (`rpython/rlib/debug.py:31-38`).
pub fn majit_log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("MAJIT_LOG").is_some())
}

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
/// is enabled at all.  Pyre keys this off `MAJIT_LOG`.
pub fn have_debug_prints() -> bool {
    majit_log_enabled()
}

/// `rlib/debug.py:168-172 have_debug_prints_for(prefix)` — true when
/// the active log filter accepts `prefix` (RPython queries the
/// `PYPYLOG=cat1,cat2:filename` filter spec, not the currently-open
/// `debug_start` stack).  Pyre has no category-level filter — `MAJIT_LOG`
/// is all-or-nothing — so this reduces to [`have_debug_prints`].  The
/// `_prefix` parameter is preserved so callers ported 1:1 from PyPy keep
/// compiling unchanged once a filter mechanism lands.
pub fn have_debug_prints_for(_prefix: &str) -> bool {
    have_debug_prints()
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
/// pops the stack top.  Mismatched stops panic, mirroring RPython's
/// `DebugLog.debug_stop` (`rpython/rlib/debug.py:30`), which raises a
/// nesting error so unbalanced calls surface immediately instead of
/// being absorbed.
pub fn debug_stop(category: &'static str) {
    if !have_debug_prints() {
        return;
    }
    CATEGORY_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        match s.last() {
            Some(top) if *top == category => {
                s.pop();
            }
            Some(top) => panic!(
                "debug_stop({category:?}) does not match the most recent debug_start({top:?})"
            ),
            None => panic!("debug_stop({category:?}) with no matching debug_start"),
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

/// Emit a single body line wrapped in a `debug_start`/`debug_stop`
/// section of the given category.  Equivalent to PyPy's common pattern
///
/// ```python
/// debug_start(cat); debug_print(msg); debug_stop(cat)
/// ```
///
/// Used for one-shot events (aborts, single-state-transition notices)
/// that don't have a natural surrounding scope on the Pyre side.  No-op
/// when [`have_debug_prints`] is false — callers may still incur the
/// cost of building `msg`; gate that with [`have_debug_prints`] for
/// hot paths where formatting is non-trivial.
pub fn log_one(category: &'static str, msg: &str) {
    if !have_debug_prints() {
        return;
    }
    debug_start(category);
    debug_print(msg);
    debug_stop(category);
}
