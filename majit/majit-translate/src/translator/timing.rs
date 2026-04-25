//! Port of `rpython/translator/timing.py` — 52-LOC accumulator that
//! `translator.driver.TranslationDriver._do` (`driver.py:271-282`)
//! wraps around every task to record per-goal elapsed time.
//!
//! Upstream file structure (line-by-line):
//!
//! ```python
//! class Timer(object):
//!     def __init__(self, timer=time.time):
//!         self.events = []
//!         self.next_even = None
//!         self.timer = timer
//!         self.t0 = None
//!
//!     def start_event(self, event):
//!         now = self.timer()
//!         if self.t0 is None:
//!             self.t0 = now
//!         self.next_event = event
//!         self.start_time = now
//!
//!     def end_event(self, event):
//!         assert self.next_event == event
//!         now = self.timer()
//!         self.events.append((event, now - self.start_time))
//!         self.next_event = None
//!         self.tk = now
//!
//!     def ttime(self):
//!         try:
//!             return self.tk - self.t0
//!         except AttributeError:
//!             return 0.0
//!
//!     def pprint(self):
//!         ...
//! ```
//!
//! Two adaptations apply:
//!
//! 1. **Logger substitution** (bounded PRE-EXISTING-ADAPTATION).
//!    Upstream uses `rpython.tool.ansi_print.AnsiLogger("Timer")` for
//!    the coloured `pprint` output. Our port has no ansi-logger port
//!    and the driver layer itself is still being stood up, so
//!    `pprint` emits plain stdout lines via `println!`. The format
//!    strings are preserved verbatim so the observable text matches
//!    upstream (modulo ANSI colour codes). *Convergence path*: when
//!    `tool::ansi_print` ports, swap `println!` for the logger call
//!    at `driver.py:250 log.info(msg)` / `Timer.pprint`'s
//!    `log.bold(...)`.
//!
//! 2. **Injectable clock** (unavoidable Rust-language adaptation).
//!    Upstream passes `timer=time.time` as a constructor kwarg so
//!    tests can inject a deterministic clock. The Rust port expresses
//!    this through the `TimeSource` trait — default implementation
//!    reads `std::time::Instant::now()`; tests plug in a counter
//!    `TimeSource` to avoid flakiness. The surface matches upstream
//!    (caller can pick the clock source at construction time).

use std::cell::{Cell, RefCell};
use std::time::Instant;

/// Pluggable clock source, matching upstream's `timer=time.time`
/// constructor arg at `timing.py:12`. Returns a monotonically non-
/// decreasing `f64` number of seconds since an epoch of the source's
/// choosing (only differences are observable).
pub trait TimeSource {
    fn now(&self) -> f64;
}

/// Default `TimeSource` backed by `std::time::Instant`. Matches
/// upstream's `time.time` in the sense that both produce
/// monotonically-non-decreasing reals; the absolute epoch differs
/// but `Timer` only observes differences so the shift is invisible.
pub struct SystemClock {
    epoch: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        SystemClock {
            epoch: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        SystemClock::new()
    }
}

impl TimeSource for SystemClock {
    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }
}

/// Port of `rpython/translator/timing.py:11-50 Timer`.
///
/// Upstream's state layout:
/// - `self.events: list[(name, elapsed)]` — per-event timings.
/// - `self.next_event: Optional[str]` — name of the currently-open
///   event (None while idle; matches upstream's typo
///   `self.next_even` on line 15 that is immediately shadowed by
///   `next_event` assignments on lines 22/29).
/// - `self.timer: Callable[[], float]` — injected clock.
/// - `self.t0: Optional[float]` — timestamp of the first
///   `start_event` call.
/// - `self.start_time: float` — timestamp of the most recent
///   `start_event`.
/// - `self.tk: float` — timestamp of the most recent `end_event`
///   (absent until the first `end_event` fires).
///
/// Rust mirrors the same shape with `Cell`/`RefCell` for interior
/// mutability (upstream mutates via `self.foo = x` under `&self`-
/// equivalent bound-method semantics).
pub struct Timer<T: TimeSource> {
    /// Upstream `self.events`.
    events: RefCell<Vec<(String, f64)>>,
    /// Upstream `self.next_event`. `None` ↔ upstream `None` (no open
    /// event); `Some(name)` ↔ an open event awaiting `end_event`.
    next_event: RefCell<Option<String>>,
    /// Upstream `self.timer`. Held by value — upstream passes the
    /// clock in on construction and never replaces it.
    timer: T,
    /// Upstream `self.t0`. `None` until the first `start_event`
    /// captures it.
    t0: Cell<Option<f64>>,
    /// Upstream `self.start_time`. Mirrored as `Option` because
    /// upstream only reads it inside `end_event` after `start_event`
    /// wrote it — the Rust port makes the sequencing explicit.
    start_time: Cell<Option<f64>>,
    /// Upstream `self.tk`. Absent until the first `end_event`.
    tk: Cell<Option<f64>>,
}

impl Timer<SystemClock> {
    /// Convenience constructor matching upstream's default
    /// `Timer()` signature at `timing.py:12`. Uses `SystemClock` so
    /// callers that do not care about the clock source get the
    /// upstream-equivalent behaviour with zero boilerplate.
    pub fn new() -> Self {
        Timer::with_source(SystemClock::new())
    }
}

impl Default for Timer<SystemClock> {
    fn default() -> Self {
        Timer::new()
    }
}

impl<T: TimeSource> Timer<T> {
    /// Upstream `Timer.__init__(self, timer=time.time)` at
    /// `timing.py:12-16`. Accepts an injected clock so downstream
    /// tests can assert on deterministic elapsed values.
    pub fn with_source(timer: T) -> Self {
        Timer {
            events: RefCell::new(Vec::new()),
            next_event: RefCell::new(None),
            timer,
            t0: Cell::new(None),
            start_time: Cell::new(None),
            tk: Cell::new(None),
        }
    }

    /// Upstream `Timer.start_event(self, event)` at `timing.py:18-23`.
    /// Records the starting timestamp; `self.t0` captures the first
    /// `start_event`'s timestamp and is never reset afterwards
    /// (enabling `ttime()`'s "total wall-clock since first start"
    /// semantics).
    pub fn start_event(&self, event: impl Into<String>) {
        let now = self.timer.now();
        if self.t0.get().is_none() {
            self.t0.set(Some(now));
        }
        *self.next_event.borrow_mut() = Some(event.into());
        self.start_time.set(Some(now));
    }

    /// Upstream `Timer.end_event(self, event)` at `timing.py:25-30`.
    /// Upstream uses a plain `assert self.next_event == event`, which
    /// panics in both `-O` and non-`-O` Python execution by default
    /// (only `python -OO` strips asserts, and PyPy translation never
    /// runs that way). Use `assert_eq!` so the Rust port matches
    /// upstream's release-also panic semantics; downgrading to
    /// `debug_assert_eq!` would silently accept a mismatched
    /// `start_event`/`end_event` pair in release builds.
    pub fn end_event(&self, event: &str) {
        let now = self.timer.now();
        {
            let open = self.next_event.borrow();
            assert_eq!(
                open.as_deref(),
                Some(event),
                "Timer::end_event: expected {:?}, got {:?}",
                open.as_deref(),
                event
            );
        }
        let elapsed = self
            .start_time
            .get()
            .map(|start| now - start)
            .unwrap_or(0.0);
        self.events.borrow_mut().push((event.to_string(), elapsed));
        *self.next_event.borrow_mut() = None;
        self.tk.set(Some(now));
    }

    /// Upstream `Timer.ttime(self)` at `timing.py:32-36`. Returns the
    /// wall-clock span between the first `start_event` and the most
    /// recent `end_event`; `0.0` when either bookend is missing
    /// (upstream's `except AttributeError: return 0.0` path).
    pub fn ttime(&self) -> f64 {
        match (self.t0.get(), self.tk.get()) {
            (Some(t0), Some(tk)) => tk - t0,
            _ => 0.0,
        }
    }

    /// Upstream `Timer.pprint(self)` at `timing.py:38-51`. Emits the
    /// per-event table + total row; ANSI colour bolding from
    /// `rpython.tool.ansi_print.AnsiLogger` is dropped — see the
    /// module docstring for the convergence path.
    pub fn pprint(&self) {
        let total_line = {
            let spacing = " ".repeat(30 - "Total:".len());
            format!("Total:{spacing} --- {:.1} s", self.ttime())
        };
        println!("Timings:");
        for (event, time) in self.events.borrow().iter() {
            let spacing = " ".repeat(30 - event.len());
            let first = format!("{event}{spacing} --- ");
            let second = format!("{time:.1} s");
            let additional_spaces =
                " ".repeat(total_line.len().saturating_sub(first.len() + second.len()));
            println!("{first}{additional_spaces}{second}");
        }
        println!("{}", "=".repeat(total_line.len()));
        println!("{total_line}");
    }

    /// Readonly accessor for the recorded events. Upstream exposes
    /// `self.events` directly on the Python object; Rust prefers an
    /// explicit getter to keep the field private-by-default.
    pub fn events(&self) -> std::cell::Ref<'_, Vec<(String, f64)>> {
        self.events.borrow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic clock for tests — returns a monotonically
    /// increasing counter set via `advance`.
    struct MockClock {
        ticks: Cell<f64>,
    }

    impl MockClock {
        fn new() -> Self {
            MockClock {
                ticks: Cell::new(0.0),
            }
        }
        fn advance(&self, by: f64) {
            self.ticks.set(self.ticks.get() + by);
        }
    }

    impl TimeSource for &MockClock {
        fn now(&self) -> f64 {
            self.ticks.get()
        }
    }

    #[test]
    fn new_timer_has_zero_ttime_until_start_end_complete() {
        // Upstream `timing.py:32-36`: `ttime()` returns 0.0 when
        // `self.tk - self.t0` would raise `AttributeError` (either
        // `t0` or `tk` unset).
        let t = Timer::new();
        assert_eq!(t.ttime(), 0.0);
    }

    #[test]
    fn start_then_end_records_event_elapsed_and_updates_ttime() {
        // Upstream `timing.py:18-30` — start_event captures `t0` and
        // `start_time`; end_event appends `(event, now - start_time)`
        // and records `tk`.
        let clock = MockClock::new();
        let timer = Timer::with_source(&clock);
        timer.start_event("annotate");
        clock.advance(0.7);
        timer.end_event("annotate");

        let evs = timer.events();
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].0, "annotate");
        assert!((evs[0].1 - 0.7).abs() < 1e-9, "elapsed={}", evs[0].1);
        // ttime = tk - t0 = 0.7 - 0 = 0.7.
        assert!(
            (timer.ttime() - 0.7).abs() < 1e-9,
            "ttime={}",
            timer.ttime()
        );
    }

    #[test]
    fn multiple_events_accumulate_in_order() {
        // Upstream stores events in append order; ttime spans first
        // t0 → last tk.
        let clock = MockClock::new();
        let timer = Timer::with_source(&clock);
        timer.start_event("annotate");
        clock.advance(0.3);
        timer.end_event("annotate");
        clock.advance(0.1);
        timer.start_event("rtype_lltype");
        clock.advance(0.5);
        timer.end_event("rtype_lltype");

        let evs = timer.events();
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].0, "annotate");
        assert!((evs[0].1 - 0.3).abs() < 1e-9);
        assert_eq!(evs[1].0, "rtype_lltype");
        assert!((evs[1].1 - 0.5).abs() < 1e-9);
        // t0 captured at first start (0.0); last end at 0.9.
        assert!((timer.ttime() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn t0_captured_only_once_on_first_start_event() {
        // Upstream `timing.py:21-22`: `if self.t0 is None: self.t0 =
        // now`. Subsequent start_event calls do NOT overwrite t0.
        let clock = MockClock::new();
        let timer = Timer::with_source(&clock);
        clock.advance(5.0);
        timer.start_event("first");
        clock.advance(1.0);
        timer.end_event("first");

        clock.advance(10.0);
        timer.start_event("second");
        clock.advance(2.0);
        timer.end_event("second");

        // t0 = 5.0, last tk = 18.0 → ttime = 13.0.
        assert!(
            (timer.ttime() - 13.0).abs() < 1e-9,
            "ttime={}",
            timer.ttime()
        );
    }

    #[test]
    #[should_panic(expected = "expected")]
    fn end_event_mismatched_name_asserts_in_debug_only() {
        // Upstream `timing.py:26`: `assert self.next_event == event`.
        // Debug-only assertion matches upstream's `assert`.
        let clock = MockClock::new();
        let timer = Timer::with_source(&clock);
        timer.start_event("open");
        timer.end_event("different");
    }

    #[test]
    fn end_event_clears_next_event_marker() {
        // Upstream `timing.py:29`: `self.next_event = None`.
        let clock = MockClock::new();
        let timer = Timer::with_source(&clock);
        timer.start_event("e");
        assert_eq!(timer.next_event.borrow().as_deref(), Some("e"),);
        clock.advance(0.1);
        timer.end_event("e");
        assert!(timer.next_event.borrow().is_none());
    }
}
