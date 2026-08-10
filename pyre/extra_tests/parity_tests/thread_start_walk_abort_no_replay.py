# CPython-suite gap: threading tests cannot trigger pyre's inline-walk abort replay.
# parity-tests reason: this is a JIT-only side-effect/resume regression.

"""A walk that aborts after executing a side-effecting residual must not replay it.

An authoritative full-body walk executes residual calls concretely as it goes.
When it then aborts on a kept-stack branch guard it cannot compile, the only
sound continuations resume FORWARD (`pyjitpl.py:2949
run_blackhole_interp_to_cancel_tracing`); handing the frame back at its entry
re-runs everything the walk already applied.  `threading.Thread.start` is where
that is observable rather than merely wasteful: the walk runs
`_start_joinable_thread`, which spawns the OS thread, and a replay calls it a
second time on a handle that is already started.

The three faces of that single replay, depending on how far the freshly spawned
worker got before the interpreter re-entered `start`:

  - the worker had already run `_started.set()` -> the fresh `Event` this object
    built in `__init__`, still unpublished, reads set,
  - it had not -> `RuntimeError: thread already started`,
  - with the stock `Thread.start`, whose own guard sees the set Event ->
    `RuntimeError: threads can only be started once`.

⚠️ This file is a REPRODUCER, not a tidy test, and the difference matters.  The
walk only reaches the aborting guard for a narrow traced shape: `start` spelled
out instead of delegating, the three Event reads kept, `Event` bound as a global
rather than reached through `threading.`, the two locals read inside the final
`try`, and failures leaving through `die` (a plain call) rather than `raise`,
whose exception edge changes the guard.  Every one of those was measured — each
simplification made the file pass against a build WITHOUT the fix, i.e. cover
nothing.  Re-measure against such a build before simplifying anything here.

Every acquire on the main thread is bounded so a lost release reports instead of
hanging, and `die` uses `os._exit` so a failure cannot block on shutdown.

⚠️ The `parity-env` line below is what keeps this file covering anything.  All
three kept-stack decline hazards are scoped to `!ctx.vstack_valid`, so once the
callee operand-stack mirror became the default an inline sub-walk describes its
own stack and the aborting guard is never reached.  Measured on a binary WITHOUT
the fix: 4/4 pass at the default setting, 4/4 fail with the mirror off (the same
deterministic round 97 either way).  Drop the pin and this test goes green
against the very defect it exists to catch.
"""

# parity-env: PYRE_FBW_CALLEE_VSTACK=0

import contextvars
import os
import threading
from threading import Event

N = 20
R = 150


def die(msg):
    os.write(2, msg.encode())
    os._exit(1)


class T(threading.Thread):
    def __init__(self, *a, **kw):
        threading.Thread.__init__(self, *a, **kw)
        self._ev_ref = self._started
        if self._started._flag:
            die("BORN TRUE: fresh Event already set at __init__ id=%d\n"
                % id(self._started))

    def start(self):
        pre = self._started
        if pre is not self._ev_ref:
            die("STALE LOAD_ATTR self._started: got id=%d %r, stored id=%d %r\n"
                % (id(pre), pre, id(self._ev_ref), self._ev_ref))
        if not isinstance(pre, Event):
            die("WRONG SLOT (pre) %r %s\n" % (pre, type(pre)))
        f_call = pre.is_set()
        f_attr = pre._flag
        f_call2 = pre.is_set()
        if not (f_call is f_attr is f_call2):
            die("DISAGREE call=%r attr=%r call2=%r ev_id=%d\n"
                % (f_call, f_attr, f_call2, id(pre)))
        if f_call:
            die("FLIPPED AFTER INIT: a fresh Event reads set before start() "
                "published it — start() ran twice. ev_id=%d thread_id=%d\n"
                % (id(pre), id(self)))
        if not threading._active_limbo_lock.acquire(True, 5.0):
            die("BLOCKED on _active_limbo_lock\n")
        try:
            threading._limbo[self] = self
        finally:
            threading._active_limbo_lock.release()
        if self._context is None:
            self._context = contextvars.Context()
        threading._start_joinable_thread(
            self._bootstrap, handle=self._os_thread_handle, daemon=self.daemon)
        post = self._started
        if not isinstance(post, Event):
            die("WRONG SLOT (post) %r %s same=%s\n" % (post, type(post), post is pre))
        # Event.wait(), with the outer Condition acquire bounded.
        cond = post._cond
        if not cond._lock.acquire(True, 5.0):
            die("LOST RELEASE on _started._cond._lock: flag=%r alive=%r waiters=%d\n"
                % (post._flag, self.is_alive(), len(cond._waiters)))
        try:
            alive_before = self.is_alive()
            w_before = len(cond._waiters)
            if not post._flag and not cond.wait(5.0):
                die("NO SIGNAL | check: alive=%r waiters=%d | now: flag=%r "
                    "alive=%r waiters=%d\n"
                    % (alive_before, w_before, post._flag, self.is_alive(),
                       len(cond._waiters)))
        finally:
            cond._lock.release()


for r in range(R):
    ts = [T(target=lambda: None) for _ in range(N)]
    try:
        for t in ts:
            t.start()
    except RuntimeError as e:
        die("START RAISED %r in round %d — start() ran twice on one handle\n"
            % (e, r))
    for i, t in enumerate(ts):
        t.join(5.0)
        if t.is_alive():
            die("JOIN TIMEOUT round=%d idx=%d\n" % (r, i))

print("OK")
