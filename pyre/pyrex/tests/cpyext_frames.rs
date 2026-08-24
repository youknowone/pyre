//! The frame an extension builds for itself, and the traceback entry it makes
//! out of one.
//!
//! The expectations were taken from CPython 3.14 running the same sequence
//! against the same entry points, with two exceptions the fixture states where
//! they arise: CPython's frame is opaque, so nothing there can read the fields
//! back or write `f_lineno`; and `PyTraceBack_Here` with no exception being
//! propagated crashes CPython, where this runtime answers `-1` as PyPy does.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import sys

import cpyext_frames as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


# ── a frame the extension made ─────────────────────────────────────────

# CPython answers `(1, <class 'frame'>)` for the same two.
eq('a made frame is a frame', m.check_new_frame(), (1, type(sys._getframe())))

code_is_kept, lineno, globals_seen, locals_seen = m.describe_new_frame('made.pyx', 11)
eq('the code is the one passed', code_is_kept, True)
# Written through the block after `PyFrame_New` returned, which is what an
# extension reporting a traceback of its own does.
eq('the line written is the line read', lineno, 11)
eq('the globals are the ones passed', globals_seen is m.__dict__, True)
# `PyFrame_New` was passed NULL for them.
eq('no locals', locals_seen, None)

# ── the traceback entry it makes ───────────────────────────────────────

raised = None
try:
    m.add_traceback('cy_func', 'hello.pyx', 42)
except ValueError as error:
    raised = error
if raised is None:
    raise AssertionError('add_traceback did not raise')

entries = []
tb = raised.__traceback__
while tb:
    entries.append((tb.tb_frame.f_code.co_filename,
                    tb.tb_frame.f_code.co_name,
                    tb.tb_lineno))
    tb = tb.tb_next

eq('the exception is the one set', str(raised), 'boom')
# Two entries: this script's own call, then the one the extension recorded.
eq('the recorded entry', entries[-1], ('hello.pyx', 'cy_func', 42))
eq('this frame is still first', entries[0][1], '<module>')

# The recorded frame reports the line the extension wrote.
tb = raised.__traceback__
while tb.tb_next:
    tb = tb.tb_next
eq('the recorded frame line', tb.tb_frame.f_lineno, 42)
eq('the recorded frame name', tb.tb_frame.f_code.co_name, 'cy_func')

# ── a frame this runtime is executing ──────────────────────────────────

# The other direction: the mirror is filled from the interpreter side, so an
# extension handed a running frame reads the same fields it would write.
def outer():
    return inner()


def inner():
    # One line, so the line the mirror snapshots is the line read beside it:
    # the mirror is filled once, when C first asks for it, and does not follow
    # the frame afterwards.
    frame = sys._getframe(); return frame, frame.f_lineno, m.describe_running_frame(frame)


frame, line_at_the_call, (code, lineno, globals_seen, back) = outer()
eq('the running code', code is frame.f_code, True)
eq('the running line', lineno, line_at_the_call)
eq('the running globals', globals_seen is globals(), True)
eq('the caller frame', back is frame.f_back, True)
eq('the caller is outer', back.f_code.co_name, 'outer')

# ── with nothing being propagated ──────────────────────────────────────

# CPython dereferences the exception it is not holding and dies; PyPy reports
# that there was nothing to record, which is what this answers.
eq('nothing to record', m.here_without_exception(), -1)

print('cpyext-frames-ok')
"#;

#[test]
fn the_frame_an_extension_builds() {
    let fixtures = Fixtures::new("cpyext-frames");
    fixtures.compile("cpyext_frames");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-frames-ok");
}
