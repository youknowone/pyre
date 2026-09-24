# cpyext-fixture: cpyext_context
# cpyext-expect: cpyext-report-ok

# The interpreter state a call runs inside: the namespace a name falls back
# to, the context variables, and reporting a failure the caller has no way to
# hand back.

import sys

import cpyext_context as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


seen = []


def recorder(kind, value, traceback):
    seen.append((kind, str(value), traceback is None))


for name in ('last_exc', 'last_type', 'last_value', 'last_traceback'):
    if hasattr(sys, name):
        delattr(sys, name)

sys.excepthook = recorder
try:
    eq('reporting one', m.report('printex', 'reported', 1), None)
finally:
    sys.excepthook = sys.__excepthook__

eq('what the hook was handed', len(seen), 1)
kind, text, no_traceback = seen[0]
eq('the class', kind is ValueError, True)
eq('the message', text, 'reported')
# Raised from C, so nothing unwound and there is no traceback to hand over.
eq('the traceback', no_traceback, True)

# The four names a post-mortem reads the exception back from.
eq('sys.last_exc', type(sys.last_exc) is ValueError, True)
eq('and its message', str(sys.last_exc), 'reported')
eq('sys.last_type', sys.last_type is ValueError, True)
eq('sys.last_value', sys.last_value is sys.last_exc, True)
eq('sys.last_traceback', sys.last_traceback is None, True)

# The indicator is cleared by the report, so the next call starts clean.
eq('nothing is left pending', m.report('printex', 'again', 0), None)

# With the flag off the names are left as the previous report set them.
eq('sys.last_exc is untouched', str(sys.last_exc), 'reported')

# `PyErr_Print` is the flag-on spelling.
del sys.last_exc
seen.clear()
sys.excepthook = recorder
try:
    eq('reporting through the short spelling',
       m.report('print', 'short', 0), None)
finally:
    sys.excepthook = sys.__excepthook__
eq('the hook saw it', [(k is ValueError, t) for k, t, _ in seen],
   [(True, 'short')])
eq('and the flag was on', str(sys.last_exc), 'short')

# A hook that raises is reported as unraisable rather than left pending, so
# the exception it raised does not surface at the next C call.
def raises(kind, value, traceback):
    raise RuntimeError('the hook failed')


sys.excepthook = raises
try:
    m.report('printex', 'swallowed', 0)
finally:
    sys.excepthook = sys.__excepthook__

# Reporting with nothing to report is the caller's mistake.
eq('reporting nothing', m.report_nothing(), 'SystemError')

print('cpyext-report-ok')
