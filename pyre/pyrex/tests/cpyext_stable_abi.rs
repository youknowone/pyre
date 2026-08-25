//! The exports an extension needs when it never reads pyre's headers.
//!
//! PyO3 and Cython declare the C ABI themselves, so what their objects import
//! is a set of *symbols* where pyre had only ever published macros. Ten such
//! names -- spelled here as a macro, or absent altogether -- were therefore
//! unresolvable at `dlopen`: a wheel pip had just built under pyre failed to
//! load, and the failure named no symbol.
//!
//! Every expectation here was taken from CPython 3.14.6 running the same
//! script against the same fixture.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import io, sys, traceback
import cpyext_stable_abi as m

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

# `Py_INCREF` and the stable ABI's `_Py_IncRef` move the count alike.
value = ['held']
start, public, stable = m.refcount_through_both(value)
eq('refcount_public', public, 1)
eq('refcount_stable', stable, 1)

# `Py_TYPE` called answers the same block the field read does.
eq('type_through_call', m.type_through_call(value), ('list', True))

# The interpreter that loaded the extension is running and not tearing down.
eq('runtime_state', m.runtime_state(), (1, 0))

# A critical section is entered and left around a call that runs Python.
eq('inside_critical_section', m.inside_critical_section(lambda: 'ran'), 'ran')

# `PyErr_PrintEx` reports through `sys.excepthook` and clears the indicator.
seen = []
sys.excepthook = lambda kind, value, tb: seen.append((kind.__name__, str(value)))
try:
    eq('print_pending_cleared', m.print_pending(ValueError('reported'), 0), True)
    eq('print_pending_hook', seen, [('ValueError', 'reported')])
    for name in ('last_exc', 'last_type', 'last_value', 'last_traceback'):
        assert not hasattr(sys, name), 'sys.%s set without the flag' % name
    eq('print_pending_flagged', m.print_pending(KeyError('recorded'), 1), True)
    eq('last_exc', repr(sys.last_exc), "KeyError('recorded')")
    eq('last_type', sys.last_type, KeyError)
    eq('last_value', repr(sys.last_value), "KeyError('recorded')")
finally:
    sys.excepthook = sys.__excepthook__

# `PyTraceBack_Print` writes the header line, then the entries `print_tb` does.
try:
    raise RuntimeError('walked')
except RuntimeError as error:
    tb = error.__traceback__
sink = io.StringIO()
eq('print_traceback', m.print_traceback(tb, sink), 0)
written = sink.getvalue()
assert written.startswith('Traceback (most recent call last):\n'), repr(written)
expected = io.StringIO()
traceback.print_tb(tb, None, expected)
eq('print_traceback_body',
   written[len('Traceback (most recent call last):\n'):], expected.getvalue())

# Anything that is not a traceback is refused rather than printed.
eq('print_traceback_refused', m.print_traceback(None, sink), -1)

# The decoder's exception carries the bytes it was decoding, NUL included.
error = m.decode_error('utf-8', b'a\x00\xff', 2, 3, 'invalid start byte')
eq('decode_error_type', type(error), UnicodeDecodeError)
eq('decode_error_encoding', error.encoding, 'utf-8')
eq('decode_error_object', error.object, b'a\x00\xff')
eq('decode_error_span', (error.start, error.end), (2, 3))
eq('decode_error_reason', error.reason, 'invalid start byte')

print('cpyext-stable-abi-ok')
"#;

#[test]
fn an_extension_that_declares_its_own_prototypes_finds_every_symbol() {
    let fixtures = Fixtures::new("cpyext-stable-abi");
    fixtures.compile("cpyext_stable_abi");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-stable-abi-ok");
}
