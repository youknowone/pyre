//! The callables an extension makes and asks about: the `PyMethodDef`-backed
//! function, the bound method, the modules dict, and the reports a call that
//! cannot raise leaves behind.
//!
//! Every expectation was taken from CPython 3.14 running this same script
//! against this same fixture.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import sys
import cpyext_callables as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


class Named:
    def method(self):
        return self


# ── what is backed by a C function ─────────────────────────────────────

eq('c function extension', m.is_c_function(m.probe), 1)
eq('c function python', m.is_c_function(Named.method), 0)
eq('c function bound python', m.is_c_function(Named().method), 0)
eq('c function int', m.is_c_function(7), 0)
eq('c function type', m.is_c_function(int), 0)

eq('probe is probe', m.is_probe(m.probe), 1)
eq('probe is not len', m.is_probe(len), 0)
eq('probe is not a str', m.is_probe('x'), 0)

# METH_O, and the receiver a module-level function carries is its module.
flags, receiver = m.describe(m.probe)
eq('probe flags', flags, 8)
eq('probe receiver', receiver is m, True)

# ── a callable built from a definition ─────────────────────────────────

made = m.make_c_function(m)
eq('made is a c function', m.is_c_function(made), 1)
eq('made is probe', m.is_probe(made), 1)
eq('made calls through', made('through'), 'through')
eq('made name', made.__name__, 'probe')
eq('made is fresh', made is m.probe, False)

# ── binding ────────────────────────────────────────────────────────────

subject = Named()
method, checked, function, bound_self = m.bind(Named.method, subject)
eq('bind checks', checked, 1)
eq('bind function', function is Named.method, True)
eq('bind self', bound_self is subject, True)
eq('bind calls', method() is subject, True)
eq('bind equals', method == subject.method, True)

# ── the modules dict ───────────────────────────────────────────────────

eq('modules is sys.modules', m.module_dict() is sys.modules, True)
eq('modules holds this one', m.module_dict()['cpyext_callables'] is m, True)

# ── __import__ with every argument stated ──────────────────────────────

eq('import top level', m.import_level('sys', None, 0) is sys, True)
# A fromlist asks for the leaf rather than the package.
encodings = m.import_level('encodings.utf_8', ['decode'], 0)
eq('import fromlist', encodings.__name__, 'encodings.utf_8')

# ── the borrowed setdefault ────────────────────────────────────────────

table = {'x': 1}
eq('setdefault present', m.set_default(table, 'x', 99), 1)
eq('setdefault absent', m.set_default(table, 'y', 2), 2)
eq('setdefault stored', table, {'x': 1, 'y': 2})

# ── a call whose keywords are a mapping ────────────────────────────────

def takes(a, b, c=3, *, d=4):
    return (a, b, c, d)


eq('vectorcall dict', m.call_with_dict(takes, (1, 2), {'d': 40}), (1, 2, 3, 40))
eq('vectorcall dict empty', m.call_with_dict(takes, (1, 2, 3), {}), (1, 2, 3, 4))
eq('vectorcall dict only kw', m.call_with_dict(takes, (), {'a': 1, 'b': 2}),
   (1, 2, 3, 4))

# ── what a call that cannot raise reports ──────────────────────────────

reports = []
previous = sys.unraisablehook
sys.unraisablehook = reports.append
try:
    m.report_unraisable(subject)
    m.report_unraisable_msg()
finally:
    sys.unraisablehook = previous

eq('two reports', len(reports), 2)
eq('named object', reports[0].object is subject, True)
eq('named exception', type(reports[0].exc_type), type)
eq('named exception value', str(reports[0].exc_value), 'boom')
eq('stated message', reports[1].err_msg,
   'Exception ignored while doing the thing')
eq('stated object', reports[1].object, None)
# Reporting clears the indicator, so nothing is pending afterwards.
eq('nothing pending', m.probe('still working'), 'still working')

# ── the runtime it is running against ──────────────────────────────────

version, interpreter = m.runtime_identity()
eq('version', version, sys.hexversion)
eq('interpreter id', interpreter, 0)

# ── tracebacks ─────────────────────────────────────────────────────────

try:
    raise ValueError('x')
except ValueError as error:
    eq('is a traceback', m.is_traceback(error.__traceback__), 1)
eq('is not a traceback', m.is_traceback(None), 0)

print('cpyext-callables-ok')
"#;

#[test]
fn the_callables_an_extension_makes() {
    let fixtures = Fixtures::new("cpyext-callables");
    fixtures.compile("cpyext_callables");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-callables-ok");
}
