//! The runtime services an extension reaches for around a call: the exception
//! a failed syscall becomes, the audit events it raises, and importing one
//! attribute out of a module.
//!
//! Every expectation was taken from CPython 3.14.6 running this same script
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
import cpyext_runtime as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


class Custom(OSError):
    pass


class Plain(Exception):
    pass


# ── the failed syscall ─────────────────────────────────────────────────

# The class is called rather than consulted, so `OSError` maps the code to its
# subclass exactly as it does from Python and a class outside the family is
# built from the same two arguments.
eq('errno enoent', m.from_errno('plain', m.ENOENT, OSError, None, None),
   ('FileNotFoundError', (2, 'No such file or directory'),
    '[Errno 2] No such file or directory'))
eq('errno eperm', m.from_errno('plain', m.EPERM, OSError, None, None),
   ('PermissionError', (1, 'Operation not permitted'),
    '[Errno 1] Operation not permitted'))
# A syscall that failed without recording which way.
eq('errno unset', m.from_errno('plain', 0, OSError, None, None),
   ('OSError', (0, 'Error'), '[Errno 0] Error'))
eq('errno subclass', m.from_errno('plain', m.ENOENT, Custom, None, None),
   ('Custom', (2, 'No such file or directory'),
    '[Errno 2] No such file or directory'))
# Outside the family there is no `__str__` folding the pair, so the message is
# the args tuple's own repr.
eq('errno foreign', m.from_errno('plain', m.ENOENT, Plain, None, None),
   ('Plain', (2, 'No such file or directory'),
    "(2, 'No such file or directory')"))
eq('errno value error', m.from_errno('plain', m.ENOENT, ValueError, None, None),
   ('ValueError', (2, 'No such file or directory'),
    "(2, 'No such file or directory')"))

# The filename spellings, which add it as the third argument.
eq('errno filename', m.from_errno('filename', m.ENOENT, OSError, '/no/such', None),
   ('FileNotFoundError', (2, 'No such file or directory'),
    "[Errno 2] No such file or directory: '/no/such'"))
eq('errno filename null', m.from_errno('filename', m.ENOENT, OSError, None, None),
   ('FileNotFoundError', (2, 'No such file or directory'),
    '[Errno 2] No such file or directory'))
eq('errno bytes object', m.from_errno('object', m.ENOENT, OSError, b'/bytes/path', None),
   ('FileNotFoundError', (2, 'No such file or directory'),
    "[Errno 2] No such file or directory: b'/bytes/path'"))
eq('errno text object', m.from_errno('object', m.ENOENT, OSError, '/text/path', None),
   ('FileNotFoundError', (2, 'No such file or directory'),
    "[Errno 2] No such file or directory: '/text/path'"))
# A second filename is the fifth argument, the fourth being the Windows code.
eq('errno two names', m.from_errno('objects', m.ENOENT, OSError, '/one', '/two'),
   ('FileNotFoundError', (2, 'No such file or directory'),
    "[Errno 2] No such file or directory: '/one' -> '/two'"))
eq('errno one of two', m.from_errno('objects', m.ENOENT, OSError, '/one', None),
   ('FileNotFoundError', (2, 'No such file or directory'),
    "[Errno 2] No such file or directory: '/one'"))

# Nothing pending is the only signal state a test can arrange without sending
# itself one.
eq('check signals', m.check_signals(), (0, None))

# ── the audit events ───────────────────────────────────────────────────

seen = []


def hook(event, args):
    if event.startswith('pyre.'):
        seen.append((event, args))


sys.addaudithook(hook)

# The format reaches the hook as a tuple however it was written: a single unit
# is wrapped, a multi-unit format is already one, and no format at all is empty.
for shape in ('none', 'empty', 'one', 'two', 'int', 'string'):
    eq('audit %s' % shape, m.audit('pyre.' + shape, shape, 'value'), (0, None))
eq('audit args', seen, [
    ('pyre.none', ()),
    ('pyre.empty', ()),
    ('pyre.one', ('value',)),
    ('pyre.two', ('value', 7)),
    ('pyre.int', (42,)),
    ('pyre.string', ('text',)),
])

# The tuple spelling: NULL is no arguments, and anything that is not a tuple is
# refused before any hook runs.
seen.clear()
eq('audit tuple', m.audit_tuple('pyre.tuple', (1, 2)), (0, None))
eq('audit empty tuple', m.audit_tuple('pyre.tuple', ()), (0, None))
eq('audit null tuple', m.audit_tuple('pyre.tuple', None), (0, None))
eq('audit list', m.audit_tuple('pyre.tuple', [1, 2]),
   (-1, ('TypeError', 'args must be tuple, got list')))
eq('audit int', m.audit_tuple('pyre.tuple', 5),
   (-1, ('TypeError', 'args must be tuple, got int')))
eq('audit tuple args', seen,
   [('pyre.tuple', (1, 2)), ('pyre.tuple', ()), ('pyre.tuple', ())])


# A hook is Python code, so it can raise, and that is what the -1 reports.
def angry(event, args):
    if event == 'pyre.angry':
        raise RuntimeError('no')


sys.addaudithook(angry)
eq('audit raising hook', m.audit('pyre.angry', 'one', 'value'),
   (-1, ('RuntimeError', 'no')))

# ── one attribute of a module ──────────────────────────────────────────

eq('attr', m.module_attr('sys', 'maxsize'), (sys.maxsize, None))
eq('attr missing', m.module_attr('sys', 'nosuchattr'),
   (None, ('AttributeError', "module 'sys' has no attribute 'nosuchattr'")))
eq('attr no module', m.module_attr('nosuchmodule', 'x'),
   (None, ('ModuleNotFoundError', "No module named 'nosuchmodule'")))
# The import runs, so a module not yet loaded is loaded.
assert 'json' not in sys.modules, 'the fixture needs json unimported'
value, error = m.module_attr('json', 'dumps')
eq('attr imports', (value.__name__, error), ('dumps', None))

eq('attr object', m.module_attr_object('sys', 'maxsize'), (sys.maxsize, None))
eq('attr object bad name', m.module_attr_object(5, 'maxsize'),
   (None, ('TypeError', 'module name must be a string')))

print('cpyext-runtime-ok')
"#;

#[test]
fn the_runtime_services() {
    let fixtures = Fixtures::new("cpyext-runtime");
    fixtures.compile("cpyext_runtime");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-runtime-ok");
}
