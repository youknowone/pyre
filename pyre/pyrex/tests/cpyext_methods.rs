//! End-to-end check for the method calling conventions, the argument parsers,
//! the object constructors and the exception indicator, driven from a
//! multi-phase (PEP 489) extension.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import sys
import cpyext_methods as m

# ── the exec slot ran, and its module-state block is per-module ────────
assert m.__name__ == 'cpyext_methods'
assert m.__doc__ == 'pyre cpyext method module'
assert m.__file__.endswith('.so'), m.__file__
assert m.ANSWER == 42
assert m.GREETING == 'hi'
assert m.OWNED == 11
assert m.SHARED == 12
assert m.VIA_DICT == 'through-dict'

assert m.bump() == 1
assert m.bump() == 2
assert m.bump() == 3

# ── the carrier ────────────────────────────────────────────────────────
assert repr(m.bump) == '<built-in function bump>'
assert m.bump.__name__ == 'bump'
assert m.bump.__qualname__ == 'bump'
assert m.bump.__doc__ == 'bump the module counter'
assert m.wrap.__doc__ is None
assert m.bump.__self__ is m
assert m.bump.__module__ == 'cpyext_methods'
assert callable(m.bump)

def rejects(call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except TypeError:
        return
    raise AssertionError('%r accepted %r %r' % (call, args, kwargs))

# ── METH_NOARGS / METH_O ───────────────────────────────────────────────
rejects(m.bump, 1)
rejects(m.bump, key=1)
assert m.wrap(7) == (7, 'seen')
rejects(m.wrap)
rejects(m.wrap, 1, 2)

# ── METH_VARARGS and PyArg_ParseTuple ──────────────────────────────────
assert m.add(1, 2) == 3
assert m.add(5) == 15
rejects(m.add)
rejects(m.add, 1, 2, 3)

# ── METH_VARARGS | METH_KEYWORDS and PyArg_ParseTupleAndKeywords ───────
assert m.greet('pyre') == 'hello pyre!'
assert m.greet('pyre', '?') == 'hello pyre?'
assert m.greet('pyre', punct='.') == 'hello pyre.'
assert m.greet(name='pyre') == 'hello pyre!'
assert m.greet(name='pyre', punct='?') == 'hello pyre?'
rejects(m.greet, 'pyre', nope=1)
rejects(m.greet)

# ── METH_FASTCALL, with and without keywords ───────────────────────────
assert m.total() == 0
assert m.total(1, 2, 3) == 6
assert m.layout() == ([], [])
assert m.layout(1, 2) == ([1, 2], [])
assert m.layout(1, a=2, b=3) == ([1], [('a', 2), ('b', 3)])

# ── PyArg_UnpackTuple and the object protocol ──────────────────────────
assert m.apply(abs, -4) == 4
assert m.apply(str, 12) == '12'
rejects(m.apply, abs)

present, text, shown, size, truth = m.inspect([1, 2], 'append')
assert (present, text, shown, size, truth) == (1, '[1, 2]', '[1, 2]', 2, 1)
present, text, shown, size, truth = m.inspect(7, 'append')
assert (present, text, shown, size, truth) == (0, '7', '7', -1, 1)

# ── Py_BuildValue ──────────────────────────────────────────────────────
assert m.build() == {
    'int': 3,
    'long': 4,
    'float': 1.5,
    'str': 'text',
    'bytes': b'raw',
    'list': [1, 2, 3],
    'tuple': ('pair', 9),
}

# ── str / bytes round trips through the mirror's byte cache ────────────
assert m.roundtrip('sévère', b'a\x00b') == ('sévère', b'a\x00b', 6, 3)
assert m.roundtrip('', b'') == ('', b'', 0, 0)
rejects(m.roundtrip, b'bytes', b'bytes')

# ── int / float ────────────────────────────────────────────────────────
assert m.numbers(42) == (42, 42.0, -42, 1, 0, 0.25)
assert m.numbers(True) == (1, 1.0, -42, 1, 0, 0.25)

# ── dict / tuple / list ────────────────────────────────────────────────
assert m.dict_ops() == (7, 1, 1, 0, 1)
assert m.sequences() == ((1, 'two'), [5, b'ab\x00c'], 2, 2)

# ── the singletons are the interpreter's own objects ───────────────────
none, true, false, ellipsis, notimplemented = m.singletons()
assert none is None
assert true is True
assert false is False
assert ellipsis is Ellipsis
assert notimplemented is NotImplemented
assert m.predicates(None) == (1, 0, 0, 0, 0, 0, 0)
assert m.predicates(True) == (0, 1, 0, 0, 0, 0, 0)
assert m.predicates('x') == (0, 0, 1, 0, 0, 0, 0)
assert m.predicates(b'x') == (0, 0, 0, 1, 0, 0, 0)
assert m.predicates(()) == (0, 0, 0, 0, 1, 0, 0)
assert m.predicates([]) == (0, 0, 0, 0, 0, 1, 0)
assert m.predicates({}) == (0, 0, 0, 0, 0, 0, 1)

# ── the exception indicator ────────────────────────────────────────────
def raises(kind, cls, message):
    try:
        m.fail(kind)
    except cls as error:
        assert str(error) == message, (kind, str(error))
        return
    raise AssertionError('fail(%r) did not raise' % (kind,))

raises('value', ValueError, 'a value complaint')
raises('format', TypeError, 'formatted message and 7')
raises('object', KeyError, "'payload'")
raises('none', StopIteration, '')
raises('memory', MemoryError, 'out of memory')
raises('argument', TypeError, 'bad argument type for built-in operation')
raises('other', RuntimeError, 'unknown failure kind')

assert m.caught() == (1, 1, 0, 1, 'swallowed')
# The swallowed exception must not leak into the caller.
assert m.bump() == 4

# ── a re-import rebuilds the module from its definition ────────────────
m.runtime_only = 1
del sys.modules['cpyext_methods']
again = __import__('cpyext_methods')
assert again is not m
assert not hasattr(again, 'runtime_only')
assert again.ANSWER == 42
assert again.bump() == 1

# ── the call entry points ──────────────────────────────────────────────
# `record` answers with exactly what it was given, so each spelling's result
# states the argument vector that reached it.
def record(*args, **kwargs):
    return (args, kwargs)

noargs, onearg, vec, vec_kw, objargs, fmt, meth_no, meth_fmt = \
    m.call_surface(record, 7, 'v')

assert noargs == ((), {}), noargs
assert onearg == ((7,), {}), onearg
assert vec == ((7, 7), {}), vec
# One positional and one named: the name comes from the `kwnames` tuple and
# names the entry *after* the positional run.
assert vec_kw == ((7,), {'kw': 'v'}), vec_kw
assert objargs == ((7, 7, 7), {}), objargs
assert fmt == ((7, 7), {}), fmt
# The method spellings, against 'abc'.
assert meth_no == 'ABC', meth_no
assert meth_fmt == 1, meth_fmt

# ── the set protocol and the raw allocators ────────────────────────────
(empty_size, after_add, after_readd, has_key,
 first, second, after_pop, after_clear,
 is_set, is_frozen, is_any, typed) = m.set_ops([1, 2, 3], 9)

assert empty_size == 0, empty_size
# 9 is not in {1,2,3}, so adding it grows the set and re-adding does not.
assert (after_add, after_readd) == (4, 4), (after_add, after_readd)
assert has_key == 1
# `discard` finds it once, then reports absence rather than raising.
assert (first, second) == (1, 0), (first, second)
assert after_pop == 2, after_pop
assert after_clear == 0, after_clear
assert (is_set, is_frozen, is_any) == (1, 1, 1), (is_set, is_frozen, is_any)
# PyMem_Calloc zeroed, PyMem_Realloc kept the bytes, PyMem_New sized right.
assert typed == 1234, typed

print('cpyext-methods-ok')
"#;

#[test]
fn calls_c_functions_through_every_supported_convention() {
    let fixtures = Fixtures::new("cpyext-methods");
    fixtures.compile("cpyext_methods");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-methods-ok");
}
