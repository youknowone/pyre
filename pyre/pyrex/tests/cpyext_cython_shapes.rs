//! The shapes a Cython module compiles to.
//!
//! Every case here was found by building SQLAlchemy's `cyextension` package
//! against these headers and importing it: a callable whose `tp_call` is
//! `PyVectorcall_Call`, a `cdef class` laid out against its base's ancestry
//! tuples, a `set` subclass whose `add` calls `PySet_Add`, and a deallocator
//! that ends in its base's.  Each reached a field or an entry point that was
//! not there; the first two recursed until the stack ran out, and the last
//! called through address zero.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import gc
import weakref

import cpyext_cython_shapes as m

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

# ── a callable whose `tp_call` is `PyVectorcall_Call` ──────────────────
# Its function is reached through the `__vectorcalloffset__` member, and
# answering through `tp_call` instead would arrive back here.
call = m.Caller()
eq('no arguments', call(), (0, None, None, None))
eq('positional', call('a', 'b'), (2, None, 'a', None))
eq('keyword', call('a', k='v'), (1, ('k',), 'a', 'v'))
eq('keywords only', call(k='v', j='w'), (0, ('k', 'j'), None, 'w'))

# ── the ancestry a `cdef class` is laid out against ────────────────────
# A type built from C, one this runtime defines, and a class written in
# Python all have to answer.
class Plain:
    pass

class Derived(dict):
    pass

eq('ancestry(Caller)', m.ancestry(type(call)), (1, 2))
eq('ancestry(object)', m.ancestry(object), (0, 1))
eq('ancestry(dict)', m.ancestry(dict), (1, 2))
eq('ancestry(Plain)', m.ancestry(Plain), (1, 2))
eq('ancestry(Derived)', m.ancestry(Derived), (1, 3))
eq('first_base(Derived)', m.first_base(Derived), dict)
eq('first_base(Caller)', m.first_base(type(call)), object)

# ── the set protocol reaches the storage, not the override ─────────────
class Loud(set):
    """Raises from every method a concrete operation must not consult."""
    def add(self, element): raise AssertionError('add consulted')
    def remove(self, element): raise AssertionError('remove consulted')
    def discard(self, element): raise AssertionError('discard consulted')
    def pop(self): raise AssertionError('pop consulted')
    def clear(self): raise AssertionError('clear consulted')
    def __contains__(self, element): raise AssertionError('__contains__ consulted')
    def __len__(self): raise AssertionError('__len__ consulted')

for tag, make in (('set', lambda: set()), ('subclass', lambda: Loud())):
    eq('add(%s)' % tag, m.set_add(make(), 'k'), 1)
    eq('contains(%s)' % tag, m.set_contains(make(), 'k'), False)

    filled = make()
    m.set_add(filled, 'k')
    eq('contains after add(%s)' % tag, m.set_contains(filled, 'k'), True)
    eq('discard present(%s)' % tag, m.set_discard(filled, 'k'), 1)
    eq('discard absent(%s)' % tag, m.set_discard(filled, 'k'), 0)

    popping = make()
    m.set_add(popping, 'only')
    eq('pop(%s)' % tag, m.set_pop(popping), 'only')

    clearing = make()
    m.set_add(clearing, 'k')
    eq('clear(%s)' % tag, m.set_clear(clearing), 0)

# The write reaches the object Python holds.
target = Loud()
m.set_add(target, 'k')
eq('write is visible to Python', sorted(set.__iter__(target)), ['k'])
eq('write changes the length', set.__len__(target), 1)

# An unhashable element is refused rather than stored -- the entry point
# hashes it itself, so the refusal cannot come from a method it skipped.
eq('add(unhashable)', m.set_add(set(), []), 'add-failed')
eq('add(unhashable, subclass)', m.set_add(Loud(), []), 'add-failed')
eq('contains(unhashable)', m.set_contains(set(), []), 'contains-failed')

# ── the spellings the generated C reaches for ─────────────────────────
eq('dict_get_size', m.dict_get_size({'a': 1, 'b': 2}), 2)
eq('dict_get_size(subclass)', m.dict_get_size(Derived(a=1)), 1)

eq('exactness(set)', m.exactness(set()), (True, False, True, True))
eq('exactness(frozenset)', m.exactness(frozenset()), (False, True, True, True))
eq('exactness(subclass)', m.exactness(Loud()), (False, False, False, True))
eq('exactness(list)', m.exactness([]), (False, False, False, False))

eq('recursive_call', m.recursive_call(None), 'entered-and-left')

# ── the chain a deallocator walks ─────────────────────────────────────
# A deallocator written for a type derived from a builtin ends in the
# base's, so the base has to carry one.
eq('base deallocators', m.base_deallocs(), (True, True, True))

Sub = m.dict_subclass()
made = m.instantiate(Sub)
made['a'] = 1
eq('the instance is a mapping', dict(made), {'a': 1})
eq('the chain has not run', m.chain_count(), 0)
del made
for _ in range(4):
    if m.chain_count():
        break
    gc.collect()
eq('the chain ran once', m.chain_count(), 1)

# Reading a class's ancestry from C does not make it outlive the last
# reference to it, which the tuples the read mints would otherwise see to:
# an MRO names its own type.  A type an extension defined is immortal by
# construction, so the class asked here is one written in Python.
watched = type('Watched', (), {})
eq('ancestry(Watched)', m.ancestry(watched), (1, 2))
watch = weakref.ref(watched)
del watched
for _ in range(4):
    if watch() is None:
        break
    gc.collect()
eq('the class was collected', watch(), None)

print('cpyext-cython-shapes-ok')
"#;

#[test]
fn the_shapes_a_cython_module_compiles_to_are_answered() {
    let fixtures = Fixtures::new("cpyext-cython-shapes");
    fixtures.compile("cpyext_cython_shapes");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-cython-shapes-ok");
}
