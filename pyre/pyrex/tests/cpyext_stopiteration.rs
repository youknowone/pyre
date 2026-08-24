//! `PyStopIterationObject.value` — the one exception field an extension reads
//! out of the block rather than through a call.
//!
//! `__Pyx_PyGen_FetchStopIterationValue` is what reads it: every Cython module
//! that iterates a generator takes the returned value out of this word.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import gc

import cpyext_stopiteration as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


word = 8

# ── the layout ─────────────────────────────────────────────────────────

# The header and one word, which is what the Rust twin asserts.
eq('layout', m.layout(), (3 * word, 4 * word))

# ── the value the block carries ────────────────────────────────────────

# What the block holds and what the attribute answers are the same object.
carried = m.value_of(StopIteration('returned'))
eq('block and attribute agree', carried[0], carried[1])
eq('the value', carried[0], 'returned')
eq('same object', carried[0] is carried[1], True)

# The default, which is what an exception raised with no argument carries.
eq('no argument', m.value_of(StopIteration()), (None, None))

# An argument that is not a string, so nothing is being read out of a
# representation of one.
subject = ['a', 'list']
read = m.value_of(StopIteration(subject))
eq('a list value', read[0] is subject, True)

# ── the sequence Cython runs ───────────────────────────────────────────

def returns_a_value():
    yield 1
    yield 2
    return 'the return'


eq('a generator return', m.fetch_returned(returns_a_value()), 'the return')


def returns_nothing():
    yield 1


eq('a generator with no return', m.fetch_returned(returns_nothing()), None)


def returns_a_list():
    return
    yield


eq('an empty generator', m.fetch_returned(returns_a_list()), None)

# An ordinary iterator stops without a value at all.
eq('a plain iterator', m.fetch_returned(iter([1, 2])), None)

# ── derived classes ────────────────────────────────────────────────────

# In Python: the storage is the base's, so the word is in the same place.
class Subclass(StopIteration):
    pass


derived = m.value_of(Subclass('from a subclass'))
eq('a subclass carries it', derived[0], 'from a subclass')
eq('a subclass agrees', derived[0], derived[1])

# In C: the type declares its own storage, which begins with the base's, so
# the field past the end is addressable without disturbing the value.
made = m.Derived('through C')
eq('the C subclass is a StopIteration', isinstance(made, StopIteration), True)
eq('the C subclass value', m.value_of(made)[0], 'through C')
m.set_derived_marker(made, 42)
eq('its own field', m.derived_marker(made), 42)
eq('and the value is untouched', m.value_of(made)[0], 'through C')

# ── across a collection ────────────────────────────────────────────────

# The word holds a reference of its own, so the value survives losing every
# other one.
held = m.value_of(StopIteration(''.join(['s', 'urvived'])))
gc.collect()
eq('the value survives a collection', held[0], 'survived')

exceptions = [StopIteration(index) for index in range(50)]
read = [m.value_of(error)[0] for error in exceptions]
gc.collect()
eq('every value', read, list(range(50)))
eq('and again after a collection',
   [m.value_of(error)[0] for error in exceptions], list(range(50)))

print('cpyext-stopiteration-ok')
"#;

#[test]
fn the_value_a_stop_iteration_carries() {
    let fixtures = Fixtures::new("cpyext-stopiteration");
    fixtures.compile("cpyext_stopiteration");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-stopiteration-ok");
}
