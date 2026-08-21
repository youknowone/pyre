//! Deriving in Python from a type an extension defined: what the subtype's own
//! mirror carries, and what a C constructor handed that subtype can do with it.
//!
//! The shape is Cython's, whose every `cdef class` compiles a `tp_new` that
//! reads `tp_alloc` and the struct size off the type it is called with.

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

import cpyext_derive as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


# ── the type the extension defined ─────────────────────────────────────

cell = m.slots_of(m.Cell)
eq('the C type allocates', cell['alloc'], True)
eq('the C type frees', cell['free'], True)
eq('the C type constructs', cell['new'], True)
assert cell['basicsize'] >= 3 * 8, cell['basicsize']

c = m.Cell(3, 'a')
eq('the C fields', (c.value, c.tag), (3, 'a'))


# ── a class derived from it in Python ──────────────────────────────────

class Sub(m.Cell):
    def doubled(self):
        return self.value * 2


sub = m.slots_of(Sub)

# What the wall was: a constructor written for C is handed the subtype, and
# every one of these is a field it reads off it.
eq('the subtype allocates', sub['alloc'], True)
eq('the subtype frees', sub['free'], True)
eq('the subtype constructs', sub['new'], True)
eq('the subtype answers attributes', sub['getattro'], True)
eq('the subtype names its base', sub['base'], m.Cell)
eq('the constructor is the base\'s', m.shares_new(Sub, m.Cell), True)

# The size is the base's, because the base's fields are what the constructor
# fills: a block sized for the plain header would take those writes past its
# end.
eq('the subtype is sized for the base', sub['basicsize'], cell['basicsize'])

s = Sub(5, 'b')
eq('the derived instance is of its own type', type(s) is Sub, True)
eq('the C fields through the subclass', (s.value, s.tag), (5, 'b'))
eq('a method the subclass defines', s.doubled(), 10)
eq('and it is still an instance of the base', isinstance(s, m.Cell), True)

# A class the interpreter built keeps its own namespace beside the C fields.
s.extra = 'python side'
eq('an attribute the subclass stores', s.extra, 'python side')
eq('the C field is unmoved', s.value, 5)


# ── two levels ─────────────────────────────────────────────────────────

class Deeper(Sub):
    pass


deeper = m.slots_of(Deeper)
eq('the second level names the first', deeper['base'], Sub)
eq('the second level is sized for the base', deeper['basicsize'], cell['basicsize'])
eq('and still constructs', m.shares_new(Deeper, m.Cell), True)
eq('the second level builds', m.Cell(7).value, 7)
eq('the second level instance', Deeper(9, 'c').tag, 'c')


# ── weakrefs ───────────────────────────────────────────────────────────

# `create_all_slots` gives a weakref slot to every namespace with no
# `__slots__`, and a namespace built from C has none.
held = m.Cell(11)
reference = weakref.ref(held)
eq('a weakref to an instance of a C type', reference() is held, True)
del held
gc.collect()
eq('and it is cleared', reference() is None, True)

held = Sub(12)
reference = weakref.ref(held)
eq('a weakref to an instance of a derived class', reference() is held, True)


# ── a collection over the C blocks ─────────────────────────────────────

# The blocks the collector reaches `tp_clear` on have no interpreter object
# left, and the clear this fixture runs calls `PyObject_ClearManagedDict` --
# which reports nothing, so anything it recorded would be found by the next
# call into the module rather than by it.
first = m.Cell(13)
second = m.Cell(14, first)
first.tag = second
eq('the cycle is through the C field', first.tag.tag is first, True)
del first
del second
gc.collect()
eq('the collection left no error behind', m.undisturbed(), True)
eq('and the module still builds', m.Cell(15).value, 15)

gc.collect()
eq('the subtype survives a collection', Sub(16, 'd').doubled(), 32)
eq('and its mirror still names its base', m.slots_of(Sub)['base'], m.Cell)

print('cpyext-derive-ok')
"#;

#[test]
fn derives_in_python_from_a_c_type() {
    let fixtures = Fixtures::new("cpyext-derive");
    fixtures.compile("cpyext_derive");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-derive-ok");
}
