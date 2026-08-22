//! Deriving in Python from a type an extension defined: what the subtype's own
//! mirror carries, what a C constructor handed that subtype can do with it, and
//! the iteration slots a caller reads off a type this runtime defines.
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

# A method the subtype does not define is reached through the base's own slot,
# not through one installed here.  A slot installed for an inherited method
# would resolve the name back to the wrapper that reads this very slot, and
# the two would call each other until the stack ran out.
eq('the repr a C base declares', repr(m.Cell(1)), '<Cell 1>')
eq('and the subtype inherits it', repr(Sub(2)), '<Cell 2>')
eq('read off the type, as a compiled module reads it',
   m.call_slot('tp_repr', Sub(3)), '<Cell 3>')

# A wrapper answers with the slot of the class it was published on, so a
# subclass that overrides the method and then names the base explicitly
# reaches the base's implementation rather than its own.


class Louder(m.Cell):
    def __repr__(self):
        return 'Louder(' + m.Cell.__repr__(self) + ')'


eq('a wrapper named on the base', repr(Louder(4)), 'Louder(<Cell 4>)')
eq('and the override is what the type answers with',
   m.call_slot('tp_repr', Louder(5)), 'Louder(<Cell 5>)')

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


# ── the iteration slots ────────────────────────────────────────────────

# A compiled loop takes `tp_iter` off the type and then `tp_iternext` off the
# iterator, rather than calling `PyObject_GetIter` and `PyIter_Next`.  A type
# this runtime defines has to carry both, or the loop dereferences a null.


class Counted:
    def __init__(self, upto):
        self.upto = upto

    def __iter__(self):
        return iter(range(self.upto))


class OwnIterator:
    def __init__(self):
        self.left = 2

    def __iter__(self):
        return self

    def __next__(self):
        if self.left == 0:
            raise StopIteration
        self.left -= 1
        return self.left


for name, walked, wanted in [
        ('a list', [1, 2, 3], [1, 2, 3]),
        ('a tuple', (1, 2), [1, 2]),
        ('a str', 'ab', ['a', 'b']),
        ('a set', {7}, [7]),
        ('a dict', {'k': 1}, ['k']),
        ('a range', range(3), [0, 1, 2]),
        ('a generator', (x * 2 for x in range(3)), [0, 2, 4]),
        ('a class with __iter__', Counted(3), [0, 1, 2]),
        ('a class with both', OwnIterator(), [1, 0])]:
    eq('walking %s through its slots' % name, m.walk_slots(walked), wanted)


class L(list):
    pass


class T(tuple):
    pass


eq('walking a list subclass', m.walk_slots(L([4, 5])), [4, 5])
eq('walking a tuple subclass', m.walk_slots(T((6,))), [6])

# The slot is only there where the method is: an object that is not iterable
# carries no `tp_iter` to read.
eq('a class with no __iter__', m.walk_slots(object()), 'no-tp_iter')

# And an iterator's own `__next__` is not published back as a method of the
# type -- the slot that reaches it resolves the same name.
eq('the method a walked iterator answers with',
   type(iter([1])).__next__.__qualname__.endswith('__next__'), True)
eq('and it still works from Python', next(iter([9])), 9)


# ── an exception the walk raises ───────────────────────────────────────

class Angry:
    def __iter__(self):
        return self

    def __next__(self):
        raise ValueError('no')


try:
    m.walk_slots(Angry())
except ValueError as exc:
    eq('the exception a slot walk reports', str(exc), 'no')
else:
    raise AssertionError('a raising __next__ was read as exhaustion')

# ── the slots a mirror carries for what this runtime answers ───────────

# Every one of these is a raw field read in a compiled module, so a NULL is
# read as "the type does not do this" and not as an error.
eq('the repr slot', m.call_slot('tp_repr', [1, 2]), '[1, 2]')
eq('the str slot', m.call_slot('tp_str', {'a': 1}), "{'a': 1}")
eq('the hash slot', m.call_slot('tp_hash', 'text'), hash('text'))
eq('the sequence length slot', m.call_slot('sq_length', [1, 2, 3]), 3)
eq('the mapping length slot', m.call_slot('mp_length', {'a': 1, 'b': 2}), 2)
eq('the int slot', m.call_slot('nb_int', 7.5), 7)
eq('the float slot', m.call_slot('nb_float', 3), 3.0)
eq('the index slot', m.call_slot('nb_index', True), 1)
eq('the negation slot', m.call_slot('nb_negative', 5), -5)
eq('the identity slot', m.call_slot('nb_positive', -5), -5)
eq('the absolute slot', m.call_slot('nb_absolute', -5), 5)
eq('the inversion slot', m.call_slot('nb_invert', 5), -6)


# A class written in Python reaches its own methods through the same slots.
class Counted:
    def __len__(self):
        return 4

    def __repr__(self):
        return '<counted>'

    def __index__(self):
        return 11


counted = Counted()
eq('a Python class fills the length slot', m.call_slot('sq_length', counted), 4)
eq('a Python class fills the repr slot', m.call_slot('tp_repr', counted), '<counted>')
eq('a Python class fills the index slot', m.call_slot('nb_index', counted), 11)

# And a name the class does not define leaves the slot NULL rather than
# installing something that would fail when called.
eq('an absent method leaves no slot', m.call_slot('nb_invert', counted), 'none')
eq('a plain object has no length slot', m.call_slot('sq_length', object()), 'none')

# An override is reached through the slot its base filled.
class Louder(Counted):
    def __repr__(self):
        return '<louder>'


eq('an override answers the slot', m.call_slot('tp_repr', Louder()), '<louder>')


class Quiet(Counted):
    pass


# A subclass that overrides nothing takes its base's slot, and that slot still
# reaches the method the base defines.
eq('an inherited method reaches the base slot', m.call_slot('sq_length', Quiet()), 4)
eq('and so does an inherited repr', m.call_slot('tp_repr', Quiet()), '<counted>')

# A raising method surfaces as the failure it is, not as a slot's own error.
class Broken:
    def __len__(self):
        raise ValueError('no length')


try:
    m.call_slot('sq_length', Broken())
except ValueError as exc:
    eq('the exception a length slot reports', str(exc), 'no length')
else:
    raise AssertionError('a raising __len__ was read as a count')


# The slots taking a second operand.
eq('the addition slot', m.call_slot('nb_add', 2, 3), 5)
eq('the subtraction slot', m.call_slot('nb_subtract', 9, 4), 5)
eq('the remainder slot', m.call_slot('nb_remainder', 9, 4), 1)
eq('the conjunction slot', m.call_slot('nb_and', 12, 10), 8)
eq('the concatenation slot', m.call_slot('sq_concat', [1], [2]), [1, 2])
eq('the subscript slot', m.call_slot('mp_subscript', {'a': 1}, 'a'), 1)
eq('the item slot', m.call_slot('sq_item', [7, 8, 9], 1), 8)
eq('the repeat slot', m.call_slot('sq_repeat', [0], 3), [0, 0, 0])

# `list` and `tuple` carry no number suite, so a caller testing for one reads
# them as the sequences they are -- their `__add__` is the concatenation slot.
eq('a list has no addition slot', m.call_slot('nb_add', [1], [2]), 'none')
eq('a tuple has no addition slot', m.call_slot('nb_add', (1,), (2,)), 'none')
eq('a str has no repetition slot', m.call_slot('nb_multiply', 'ab', 2), 'none')
eq('a str still repeats', m.call_slot('sq_repeat', 'ab', 2), 'abab')


class Boxed:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return Boxed(self.value + other.value)

    def __getitem__(self, key):
        return ('got', key)

    def __setitem__(self, key, value):
        self.value = (key, value)

    def __delitem__(self, key):
        self.value = ('deleted', key)

    def __pow__(self, other, modulus=None):
        return (self.value, other, modulus)


eq('a Python class fills the addition slot',
   m.call_slot('nb_add', Boxed(2), Boxed(3)).value, 5)

# One `__getitem__` reaches both of the slots an item read is spelled with.
eq('the mapping half', m.call_slot('mp_subscript', Boxed(0), 'k'), ('got', 'k'))
eq('the sequence half', m.call_slot('sq_item', Boxed(0), 2), ('got', 2))

# And one slot answers for the assignment and the deletion both.
box = Boxed(0)
m.call_slot('mp_ass_subscript', box, 'k', 'v')
eq('the assignment a slot makes', box.value, ('k', 'v'))
m.call_slot('mp_ass_subscript', box, 'k')
eq('the deletion the same slot makes', box.value, ('deleted', 'k'))
m.call_slot('sq_ass_item', box, 2, 'v')
eq('the assignment the sequence half makes', box.value, (2, 'v'))

# The power slot takes the modulus as its third operand, and a caller with
# none of its own passes None.
eq('the power slot', m.call_slot('nb_power', Boxed(2), 3), (2, 3, None))
eq('the power slot with a modulus', m.call_slot('nb_power', Boxed(2), 3, 5), (2, 3, 5))
eq('an int answers the power slot', m.call_slot('nb_power', 2, 10), 1024)


# ── the slots handed a call's own arguments ────────────────────────────

class Adder:
    def __init__(self, base=0):
        self.base = base

    def __call__(self, *args, **kwds):
        return (self.base, args, sorted(kwds.items()))


eq('the call slot', m.call_with_args('tp_call', Adder(1), (2, 3)), (1, (2, 3), []))
eq('the call slot with keywords',
   m.call_with_args('tp_call', Adder(1), (), {'k': 4}), (1, (), [('k', 4)]))

# `__call__` is not handed down, so a class that defines none carries no slot
# to read even though its base does.
class Quietly(Adder):
    pass


eq('a subclass that defines no __call__', m.call_with_args('tp_call', Quietly(), ()), 'none')

started = Adder(0)
m.call_with_args('tp_init', started, (7,))
eq('the init slot ran', started.base, 7)
eq('the constructor slot', type(m.call_with_args('tp_new', Adder, ())), Adder)

# Attribute reads go through the type's own `__getattribute__`, not past it.
class Watched:
    def __getattribute__(self, name):
        return 'saw ' + name


eq('the attribute slot', m.call_slot('tp_getattro', Watched(), 'anything'), 'saw anything')
eq('a plain object still reads its attributes',
   m.call_slot('tp_getattro', Adder(5), 'base'), 5)


# ── the descriptor slots ───────────────────────────────────────────────

class Described:
    def __get__(self, obj, of_type):
        return ('got', obj is None, of_type is None)

    def __set__(self, obj, value):
        obj.stored = value

    def __delete__(self, obj):
        obj.stored = 'gone'


class Holder:
    pass


d = Described()
eq('the descriptor read slot', m.call_slot('tp_descr_get', d, Holder(), Holder),
   ('got', False, False))
eq('and a class access hands it no receiver',
   m.call_slot('tp_descr_get', d, None, Holder), ('got', True, False))

holder = Holder()
m.call_slot('tp_descr_set', d, holder, 'value')
eq('the descriptor write slot', holder.stored, 'value')
m.call_slot('tp_descr_set', d, holder)
eq('and the deletion the same slot answers for', holder.stored, 'gone')


# The suites a heap type names are the blocks its own layout declares, so the
# two ways an extension reaches one land on the same words.
eq('a class names its own suites', m.suites_are_embedded(Counted), (1, 1, 1, 1, 1))
eq('a static type has no such block', m.suites_are_embedded(dict), 'not-a-heap-type')


print('cpyext-derive-ok')
"#;

#[test]
fn derives_in_python_from_a_c_type() {
    let fixtures = Fixtures::new("cpyext-derive");
    fixtures.compile("cpyext_derive");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-derive-ok");
}
