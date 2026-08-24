//! A slot read off one type and called with an instance of another.
//!
//! `get_slot_tp_function` binds the method on the type that OWNS the slot, so
//! `base->tp_slot(self)` from a subclass runs the base's implementation.  A
//! slot that resolved the name on the receiver instead would answer with the
//! subclass's override -- or, where the override is the very wrapper that
//! reads this slot, call itself until the stack ran out.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import cpyext_bound_slot as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


class Loud(dict):
    def __repr__(self):
        return 'LOUD'

    def __len__(self):
        return 999


class LoudItems(list):
    def __getitem__(self, at):
        return 'LOUD'


loud = Loud(a=1)
items = LoudItems([7, 8])

# The slot was read off `dict`, so it answers with `dict`'s method even though
# the receiver's own type overrides it.
eq('a slot read off the base', m.repr_through(loud), "{'a': 1}")
eq('a count read off the base', m.length_through(loud), 1)
eq('an item read off the base', m.item_through(items, 0), 7)

# And the override is still what the language answers with, which is what
# would break if the slot were simply pinned to the base everywhere.
eq('the override is untouched', (repr(loud), len(loud), items[0]), ('LOUD', 999, 'LOUD'))

# A class that defines none of its own takes the same slot as the type it
# inherits the method from; one that defines its own does not.
class Plain(dict):
    pass


eq('an inherited method shares the base\'s slot',
   m.same_repr_slot(Plain, dict), True)
eq('and one of its own does not', m.same_repr_slot(Loud, dict), False)

# Reading the same slot twice hands back one function, so a mirror refilled
# does not spend a second entry.
eq('the slot is stable', m.same_repr_slot(dict, dict), True)

print('cpyext-bound-slot-ok')
"#;

#[test]
fn a_slot_read_off_a_base_runs_the_bases_method() {
    let fixtures = Fixtures::new("cpyext-bound-slot");
    fixtures.compile("cpyext_bound_slot");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-bound-slot-ok");
}
