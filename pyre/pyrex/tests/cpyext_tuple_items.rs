//! The tuple item array `PyTuple_GET_ITEM` and `PyTuple_SET_ITEM` reach.
//!
//! Both are assignments over an array upstream rather than calls, and an
//! extension takes the address of a slot and writes one that already holds a
//! value.  cffi does all three.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import cpyext_tuple_items as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


a, b, c = object(), object(), object()

eq('a slot read through its address', m.first_through_address((a, b, c)), a)
eq('and through the address of an arity-2 tuple',
   m.first_through_address((7, 8)), 7)
eq('every slot', m.items((a, b, c)), [a, b, c])
eq('an empty tuple has none', m.items(()), [])

# The decorator shape: slot 1 holds `b`, `c` stands in it for one call, and
# `b` goes back.  Neither write owns what it stores.
outer = (a, b, 'tail')
seen, from_c = m.borrow_swap(outer, c, lambda t: t[1])
eq('the reader saw the substitute', seen, c)
eq('so did C', from_c, c)
eq('and the tuple reads its own value again', outer[1], b)
eq('the slots either side are untouched', (outer[0], outer[2]), (a, 'tail'))

# A slot written twice reads the second value: the setter overwrites rather
# than rejecting a filled slot.
eq('a slot written twice', m.set_twice(a, b), (b,))

# An address taken before the slot was filled reads what was put there.
eq('a slot filled after its address was taken', m.address_then_set(c), c)

print('cpyext-tuple-items-ok')
"#;

#[test]
fn the_item_array_a_tuple_mirror_hands_out() {
    let fixtures = Fixtures::new("cpyext-tuple-items");
    fixtures.compile("cpyext_tuple_items");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-tuple-items-ok");
}
