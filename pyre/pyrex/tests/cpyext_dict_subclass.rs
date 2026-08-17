//! Every `PyDict_*` entry point, and the keyword mapping of `PyObject_Call`,
//! against a `dict` subclass.
//!
//! `PyDict_Check` is the gate they all apply, and it admits a subclass — so each
//! one has to reach the subclass's concrete mapping, and reach it without
//! consulting any hook the subclass overrides.  In pyre a `dict` subclass
//! instance is not a dict but an object holding one, so "the mapping" is a
//! resolution step rather than the argument itself; a `list` subclass, by
//! contrast, is a `W_ListObject` and needs none of this.
//!
//! The subclass under test raises from every hook a concrete read must not go
//! through, so a wrong dispatch fails loudly instead of quietly agreeing.  Each
//! operation reports the outcome that follows it rather than whether it raised:
//! `PyDict_Clear` on a rejected argument sets no error and simply does nothing,
//! which a raised-or-not check reads as success.
//!
//! Every expectation here was taken from CPython 3.14.6 running this same
//! script against this same fixture.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SUBCLASS_SCRIPT: &str = r#"
import collections
import types

import cpyext_dict_subclass as m

class Plain(dict):
    pass

class Loud(dict):
    """Raises from every hook a concrete read must not consult.

    `__iter__` is left alone deliberately: `PyDict_Merge` is specified to
    consult it when choosing how to read its source, so overriding it here
    would make one correct answer look like a defect.  `Iterating` covers that
    on its own.
    """
    def __len__(self): raise AssertionError('__len__ consulted')
    def __getitem__(self, key): raise AssertionError('__getitem__ consulted')
    def __setitem__(self, key, value): raise AssertionError('__setitem__ consulted')
    def __delitem__(self, key): raise AssertionError('__delitem__ consulted')
    def __contains__(self, key): raise AssertionError('__contains__ consulted')
    def keys(self): raise AssertionError('keys consulted')
    def values(self): raise AssertionError('values consulted')
    def items(self): raise AssertionError('items consulted')
    def copy(self): raise AssertionError('copy consulted')
    def clear(self): raise AssertionError('clear consulted')
    def update(self, *args, **kwargs): raise AssertionError('update consulted')

class Iterating(dict):
    """Iterates its own way, so a merge has to read it through keys()."""
    def __iter__(self): return iter(['b'])
    def keys(self): return ['b']

class Missing(dict):
    def __missing__(self, key): raise AssertionError('__missing__ consulted')

def take(**kwargs):
    return kwargs

def one(cls=dict):
    return cls({'k': 'value'})

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

# ── the predicates ─────────────────────────────────────────────────────
for tag, value in (('dict', {}), ('subclass', Plain()), ('loud', Loud())):
    eq('check(%s)' % tag, m.check(value), True)
    eq('check_exact(%s)' % tag, m.check_exact(value), tag == 'dict')
    eq('mapping_check(%s)' % tag, m.mapping_check(value), True)
    # A mapping is not a sequence, and that has to hold for a subclass too.
    eq('sequence_check(%s)' % tag, m.sequence_check(value), False)
    eq('iter_check(%s)' % tag, m.iter_check(value), False)

eq('check(mappingproxy)', m.check(types.MappingProxyType({})), False)
eq('check(list)', m.check([]), False)
eq('check(OrderedDict)', m.check(collections.OrderedDict()), True)
eq('sequence_check(list)', m.sequence_check([]), True)
eq('iter_check(iter(dict))', m.iter_check(iter({})), True)

# ── reads reach the mapping without going through an override ──────────
for tag, cls in (('dict', dict), ('subclass', Plain), ('loud', Loud)):
    eq('size(%s)' % tag, m.size(one(cls)), 1)
    eq('getitem(%s)' % tag, m.getitem(one(cls)), 'value')
    eq('getitem_string(%s)' % tag, m.getitem_string(one(cls)), 'value')
    eq('getitem_ref(%s)' % tag, m.getitem_ref(one(cls)), 'value')
    eq('getitem_string_ref(%s)' % tag, m.getitem_string_ref(one(cls)), 'value')
    eq('getitem_with_error(%s)' % tag, m.getitem_with_error(one(cls)), 'value')
    eq('contains(%s)' % tag, m.contains(one(cls)), True)
    eq('keys(%s)' % tag, m.keys(one(cls)), ['k'])
    eq('values(%s)' % tag, m.values(one(cls)), ['value'])
    eq('items(%s)' % tag, m.items(one(cls)), [('k', 'value')])
    eq('next_items(%s)' % tag, m.next_items(one(cls)), [('k', 'value')])
    copied = m.copy(one(cls))
    eq('copy(%s)' % tag, copied, {'k': 'value'})
    # The copy is an exact dict whatever the source was.
    eq('type(copy(%s))' % tag, type(copied) is dict, True)

# `__missing__` hangs off `__getitem__`, which a concrete read does not go
# through, so an absent key reads as absent.
eq('getitem(absent, __missing__ defined)', m.getitem(Missing()), 'getitem-missing')

# ── writes land in the mapping, and Python sees them ───────────────────
for tag, cls in (('dict', dict), ('subclass', Plain), ('loud', Loud)):
    eq('clear_then_size(%s)' % tag, m.clear_then_size(one(cls)), 0)
    eq('setitem_then_read(%s)' % tag, m.setitem_then_read(one(cls)), None)
    eq('setitem_string_then_read(%s)' % tag, m.setitem_string_then_read(one(cls)), None)
    eq('delitem_then_contains(%s)' % tag, m.delitem_then_contains(one(cls)), False)
    eq('delitem_string_then_contains(%s)' % tag,
       m.delitem_string_then_contains(one(cls)), False)

# The write has to reach the object Python holds, not a copy of it.
target = one(Plain)
m.setitem_then_read(target)
eq('write is visible to Python', sorted(dict.items(target)),
   [('added', None), ('k', 'value')])
eq('write changes the length', dict.__len__(target), 2)

cleared = one(Plain)
m.clear_then_size(cleared)
eq('clear is visible to Python', dict.__len__(cleared), 0)

# ── merge, in both directions ──────────────────────────────────────────
eq('merge_from(dict)', m.merge_from({'a': 1}), {'a': 1})
eq('merge_from(subclass)', m.merge_from(Plain({'a': 1})), {'a': 1})
eq('merge_from(loud)', m.merge_from(Loud({'a': 1})), {'a': 1})
eq('update_from(subclass)', m.update_from(Plain({'a': 1})), {'a': 1})
eq('merge_from(OrderedDict)', m.merge_from(collections.OrderedDict(a=1)), {'a': 1})
eq('merge_from_seq2', m.merge_from_seq2([('a', 1)]), {'a': 1})
# A source that iterates its own way is read through keys(), so the override is
# what decides the membership -- 'a' is in the mapping and does not arrive.
eq('merge_from(iterating)', m.merge_from(Iterating({'a': 1, 'b': 2})), {'b': 2})

into = Plain({'a': 'kept'})
m.merge_into(into, {'a': 'new', 'b': 2}, 0)
eq('merge_into keeps under override=0', sorted(dict.items(into)),
   [('a', 'kept'), ('b', 2)])
m.merge_into(into, {'a': 'new'}, 1)
eq('merge_into replaces under override=1', sorted(dict.items(into)),
   [('a', 'new'), ('b', 2)])

# ── the keyword mapping of a call ──────────────────────────────────────
eq('call_kwargs(dict)', m.call_kwargs(take, {'a': 1}), {'a': 1})
eq('call_kwargs(subclass)', m.call_kwargs(take, Plain({'a': 1})), {'a': 1})
eq('call_kwargs(loud)', m.call_kwargs(take, Loud({'a': 1})), {'a': 1})

print('cpyext-dict-subclass-ok')
"#;

const REFUSED_SCRIPT: &str = r#"
import types

import cpyext_dict_subclass as m

def take(**kwargs):
    return kwargs

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

# What is not a dict is refused, and refused the way each entry point is
# specified to: `PyDict_GetItem` cannot fail, so it answers NULL with no error;
# `PyDict_Clear` returns having done nothing and sets none either.
for tag, value in (('list', []),
                   ('mappingproxy', types.MappingProxyType({'k': 'value'})),
                   ('None', None),
                   ('str', 'k')):
    eq('size(%s)' % tag, m.size(value), -1)
    eq('getitem(%s)' % tag, m.getitem(value), 'getitem-missing')
    eq('keys(%s)' % tag, m.keys(value), 'keys-failed')
    eq('clear_then_size(%s)' % tag, m.clear_then_size(value), 'clear-size-failed')

eq('merge_from(list)', m.merge_from([1, 2]), 'merge-failed')

# A keyword mapping that is not a dict is refused rather than read.  CPython
# enforces that argument with an assert, so a release build there reads the
# object as a dict anyway and the outcome is undefined -- there is nothing to
# match, and a `TypeError` is what the argument can be given.
eq('call_kwargs(list)', m.call_kwargs(take, []), 'call-failed')

print('cpyext-dict-refused-ok')
"#;

#[test]
fn every_dict_entry_point_reaches_a_subclasss_mapping() {
    let fixtures = Fixtures::new("cpyext-dict-subclass");
    fixtures.compile("cpyext_dict_subclass");
    fixtures.expect_ok(SUBCLASS_SCRIPT, &[], "cpyext-dict-subclass-ok");
}

#[test]
fn a_dict_entry_point_refuses_what_is_not_a_dict() {
    let fixtures = Fixtures::new("cpyext-dict-refused");
    fixtures.compile("cpyext_dict_subclass");
    fixtures.expect_ok(REFUSED_SCRIPT, &[], "cpyext-dict-refused-ok");
}
