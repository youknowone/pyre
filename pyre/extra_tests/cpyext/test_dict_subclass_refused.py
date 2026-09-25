# cpyext-fixture: cpyext_dict_subclass
# cpyext-expect: cpyext-dict-refused-ok

# Every `PyDict_*` entry point, and the keyword mapping of `PyObject_Call`,
# against a `dict` subclass.
#
# `PyDict_Check` is the gate they all apply, and it admits a subclass — so each
# one has to reach the subclass's concrete mapping, and reach it without
# consulting any hook the subclass overrides.  In pyre a `dict` subclass
# instance is not a dict but an object holding one, so "the mapping" is a
# resolution step rather than the argument itself; a `list` subclass, by
# contrast, is a `W_ListObject` and needs none of this.
#
# The subclass under test raises from every hook a concrete read must not go
# through, so a wrong dispatch fails loudly instead of quietly agreeing.  Each
# operation reports the outcome that follows it rather than whether it raised:
# `PyDict_Clear` on a rejected argument sets no error and simply does nothing,
# which a raised-or-not check reads as success.
#
# Every expectation here was taken from CPython 3.14.6 running this same
# script against this same fixture.

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
