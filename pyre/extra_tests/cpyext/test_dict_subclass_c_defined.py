# cpyext-fixture: cpyext_dict_subclass
# cpyext-expect: cpyext-c-dict-subclass-ok

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

import cpyext_dict_subclass as m

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

def one(cls):
    return cls({'k': 'value'})

# A `dict` subclass whose type was built from C rather than by `class`.  Its
# instances are dicts, so the `dict` methods take them and the entry points
# above reach the same mapping -- the two routes that build a type have to
# arrive at the same instance layout.  A Python class under it is the shape
# that decides whether the reservation is made once or twice.
class Under(m.CDict):
    pass

for tag, cls in (('CDict', m.CDict), ('under-CDict', Under)):
    empty = cls()
    eq('%s: repr-empty' % tag, dict.__repr__(empty), '{}')
    eq('%s: len-empty' % tag, dict.__len__(empty), 0)
    empty['k'] = 'value'
    eq('%s: subscript' % tag, empty['k'], 'value')
    eq('%s: repr' % tag, dict.__repr__(empty), "{'k': 'value'}")

    eq('%s: isinstance' % tag, isinstance(one(cls), dict), True)
    eq('%s: check' % tag, m.check(one(cls)), True)
    eq('%s: check_exact' % tag, m.check_exact(one(cls)), False)
    eq('%s: size' % tag, m.size(one(cls)), 1)
    eq('%s: getitem' % tag, m.getitem(one(cls)), 'value')
    eq('%s: getitem_string' % tag, m.getitem_string(one(cls)), 'value')
    eq('%s: contains' % tag, m.contains(one(cls)), True)
    eq('%s: keys' % tag, m.keys(one(cls)), ['k'])
    eq('%s: items' % tag, m.items(one(cls)), [('k', 'value')])
    eq('%s: copy' % tag, m.copy(one(cls)), {'k': 'value'})
    eq('%s: clear_then_size' % tag, m.clear_then_size(one(cls)), 0)

    # The write reaches the object Python holds.
    target = one(cls)
    m.setitem_then_read(target)
    eq('%s: write is visible' % tag, sorted(dict.items(target)),
       [('added', None), ('k', 'value')])

    # `dict.__init__` is what a C base's `tp_init` runs, and it refuses a
    # receiver whose mapping it cannot reach.
    fresh = cls()
    dict.__init__(fresh, {'k': 'value'})
    eq('%s: after __init__' % tag, dict.__repr__(fresh), "{'k': 'value'}")

# The mapping belongs to the instance, not to the type.
first, second = m.CDict(), m.CDict()
first['k'] = 'value'
eq('one instance per mapping', dict.__len__(second), 0)

print('cpyext-c-dict-subclass-ok')
