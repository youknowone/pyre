# CPython-suite gap: no suite test reads an attribute of a devolved instance of
# a class that also defines `__getattr__`, under a loop hot enough to specialize.
# parity-tests reason: this is a pyre JIT `__getattr__`-fold regression.

# The fold that replaces `obj.name` with the type's `__getattr__` proves the
# name is absent from the instance by asking `find_map_attr(name, DICT)` and
# taking None for absence.  mapdict.py:1534-1536 states that call "will always
# return None if attrkind==DICT" once the map is rooted at a
# `DevolvedDictTerminator`, so for a devolved instance the answer is the same
# whether or not the attribute is there.  Upstream's own case of pinning a map
# to cache a negative instance lookup — `LOAD_METHOD_mapdict_fill_cache_method`
# — refuses the shape outright (mapdict.py:1569).
#
# The map guard cannot stand in: the devolved terminator is a per-class
# singleton, so the pinned map word is identical for every devolved instance of
# the class and unchanged by a later attribute assignment.

N = 12000


class Hooked:
    def __getattr__(self, name):
        return 'hook'


def non_string_key_devolves():
    obj = Hooked()
    # A non-str `__dict__` key forces the object strategy at any attribute
    # count, without waiting for the attribute-count limit.
    obj.__dict__[1] = 'sentinel'
    obj.__dict__['present'] = 'real'
    seen = set()
    for _ in range(N):
        seen.add(obj.present)
    assert seen == {'real'}, 'devolved instance answered from the hook: %r' % (seen,)


def assignment_after_devolving_is_seen():
    obj = Hooked()
    obj.__dict__[1] = 'sentinel'
    seen = []
    for i in range(N):
        seen.append(obj.later)
        if i == N // 2:
            obj.later = 'assigned'
    assert seen[0] == 'hook', 'absent attribute did not reach the hook: %r' % (seen[0],)
    assert seen[-1] == 'assigned', (
        'assignment on a devolved instance was not seen: %r' % (seen[-1],)
    )


def hook_still_answers_a_real_miss():
    obj = Hooked()
    obj.__dict__[1] = 'sentinel'
    seen = set()
    for _ in range(N):
        seen.add(obj.missing)
    assert seen == {'hook'}, 'the decline swallowed the hook: %r' % (seen,)


non_string_key_devolves()
assignment_after_devolving_is_seen()
hook_still_answers_a_real_miss()
print("OK")
