# CPython-suite gap: `test_abc` asks membership only through classes whose
# caches it never inspects, so it covers neither the empty-collection answer
# nor an item that cannot carry a weakref, and it cannot cover a replaced
# collection at all -- the reference build keeps the three collections in the
# type's own struct, so there is no attribute there to rebind.
#
# `_abc_instancecheck` answers membership without entering
# `SimpleWeakSet.__contains__`, which is sound only while the receiver really
# is the collection `_abc_init` installed.  A rebound cache, or one whose
# `data` is no longer a set, must take the membership protocol and answer
# whatever the replacement spells.
#
# parity-tests reason: every arm is a silent wrong answer rather than a crash.
# An empty collection that answered "present", or a replaced one read as if it
# were still the original, reports a class as registered when it is not, and
# nothing downstream would notice.
#
# The last two arms name a collection this implementation spells as a class
# attribute; where it is not spellable there is nothing to replace and the arm
# does not apply.
import _abc
import gc
import weakref
from abc import ABCMeta


class Plain:
    pass


def empty_collection_answers_then_fills():
    class Fresh(metaclass=ABCMeta):
        pass

    class Late(Fresh):
        pass

    obj = Late()
    # The first question is asked against an empty collection, which holds no
    # entry to match; the same question after the walk recorded one must flip.
    assert isinstance(Plain(), Fresh) is False
    assert isinstance(obj, Fresh) is True
    assert isinstance(obj, Fresh) is True, 'the recorded entry was not found'


def unweakreferenceable_class_reaches_the_subclass_check():
    class Target(metaclass=ABCMeta):
        pass

    class Impl(Target):
        pass

    class Claims:
        __class__ = 3

    # An empty collection and a populated one reach the probe differently, and
    # neither may turn the missing weakref into an answer of its own: the
    # membership tests report "absent" and the subclass check that follows is
    # what raises.
    for _ in range(2):
        try:
            isinstance(Claims(), Target)
        except TypeError:
            pass
        else:
            raise AssertionError('a non-class __class__ must reach the check')
        assert isinstance(Impl(), Target) is True


def a_collected_entry_stops_matching():
    class Held(metaclass=ABCMeta):
        pass

    class Doomed(Held):
        pass

    assert isinstance(Doomed(), Held) is True
    entries = _abc._get_dump(Held)[1]
    assert len(entries) == 1, 'the hit was not recorded: %r' % (entries,)
    del Doomed
    gc.collect()
    live = [ref for ref in _abc._get_dump(Held)[1] if ref() is not None]
    assert not live, 'a dead entry stayed in the cache: %r' % (live,)


def rebound_collection_is_consulted():
    class Asked(metaclass=ABCMeta):
        pass

    class Liar:
        def __init__(self):
            self.asked = []

        def __contains__(self, item):
            self.asked.append(item)
            return True

        def add(self, item):
            raise AssertionError('a hit records nothing')

    cache = Liar()
    Asked._abc_cache = cache
    assert isinstance(Plain(), Asked) is True, 'the replacement was bypassed'
    assert cache.asked, 'the replacement was never asked'


def replaced_data_is_consulted():
    class Swapped(metaclass=ABCMeta):
        pass

    class Impl(Swapped):
        pass

    obj = Impl()
    assert isinstance(obj, Swapped) is True
    # A list answers `in` by a linear scan rather than by a hashed probe, and a
    # callback-less weakref compares equal to the callback-carrying one the hit
    # above stored, so the entry is still found -- by the other route.
    Swapped._abc_cache.data = [weakref.ref(Impl)]
    assert isinstance(obj, Swapped) is True, 'a replaced data must still answer'


empty_collection_answers_then_fills()
unweakreferenceable_class_reaches_the_subclass_check()
a_collected_entry_stops_matching()
if hasattr(ABCMeta('_Probe', (), {}), '_abc_cache'):
    rebound_collection_is_consulted()
    replaced_data_is_consulted()
print('OK')
