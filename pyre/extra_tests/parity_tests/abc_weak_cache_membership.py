# CPython-suite gap: `test_abc` asks membership only through classes whose
# caches it never inspects, so it covers neither the empty-collection answer,
# nor an item that cannot carry a weakref, nor a class whose hash raises, and
# it cannot cover a replaced collection at all -- the reference build keeps the
# three collections in the type's own struct, so there is no attribute there to
# rebind.
#
# `_abc_instancecheck` answers membership without entering
# `SimpleWeakSet.__contains__`, which is sound only while the receiver really
# is the collection `_abc_init` installed and its `data` really is a `set`.
# Anything else must take the membership protocol and answer whatever the
# replacement spells -- including a subclass whose `__contains__` lies.
#
# parity-tests reason: every arm is a silent wrong answer or a doubled side
# effect rather than a crash.  A collection read past its own override reports
# a class as registered when it is not, and a probe that answers and then
# re-asks evaluates an observable `__hash__` twice.
#
# The last three arms name a collection this implementation spells as a class
# attribute; where it is not spellable there is nothing to replace and the arm
# does not apply.
import _abc
import gc
import weakref
from abc import ABCMeta


class Plain:
    pass


calls = []


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


def a_raising_hash_runs_once():
    class Boom(type):
        def __hash__(cls):
            calls.append(cls)
            raise ValueError('boom')

    class Fused(ABCMeta, Boom):
        pass

    class Warm(metaclass=ABCMeta):
        pass

    class Member(Warm):
        pass

    class Unhashable(metaclass=Fused):
        pass

    class Claims:
        __class__ = Unhashable

    # A populated collection, so the probe is really built and really hashed.
    assert isinstance(Member(), Warm) is True
    del calls[:]
    try:
        isinstance(Claims(), Warm)
    except ValueError:
        pass
    else:
        raise AssertionError('the raising hash was swallowed')
    # The membership expression evaluates the hash once.  Answering the probe
    # and then re-asking through the protocol would evaluate it twice, which is
    # observable here and in any hash carrying a side effect.
    assert len(calls) == 1, 'hash ran %d times' % (len(calls),)


def a_raising_hash_names_a_set_element():
    class Unhashable(ABCMeta):
        __hash__ = None

    class Warm(metaclass=ABCMeta):
        pass

    class Member(Warm):
        pass

    class Refuses(metaclass=Unhashable):
        pass

    class Claims:
        __class__ = Refuses

    # A populated collection, so the probe is really built and really hashed.
    assert isinstance(Member(), Warm) is True
    try:
        isinstance(Claims(), Warm)
    except TypeError as exc:
        message = str(exc)
    else:
        raise AssertionError('an unhashable class was accepted')
    # The collection is a set and the failure names the operand as one.  The
    # dict spelling of the same recovery renames it, which is what a reader of
    # the message sees.
    assert 'as a set element' in message, message


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


def a_data_subclass_keeps_its_override():
    class Shadowed(metaclass=ABCMeta):
        pass

    class Lying(set):
        def __contains__(self, item):
            return True

    # An empty table whose `__contains__` claims every member.  Reading the
    # table would answer False; the override answers True, and the override is
    # what the membership protocol reaches -- a subclass keeps the base layout,
    # so a layout test alone does not tell the two apart.
    Shadowed._abc_cache.data = Lying()
    assert isinstance(Plain(), Shadowed) is True, 'a data subclass was read past'


def a_mutated_contains_body_is_consulted():
    class Mutated(metaclass=ABCMeta):
        pass

    class Impl(Mutated):
        pass

    assert isinstance(Impl(), Mutated) is True

    collection = type(Mutated._abc_cache)
    original = collection.__contains__.__code__

    def claims_everything(self, item):
        return True

    # Assigning `__code__` leaves the same function installed on the same
    # class, so an identity test over the method alone still matches while the
    # body it runs no longer does.
    collection.__contains__.__code__ = claims_everything.__code__
    try:
        assert isinstance(Plain(), Mutated) is True, 'the mutated body was read past'
    finally:
        collection.__contains__.__code__ = original


def a_non_function_contains_declines():
    class Guarded(metaclass=ABCMeta):
        pass

    class Impl(Guarded):
        pass

    assert isinstance(Impl(), Guarded) is True

    collection = type(Guarded._abc_cache)
    original = collection.__contains__

    class Descriptor:
        def __get__(self, obj, objtype=None):
            return lambda item: True

    # Not a function at all.  Reading the code object off the installed method
    # is only defined for a function, so the shortcut has to reject this before
    # the read rather than by comparing whatever the read returned.
    collection.__contains__ = Descriptor()
    try:
        assert isinstance(Plain(), Guarded) is True, 'a non-function __contains__ was read past'
    finally:
        collection.__contains__ = original


def a_mutated_ref_global_is_consulted():
    class Watched(metaclass=ABCMeta):
        pass

    class Impl(Watched):
        pass

    assert isinstance(Impl(), Watched) is True

    collection = type(Watched._abc_cache)
    namespace = collection.__contains__.__globals__
    original = namespace['ref']
    calls = []

    def watching_ref(item, *rest):
        calls.append(item)
        return original(item, *rest)

    # `ref` is the whole free-variable surface of the app-level body, and
    # rebinding it moves neither the method nor its code object.
    namespace['ref'] = watching_ref
    try:
        isinstance(Impl(), Watched)
        assert calls, 'the rebound `ref` global was not consulted'
    finally:
        namespace['ref'] = original


empty_collection_answers_then_fills()
unweakreferenceable_class_reaches_the_subclass_check()
a_raising_hash_runs_once()
a_raising_hash_names_a_set_element()
a_collected_entry_stops_matching()
if hasattr(ABCMeta('_Probe', (), {}), '_abc_cache'):
    rebound_collection_is_consulted()
    replaced_data_is_consulted()
    a_data_subclass_keeps_its_override()
    a_mutated_contains_body_is_consulted()
    a_non_function_contains_declines()
    a_mutated_ref_global_is_consulted()
print('OK')
