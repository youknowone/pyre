# CPython-suite gap: no suite test asks a module's own helpers what
# `__module__` they report.  `test_atexit`, `test_ordered_dict`,
# `test_hashlib` and `test_contextvars` all exercise what those objects *do*,
# and each name here is reached through its module, so a wrong owner never
# changes an answer any of them checks.  The name only surfaces when something
# asks the object where it came from -- a traceback header, a `repr`, `help`,
# or `pickle` resolving a reference.
#
# The subject is where the answer comes from.  A class reads `__module__` off
# the globals it is defined in, through a name lookup that reaches the
# builtins module when its own namespace has none -- so a namespace with no
# `__name__` makes every class in it claim `builtins`.  A function reads the
# same entry with a plain dict lookup instead, so the same namespace leaves
# it with `None`.  Both are silent: the object is built, and only its account
# of itself is wrong.
#
# parity-tests reason: several of these are written in Python here and in C on
# CPython, and the namespace such a source runs in is the runtime's own
# construction rather than a real module's dict.  That construction is what
# has to bind the name, and nothing else in this suite would notice if it
# stopped: pickling by reference is the one arm that fails loudly, and it
# fails only for a subset.  CPython and PyPy agree on every arm below.
import pickle
import sys

import atexit
import collections
import contextvars
import _contextvars
import hashlib
import typing
import _io


def a_nameless_namespace_answers_two_different_ways():
    namespace = {}
    exec('class Cls: pass\ndef fn(): pass\n', namespace)

    # The two halves of what an unnamed namespace costs.  Neither raises, and
    # a runtime that builds its own module sources this way inherits both.
    assert namespace['Cls'].__module__ == 'builtins', namespace['Cls'].__module__
    assert namespace['fn'].__module__ is None, namespace['fn'].__module__


def a_native_modules_own_helpers_name_it():
    # `atexit` is a C module whose callbacks are pure Python here.  Whichever
    # half a name is written in, it belongs to `atexit`.
    for helper in (atexit.register, atexit.unregister, atexit._ncallbacks):
        assert helper.__module__ == 'atexit', (helper, helper.__module__)


def an_accelerated_class_names_the_module_it_is_reached_through():
    # Each of these is a class an accelerator module defines and a public
    # module re-exports, so the owner it reports is the one a reader can
    # import it from -- not always the one whose source defines it.
    expected = [
        (collections.OrderedDict, 'collections'),
        (collections.defaultdict, 'collections'),
        (hashlib.blake2b, '_blake2'),
        (contextvars.ContextVar, '_contextvars'),
        (contextvars.Context, '_contextvars'),
        (typing.TypeVar, 'typing'),
        (_io.IncrementalNewlineDecoder, '_io'),
    ]
    for cls, owner in expected:
        assert cls.__module__ == owner, (cls, cls.__module__, owner)


def the_reported_owner_is_the_one_pickle_resolves():
    # The arm that makes a wrong owner an error rather than a cosmetic slip.
    # A class and a function both pickle as a reference -- module name plus
    # qualified name -- so the owner has to be a module the loader can import
    # and find the object in.  `None` and `builtins` are each unusable, and
    # the failure lands on the caller pickling an ordinary object.
    for obj in (
        atexit.register,
        atexit.unregister,
        collections.OrderedDict,
        collections.defaultdict,
        hashlib.blake2b,
        contextvars.ContextVar,
        contextvars.Context,
        typing.TypeVar,
        _io.IncrementalNewlineDecoder,
    ):
        restored = pickle.loads(pickle.dumps(obj))
        assert restored is obj, (obj, restored)
        module = sys.modules[obj.__module__]
        assert getattr(module, obj.__name__) is obj, (obj, module)

    assert contextvars.Context is _contextvars.Context
    assert repr(contextvars.Context) == "<class '_contextvars.Context'>"
    assert repr(contextvars.copy_context()).startswith(
        '<_contextvars.Context object at 0x'
    )


a_nameless_namespace_answers_two_different_ways()
a_native_modules_own_helpers_name_it()
an_accelerated_class_names_the_module_it_is_reached_through()
the_reported_owner_is_the_one_pickle_resolves()
print('OK')
