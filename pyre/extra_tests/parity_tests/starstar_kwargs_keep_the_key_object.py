# CPython-suite gap: `test_extcall` covers the shapes a `**` unpack accepts and
# the errors a bad mapping raises, but every key it passes is an exact `str`, so
# nothing there notices an implementation that rebuilds the key.  `test_unicode`
# and `test_userstring` exercise `str` subclasses as values and as dict keys,
# never as the name half of a `**` mapping.
#
# parity-tests reason: `Arguments._collect_keyword_args` forwards the key the
# mapping supplied -- `space.setitem(w_kwds, w_key, keywords_w[i])` with the
# object out of `keyword_names_w` -- so a subclass instance reaches the callee's
# `**kwargs` dict unchanged, keeps its own `__repr__`, and compares by the
# subclass's rules.  An implementation that carries keyword names as plain text
# and re-wraps them loses that identity, and the loss is invisible to a timing
# run and to every arm that only reads the name.  The error message an unknown
# keyword produces reads the same object, so it is pinned here too.
#
# CPython 3.14 and PyPy agree on every arm below.
class S(str):
    def __repr__(self):
        return "S(%s)" % str.__repr__(self)


def a_star_star_key_reaches_kwargs_unchanged():
    def f(**kw):
        return kw

    kw = f(**{S("a"): 1})
    assert list(map(type, kw)) == [S], list(map(type, kw))
    assert list(map(repr, kw)) == ["S('a')"], list(map(repr, kw))
    assert kw == {"a": 1}, kw


def a_matched_name_is_bound_and_the_rest_keep_their_keys():
    def g(a, **kw):
        return a, kw

    a, kw = g(**{S("a"): 1, S("b"): 2})
    assert a == 1, a
    assert list(map(type, kw)) == [S], list(map(type, kw))


def the_keys_survive_a_sort():
    def f(**kw):
        return kw

    kw = f(**{S("z"): 1, S("a"): 2})
    assert list(map(repr, sorted(kw))) == ["S('a')", "S('z')"], sorted(kw)


def a_literal_keyword_is_an_exact_str():
    def f(**kw):
        return kw

    kw = f(a=1)
    assert list(map(type, kw)) == [str], list(map(type, kw))


def an_unknown_keyword_names_the_key_it_was_given():
    def h(a):
        return a

    try:
        h(**{S("b"): 1})
    except TypeError as exc:
        assert "'b'" in str(exc), str(exc)
    else:
        raise AssertionError("expected TypeError")


def a_duplicate_keyword_is_still_a_type_error():
    def f(a, **kw):
        return a, kw

    try:
        f(1, **{S("a"): 2})
    except TypeError as exc:
        assert "'a'" in str(exc), str(exc)
    else:
        raise AssertionError("expected TypeError")


def every_dispatch_arm_keeps_the_key():
    # The call surface reaches a callee through several arms, and each one
    # rebuilds its argument view before forwarding.  An arm that carries the
    # name half as text rather than as the object loses the key on its own,
    # whatever the arms around it do, so every arm is pinned separately.
    def types_of(kw):
        return list(map(type, kw))

    class WithInit:
        def __init__(self, **kw):
            self.kw = kw

    class WithNew:
        def __new__(cls, **kw):
            self = object.__new__(cls)
            self.kw = kw
            return self

    class Meta(type):
        def __call__(cls, **kw):
            return kw

    class ViaMeta(metaclass=Meta):
        pass

    class Callable:
        def __call__(self, **kw):
            return kw

    class Holder:
        def bound(self, **kw):
            return kw

        @staticmethod
        def static(**kw):
            return kw

        @classmethod
        def classy(cls, **kw):
            return kw

    def plain(**kw):
        return kw

    arms = [
        ("__init__", lambda: WithInit(**{S("a"): 1}).kw),
        ("type.__call__", lambda: type.__call__(WithInit, **{S("a"): 1}).kw),
        ("__new__", lambda: WithNew(**{S("a"): 1}).kw),
        ("metaclass __call__", lambda: ViaMeta(**{S("a"): 1})),
        ("instance __call__", lambda: Callable()(**{S("a"): 1})),
        ("bound method", lambda: Holder().bound(**{S("a"): 1})),
        ("staticmethod", lambda: Holder.static(**{S("a"): 1})),
        ("classmethod", lambda: Holder.classy(**{S("a"): 1})),
        ("builtin dict", lambda: dict(**{S("a"): 1})),
        ("forwarded twice", lambda: plain(**plain(**{S("a"): 1}))),
    ]
    for name, call in arms:
        assert types_of(call()) == [S], (name, types_of(call()))


a_star_star_key_reaches_kwargs_unchanged()
a_matched_name_is_bound_and_the_rest_keep_their_keys()
the_keys_survive_a_sort()
a_literal_keyword_is_an_exact_str()
an_unknown_keyword_names_the_key_it_was_given()
a_duplicate_keyword_is_still_a_type_error()
every_dispatch_arm_keeps_the_key()
print('OK')
