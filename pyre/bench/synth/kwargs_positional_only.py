# pyre-check: max-pypy-ratio=6
# argument.py _match_keywords: a positional-only parameter passed by keyword is
# absorbed into **kwargs when present, else raises; a keyword that duplicates an
# already-bound positional raises "multiple values".  Exercised through the
# **dict (CALL_FUNCTION_EX) binder.


def with_varkw(a, /, **kw):
    return (a, sorted(kw.items()))


def posonly_only(a, /):
    return a


def plain(b):
    return b


def call_kind(fn, *args, **kw):
    try:
        return ('ok', fn(*args, **kw))
    except TypeError:
        return ('TypeError',)


def main():
    # 'a' is positional-only with **kw: a=1 lands in **kw, leaving the
    # positional 'a' unfilled -> missing-argument TypeError.
    print('posonly_absorbed', call_kind(with_varkw, **{'a': 1})[0])
    # the surrogate-free **kw still binds a real positional + extra keyword.
    print('ok_call', call_kind(with_varkw, 5, **{'z': 9}))
    # 'a' is positional-only with no **kw: passing it by keyword errors.
    print('posonly_reject', call_kind(posonly_only, **{'a': 1})[0])
    # a keyword duplicating a bound positional errors.
    print('duplicate', call_kind(plain, 1, **{'b': 2})[0])


main()
