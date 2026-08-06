# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# argument.py:106-150 _combine_starstarargs_wrapped: CALL_FUNCTION_EX accepts
# any mapping via keys()/__getitem__, rejects a non-str key with TypeError, and
# rejects a non-mapping after ** with TypeError.


class Mapping:
    def keys(self):
        return ['x', 'y']

    def __getitem__(self, k):
        return {'x': 1, 'y': 2}[k]


def f(**kw):
    return sorted(kw.items())


def attempt(maker):
    try:
        return ('ok', maker())
    except TypeError:
        return ('TypeError',)


def main():
    # arbitrary mapping is unpacked via keys() + __getitem__
    print('mapping', attempt(lambda: f(**Mapping())))
    # a plain dict still works
    print('dict', attempt(lambda: f(**{'a': 10})))
    # a non-str key is rejected
    print('nonstr', attempt(lambda: f(**{1: 2}))[0])
    # a non-mapping after ** is rejected
    print('nonmapping', attempt(lambda: f(**[('a', 1)]))[0])


main()
