# CPython-suite gap: test_cprofile and test_profile only ever pass real bools
# to `enable`, so neither exercises the failing-conversion path, and the leak
# they would expose is process-wide rather than per-test.
# parity-tests reason: the tool id is a process singleton, so the damage shows
# up on a *later, unrelated* profiler; that cross-object effect is what this
# checks, on both backends, on every leg.

"""A failed `Profiler.enable` must not keep the profiler tool id.

`enable` takes `subcalls` and `builtins` through the index/bool protocol, so
an argument whose `__bool__` raises makes the call fail. The tool id is a
single process-wide slot: if the failed call has already claimed it and
nothing releases it, `disable` cannot help -- it is a no-op while the
profiler never became enabled -- and every later `enable`, on any profiler
object, reports that another tool is active.

The discriminator is therefore a *second, independent* profiler enabling
successfully after the first one's `enable` raised.
"""

import _lsprof


class Raises:
    def __bool__(self):
        raise ValueError("no truth value")


def main():
    first = _lsprof.Profiler()
    try:
        first.enable(Raises())
    except ValueError:
        pass
    else:
        raise AssertionError("enable() accepted an argument whose __bool__ raises")

    # The failed call must have left the tool id free.
    second = _lsprof.Profiler()
    second.enable()
    second.disable()

    # And the id must still be reusable after a clean enable/disable pair.
    third = _lsprof.Profiler()
    third.enable()
    third.disable()

    print("OK")


main()
