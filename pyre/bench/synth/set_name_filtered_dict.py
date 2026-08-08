# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest ratio the CI
# runners observe (6.8x), rounded up.
# typeobject.py:1006 _set_names — type.__new__ calls __set_name__(owner, name)
# for each descriptor in the type's final __dict__.  Each descriptor is visited
# once, with the class as owner and its own attribute name.


class Tracker:
    def __init__(self, tag):
        self.tag = tag

    def __set_name__(self, owner, name):
        events.append((owner.__name__, name, self.tag))


events = []


class C:
    first = Tracker('a')
    second = Tracker('b')

    def __init__(self):
        # a zero-arg super() forces a __classcell__ entry into the class body,
        # which must NOT be visited by __set_name__.
        super().__init__()


def main():
    print('events', sorted(events))


main()
