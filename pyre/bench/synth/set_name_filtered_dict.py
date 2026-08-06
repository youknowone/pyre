# pyre-check: max-pypy-ratio=14
# pyre-check: min-pypy-ratio=0.5
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest ratio the CI
# runners observe (6.8x), rounded up, and the floor is half the fastest
# (1.0x) — a derived floor of ceiling/5 would sit above it.
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
