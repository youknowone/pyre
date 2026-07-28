# pyre-check: max-pypy-ratio=5
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
