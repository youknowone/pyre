"""Hot property get/set plus Python 3.14 metadata surface."""


class Counter:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        "counter value"
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


def main():
    counter = Counter()
    total = 0
    i = 0
    while i < 200000:
        counter.value = i
        total += counter.value
        i += 1

    prop = Counter.__dict__["value"]
    print(total, counter.value)
    # PyPy 3.11 has not yet gained property.__name__; keep the synthetic
    # output version-neutral while the dedicated 3.14 parity test covers it.
    print(prop.fget.__name__, prop.__doc__)


main()
