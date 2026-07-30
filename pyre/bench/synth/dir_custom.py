# pyre-check: max-pypy-ratio=12
# app_inspect.py:57-62 — dir(obj) is driven by a custom __dir__; its result is
# returned sorted.  An object without one still enumerates its real attributes.


class WithDir:
    def __dir__(self):
        return ['banana', 'apple', 'cherry']


class Normal:
    def __init__(self):
        self.z = 1


def main():
    # the custom __dir__ result is returned, sorted
    print('custom', dir(WithDir()))
    # a normal object still lists its instance attribute
    print('normal_has_z', 'z' in dir(Normal()))


main()
