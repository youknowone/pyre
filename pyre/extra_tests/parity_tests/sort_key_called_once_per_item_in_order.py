# CPython-suite gap: test_sort checks the sorted result, not that the key
# function ran once per item in list order, including an empty list and a
# key that mutates the receiver.
# parity-tests reason: `_compute_keys_for_sorting` must call the key once
# per item in list order; a key that mutates the list raises ValueError
# "list modified during sort".

"""A sort key is called once per item in list order."""


def check(n):
    seen = []

    def key(v):
        seen.append(v)
        return v

    xs = list(range(n))
    xs.sort(key=key)
    assert seen == list(range(n)), (n, seen)
    assert xs == list(range(n)), (n, xs)


for n in (0, 1, 2, 5000):
    check(n)

xs = [3, 1, 2]


def mut(v):
    xs.append(0)
    return v


try:
    xs.sort(key=mut)
except ValueError as exc:
    assert str(exc) == "list modified during sort", str(exc)
else:
    raise AssertionError("expected ValueError")

print("OK")
