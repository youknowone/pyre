# pyre-check: max-pypy-ratio=61
# Trip count kept clear of the major-collection threshold check.py pins: at
# the previous 410000 this loop crossed it, and the eval-breaker bailout that
# follows re-enters through a bridge whose guard can then fail once more,
# which moves guard_failures for reasons outside this fixture. Crossing
# resumes between 0.9x and 1.0x of the old count; gated counters unchanged.
N = 205000


class A:
    pass


def main():
    total = 0
    i = 0
    while i < N:
        a = A()
        a.x = i
        a.y = i + 1
        old = a.__dict__               # materialize the instance-backed view
        a.__dict__ = {"z": i + 2}      # reassign -> old must detach to a snapshot
        a.w = i + 3                    # mutate the instance after the reassign
        # `old` is an independent snapshot of the pre-reassign attributes,
        # so it keeps {x, y} and never sees the later `w` store.
        total = total + old["x"] + old["y"]
        total = total + len(old)
        total = total + a.z
        if "w" in old:
            total = total + 1000000
        i = i + 1
    print(total)


main()
