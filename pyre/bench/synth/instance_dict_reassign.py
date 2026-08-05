# pyre-check: max-pypy-ratio=61
# pyre-check: min-pypy-ratio=4.82
N = 50000


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
