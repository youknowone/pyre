# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=run
# A closure over the first argument replaces fast-local slot zero with a Cell.
# Zero-argument super() must read that live Cell.contents in the traced portal
# frame; treating the cell object itself as the receiver either misbinds the
# proxy or leaves the whole lookup residual.
N = 20000


class Base:
    value = 3


class Child(Base):
    def control(self, n):
        def capture():
            return self

        total = 0
        for _ in range(n):
            total += 3
        return total, capture() is self

    def run(self, n):
        def capture():
            return self

        total = 0
        for _ in range(n):
            proxy = super()
            total += proxy.value
        return total, capture() is self


def main():
    child = Child()
    control, control_identity_ok = child.control(N)
    got, identity_ok = child.run(N)
    if control != 3 * N or got != 3 * N or not control_identity_ok or not identity_ok:
        print("FAIL", control, got, control_identity_ok, identity_ok)
        return 1
    print("PASS", got)
    return 0


raise SystemExit(main())
