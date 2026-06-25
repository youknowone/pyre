# Bare `raise` (RERAISE) inside a handler re-raises the exception currently
# being handled, which an outer handler then catches. This exercises the
# current-exception slot on the re-raise path: the bare raise must pick up the
# active exception from the slot, and after the inner handler unwinds the slot
# must restore to the outer/None prev. Alternating the re-raise by iteration
# parity keeps both the re-raised and the locally-finished paths hot.
#   even i -> bare re-raise -> outer except (reraised++)
#   odd  i -> handled locally               (caught++)
# With N=30000: caught=15000, reraised=15000. Deterministic.
N = 30000


def run(n):
    caught = 0
    reraised = 0
    i = 0
    while i < n:
        try:
            try:
                raise ValueError("x")
            except ValueError:
                if (i & 1) == 0:
                    raise
                caught += 1
        except ValueError:
            reraised += 1
        i += 1
    return caught, reraised


def main():
    c, r = run(N)
    print(c, r)


main()
