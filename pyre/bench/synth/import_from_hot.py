# pyre-check: max-pypy-ratio=180
# IMPORT_FROM in a hot loop: `from math import pi, e` runs IMPORT_FROM for
# each imported name every iteration.  The compiled per-CodeObject jitcode
# walks the import_from residual (getattr on the peeked module, with a
# `sys.modules` fallback) instead of an abort_permanent marker, so the
# hot body JIT-compiles rather than declining to the trait leg.  The
# residual's `__path__` probe runs suppressed (`_PyObject_LookupAttr`), so
# the discarded miss skips the `__spec__` shadowing diagnosis.  Output is
# verified against CPython/PyPy.
N = 200000


def main():
    acc = 0
    i = 0
    while i < N:
        from math import pi, e
        if pi > 3 and e > 2:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
