# pyre-check: spec-folds=make_function,set_function_attribute
# An annotated `def` in a hot FOR_ITER body: the SET_FUNCTION_ATTRIBUTE arm
# 3.14 reaches for far more often than the defaults one, and the one that
# decides whether anything else in the definition sequence folds.
#
# A return annotation alone compiles to two MAKE_FUNCTIONs -- one for the
# `__annotate__` closure PEP 649 defers the annotation to -- and a single
# `SET_FUNCTION_ATTRIBUTE annotate`, emitted BEFORE any defaults stamp. So this
# fixture's one attribute stamp IS the annotate arm: `spec-folds` firing here
# cannot be satisfied by the defaults arm the way it can in
# `foriter_make_function_body`, which is why the shape is worth its own
# fixture rather than an annotation added to that one.
#
# The fold census observes both definition operations directly.  Low
# thresholds keep the loop compiled without hundreds of millions of arithmetic
# iterations whose only purpose was lifting a wall-clock ratio above its floor.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 20000


def main():
    total = 0
    for i in range(N):

        def add(value) -> int:
            return value + 1

        total += add(i)
    print(total)


main()
# Expected: 200010000
