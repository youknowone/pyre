# pyre-check: spec-folds=make_function,set_function_attribute
# MAKE_FUNCTION plus SET_FUNCTION_ATTRIBUTE in a hot FOR_ITER body. The default
# value forces the companion attribute initializer onto the definition path.
#
# `spec-folds` is what gates the subject. Both opcodes emit their effect inline
# -- the allocation and its `Function.__init__` stores, and then the single
# `SetfieldGc` the attribute flag names -- so the definition sequence
# virtualizes away entirely when the function does not escape, which is what
# the loop here does. What is left to time is the arithmetic, and no throughput
# number can distinguish "the fold fired" from "the loop was cheap anyway"; a
# census that says each fold fired at least once can.
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

        def add(value=i):
            return value + 1

        total += add()
    print(total)


main()
# Expected: 200010000
