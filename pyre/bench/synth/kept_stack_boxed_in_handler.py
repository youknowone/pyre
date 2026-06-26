# A boxed-int conditional-expression keeps a heap int (>= 256) live on the
# operand stack across a `goto_if_not` branch guard INSIDE an exception
# handler body.  The kept-stack guard's not-taken arm resumes at a PC the
# exception table protects, where the partial walker snapshot cannot
# reconstruct the kept handler-state operand — so the guard must decline to
# the interpreter (`branch_resume_inside_exc_region`, the flat-free
# replacement for the accidental `stack_slot_color_map` boxed-int decline).
#
# Regression guard: with the decline missing, the bridge / blackhole resume
# rebuilds the kept boxed int (or the re-raised exception) as NULL / a wrong
# value and the handler arithmetic diverges from the interpreter.  Pure
# arithmetic -> deterministic checksum.
N = 300000


def main():
    acc = 9
    i = 0
    while i < N:
        try:
            if i % 3 == 0:
                raise ValueError("x")
            acc = (acc + 2) % 999983
        except ValueError:
            acc = 777777 if (i % 2 == 0) else acc
            acc = (acc + 5) % 999983
        i += 1
    print(acc)


main()
