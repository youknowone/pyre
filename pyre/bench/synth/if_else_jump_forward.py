N = 1500000


def main():
    # `if cond: ... else: ...` inside a hot `while` loop.  The THEN block ends
    # with an unconditional forward jump over the ELSE body (JUMP_FORWARD), so
    # the loop body exercises JUMP_FORWARD on the hot (THEN) path.  The
    # condition is THEN-dominant (true 6/7 of the time) so the compiled trace
    # stabilizes on the THEN arm — which carries the forward jump — while the
    # ELSE arm stays live as an occasional bridge.  Pure integer arithmetic
    # keeps the result identical across CPython, PyPy, and pyre.
    i = 0
    acc = 0
    while i < N:
        if i % 7 != 0:
            acc = acc + (i % 17)
        else:
            acc = acc - (i % 31)
        i = i + 1
    print(acc)


main()
