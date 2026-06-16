# A hot loop FOLLOWED by a try/except whose handler `return`s.  The
# `return`-from-`except` cleanup compiles to `SWAP 2; POP_EXCEPT;
# RETURN_VALUE` (the SWAP exchanges the return value with the saved
# exception state so POP_EXCEPT can pop the state and leave the value at
# TOS).  The try/except is reached only after the loop's exit guard fails,
# so the blackhole walks forward through the SWAP on the resume path.
# Previously the codewriter emitted `abort_permanent` for SWAP, so the
# resume walk failed ("call failed" / uncaught escape).  This bench pins
# that the SWAP-bearing handler tail resumes byte-identically.
N = 2000000


def raise_then_return(n):
    i = 0
    while i < n:
        i = i + 1
    try:
        raise ValueError(i)
    except ValueError:
        return -1
    return 0


def raise_then_return_value(n):
    i = 0
    s = 0
    while i < n:
        s = s + i
        i = i + 1
    try:
        raise ValueError(s)
    except ValueError as e:
        return e.args[0] + 7
    return 0


def computed_except_return(n):
    i = 0
    acc = 0
    while i < n:
        acc = acc + (i & 1)
        i = i + 1
    try:
        raise IndexError(acc)
    except IndexError as e:
        # `return acc - e.args[0] + 5` computes the value before
        # POP_EXCEPT, forcing the SWAP-2 reorder in the handler tail.
        return acc - e.args[0] + 5
    return 0


def multi_clause_second(n):
    # Multiple except clauses reached via loop-exit resume: the raised
    # IndexError must skip the (non-matching) KeyError clause and be caught
    # by the second clause.  Exercises CHECK_EXC_MATCH against a proper
    # exception type object on the resume walk.
    i = 0
    while i < n:
        i = i + 1
    try:
        raise IndexError(i)
    except KeyError:
        return -100
    except IndexError:
        return i - 3
    return 0


def main():
    print(raise_then_return(N))
    print(raise_then_return_value(N))
    print(computed_except_return(N))
    print(multi_clause_second(N))


main()
