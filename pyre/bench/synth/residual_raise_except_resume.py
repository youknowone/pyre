# A hot loop FOLLOWED by a try/except whose body raises through a
# *may-force residual call* (`//`, subscript) rather than an explicit
# `raise`.  The try/except is reached only after the loop's exit guard
# fails, so the blackhole walks forward and executes the residual call
# concretely on the resume path.  When that call raises, the exception
# must be routed to the in-frame `catch_exception` adjacency the
# codewriter emitted right after the call; previously the catch search
# started at the call's operand byte (the dispatch loop advances the
# position only after the handler returns) so the catch was never found
# and the exception escaped the handler.  This bench pins that a
# residual-call raise caught on the resume path returns byte-identically.
#
# These cases keep only int / immutable-literal locals live across the
# `try` — a live heap-object local (e.g. a list bound before the `try`)
# reconstructed on the resume walk is the separate #124 kept-stack gate.
N = 2000000


def divide_by_zero_const(n):
    i = 0
    while i < n:
        i = i + 1
    try:
        return 1 // 0
    except ZeroDivisionError:
        return -2


def divide_by_zero_computed(n):
    i = 0
    acc = 0
    while i < n:
        acc = acc + 1
        i = i + 1
    d = acc - acc
    try:
        return acc // d
    except ZeroDivisionError:
        return acc + 1


def modulo_by_zero(n):
    i = 0
    while i < n:
        i = i + 1
    try:
        return i % 0
    except ZeroDivisionError:
        return i - 5


def tuple_index_resume(n):
    i = 0
    while i < n:
        i = i + 1
    try:
        return (10, 20, 30)[i]
    except IndexError:
        return -7


def main():
    print(divide_by_zero_const(N))
    print(divide_by_zero_computed(N))
    print(modulo_by_zero(N))
    print(tuple_index_resume(N))


main()
