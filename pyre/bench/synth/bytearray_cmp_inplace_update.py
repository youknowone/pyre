# A hot loop compares two bytearrays, mutates one in place (same buffer
# pointer and length), then compares again. `ll_bytes_strcmp` is elidable, so
# a CALL_PURE keyed on those slice words can reuse the first result; the
# second comparison must observe the store. Mixed `bytearray`/`bytes` is the
# same hazard; `bytes`/`bytes` stays on the elidable path and is only a
# control.
#
# This guards a latent hazard, and did not reproduce one when it was written:
# the traces residualize `compare` as `CallMayForceR`, so no CALL_PURE for
# `ll_bytes_strcmp` is in the loop and the fixture passes with or without the
# bytearray split. It is here to fail the day that call does descend, which is
# the day the stale result becomes reachable and silent.
#
# No `max-pypy-ratio`: this is a comparison-result oracle, not a workload.
#
# Expected output:
#   intra 300000 0
#   across False
#   mixed False
#   bytes True
N = 300000


def intra(n):
    a = bytearray(b"xx")
    b = bytearray(b"xx")
    n_eq = 0
    n_stale = 0
    i = 0
    while i < n:
        if a == b:
            n_eq += 1
        a[0] = 1
        if a == b:
            n_stale += 1
        a[0] = 120
        i += 1
    return n_eq, n_stale


def across(n):
    a = bytearray(b"xx")
    b = bytearray(b"xx")
    last = True
    i = 0
    midpoint = n // 2
    while i < n:
        last = a == b
        if i == midpoint:
            a[0] = 1
        i += 1
    return last


def mixed(n):
    a = bytearray(b"xx")
    b = b"xx"
    last = True
    i = 0
    midpoint = n // 2
    while i < n:
        last = a == b
        if i == midpoint:
            a[0] = 1
        i += 1
    return last


def bytes_ctrl(n):
    a = b"xx"
    b = b"xx"
    last = False
    i = 0
    while i < n:
        last = a == b
        i += 1
    return last


def main():
    n_eq, n_stale = intra(N)
    print("intra", n_eq, n_stale)
    print("across", across(N))
    print("mixed", mixed(N))
    print("bytes", bytes_ctrl(N))


main()
