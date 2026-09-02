# pyre-check: max-pypy-ratio=2.9
# The ceiling covers a startup deficit rather than a slower loop: a pyre
# backend subtracts pypy's startup, not its own, so the 0.05s-0.11s a pyre
# process spends above pypy to reach the first bytecode stays inside this
# fixture's numerator, against a pypy execution of about a sixth of a
# second.  The readings that leaves are 1.36x-2.3x across the hosts; 2.9
# covers the widest with a quarter's headroom and its 0.483x derived floor
# stays under the narrowest.
# GC heap-array ops on the blackhole guard-failure resume path. The loop
# body reads and writes a list (getarrayitem_gc_i / setarrayitem_gc_i) and
# the function returns a tuple (new_array_clear + setarrayitem_gc_r +
# newtuple) at loop exit, where the `i < N` guard fails and execution
# resumes forward in the blackhole. Those GC-array opcodes were absent from
# the production blackhole builder's dispatch table, so any deopt resuming
# through a tuple/list build or an array element access panicked with an
# unwired opcode. A traced loop that never deopts through these ops would
# not cover them.
N = 34160000


def work(n):
    buf = [0, 0, 0, 0]
    total = 0
    i = 0
    while i < n:
        d = (i % 7) + 1
        buf[i % 4] = (buf[i % 4] + (i * 13 + 5) // d) % 1000003
        total = (total + buf[i % 4]) % 1000003
        i += 1
    return total, buf[0], buf[1], buf[2], buf[3]


def main():
    print(work(N))


main()
