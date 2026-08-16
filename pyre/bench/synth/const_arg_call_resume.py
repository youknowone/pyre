# pyre-check: max-pypy-ratio=34
# pypy JITs these tiny list loops to a near-zero denominator, so the ratio is
# dominated by measurement noise on the CI runner (measured 35-43x); the gate
# is set above that band rather than tightened onto the collapsed baseline.
# A hot call whose positional argument is a literal (`build(1000)`) into a
# callee that grows a list over `range(n)`. The list's realloc-boundary append
# declines the FBW fold, so a guard resumes the caller frame at the CALL site.
# The literal argument is loaded into the walker's unboxed int bank, so the
# Ref-only operand mirror left its stack slot a NONE hole; `stack_sync` then
# omitted it and the single-frame bridge rebuilt the argument slot NULL. The
# re-executed CALL passed NULL, so the callee's parameter `n` reconstructed
# unbound -> `UnboundLocalError`. The comprehension and explicit-append forms
# and int / object element strategies each exercise the same caller-frame
# resume; every form must reach `n` (never a dropped element or an unbound
# local).
#
# `bridges_compiled` and `guard_failures` here do not reproduce, so a move in
# either is not evidence about a diff until one configuration has been repeated.
# The same binary on the same source with the same argv, the same child
# environment and PYTHONHASHSEED all held fixed reads six bridges / 1204 or
# seven / 1404, run to run -- 1 of 6 runs, then 2 of 6, then 0 of 8 an hour
# later. CI has seen it too: `dynasm/synth/const_arg_call_resume` came up
# `jit-stats unstable (not gated)` on the windows runner. A machine settles on
# one value and repeats it many times over, which is what the stability re-runs
# cannot see through -- they ask whether a reading reproduces itself, and this
# one does, until it does not. Whichever way it is settled locally, it will
# stay that way long enough to look like a property of the tree.
#
# The two readings are one bridge generation apart. Each of the three resume
# guards keeps failing after its first bridge is attached, so another is
# compiled every `trace_eagerness` (200) failures. Under MAJIT_GUARDLOG both
# readings show the same three guard sites and differ only in their failure
# totals: 402/402/400 sums to 1204 and floors to 2+2+2 = six bridges,
# 602/402/400 to 1404 and 3+2+2 = seven.
#
# Two things that do not settle it, both measured, so neither is worth trying
# again:
#
# * The trip count. 220, 240, 260, 280, 300, 320, 360 and 400 read the same
#   counters, so there is no value that lands away from the boundary; only at
#   200 does a guard stop reaching the threshold at all, and that alternates
#   too (1199 / 1397). Halving the file is what settled `str_fstring`, whose
#   counters moved with a major-collection crossing -- `back_edge_polls` reads
#   0 in every run here at every trip count, which is what says this is not
#   that class.
# * Re-recording. It pins whichever side the recording machine was on, and
#   everything moves it: adding these comment lines alone -- text the parser
#   drops, changing nothing the JIT can see -- moved a wasm reading from 1404
#   to 1204 for eight runs in a row.
#
# What the fixture is actually gated on is its printed sum against cpython and
# pypy, which is what catches the bug described above.


def comp(n):
    return [i for i in range(n)]


def obj_comp(n):
    return [(i, i) for i in range(n)]


def build(n):
    r = []
    for i in range(n):
        r.append(i)
    return r


total = 0
k = 0
while k < 400:
    total += len(comp(1000))
    total += len(obj_comp(1000))
    total += len(build(1000))
    k += 1
print(total)
