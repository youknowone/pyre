# pyre-check: max-pypy-ratio=86
# pyre-check: jitstats-band=guard_failures=23,loops_compiled=1
# Both bands cover one measured effect: this fixture's loop count is decided by
# a warm-up race, not by whether its arm compiles.
#
# `forward` calls `adjust` exactly once per iteration, so the two green keys
# tick their function-entry counters in lockstep and which of them crosses the
# threshold first is emergent. When `forward` wins, it traces `adjust` inlined
# and `adjust` never reaches the entry door again -- seven loops. When `adjust`
# wins, it also earns its own 36-op entry trace -- eight loops -- and `forward`
# still inlines it, its trace going 69 ops to 70 with a `GuardNotInvalidated`
# as the extra op. Both outcomes are live in one tree at once: windows reads
# seven while macos and ubuntu read eight, byte-identical to each other.
#
# The race is sensitive to how much unrelated work runs before it. Registering
# one more GC type shifts it -- three added descrs at startup were enough to
# flip the order -- and so does `PYTHONDONTWRITEBYTECODE`: one binary read
# seven loops with bytecode writes suppressed and eight with them enabled, at
# every N from 48000 through 128000. None of it reaches the arm. The arm's
# trace was read directly in all four combinations and is identical op for op,
# same call targets modulo the ASLR slide, differing only in the order of an
# unordered `SetfieldGc` write-back set.
#
# So width 1 on `loops_compiled` admits exactly the second outcome, and
# `guard_failures` has to admit what that outcome costs: 1005 -> 1018 dynasm
# and 1009 -> 1024 cranelift on both failing runners, 15 counts at the widest.
# The remaining 8 is what this directive already carried, for two things this
# fixture is also not about. One is the host: one tree read 1003 dynasm / 1008
# cranelift on macOS and ubuntu and 1004 / 1009 on windows in a single CI run
# (`1d212895c6b`), with the loop and bridge counts agreeing everywhere. The
# other is the collection schedule -- one binary swept across nursery sizes
# read 1034 / 1014 / 1007 / 1007 at 2 / 4 / 6 / 8 MB, 27 counts, while
# `loops_compiled` and `bridges_compiled` did not move. Suppressing the whole
# trace-time fold table moved it by one count and suppressing the folds this
# branch adds by none, so it is not reading those either.
#
# `bridges_compiled` stays gated exactly at 5 and held at 5 in every
# configuration measured above, and the regression floor still gates
# `loops_aborted` at 0, so the dead-bridge and abort classes this suite exists
# to catch are untouched by either band. What is given up is reading a
# one-count fall in `loops_compiled` as "a hot loop went back to interpreted",
# which this fixture cannot support anyway while the race decides that count.
#
# The ceiling is a function of N, so raising N refits it. pypy's execution here
# is almost all fixed cost -- doubling N moved it 0.035s to 0.039s -- while this
# backend pays roughly 27us per iteration, so the ratio tracks N nearly one for
# one. At N = 64000 the local cranelift ratio reads 44.9x, and the ceiling sits
# just under twice it.
# An inlined closure keeps its freevar cells in MIFrame.registers_r across a
# may-force call.  The `Fraction` arithmetic in `forward` is that may-force
# call and invalidates heap-cache facts; the following LOAD_DEREF must recover
# `adjust` from the callee frame's own shadow instead of becoming an unstamped
# GetarrayitemGcR and aborting at the result branch.
#
# The abort is what this fixture guards, and check.py's regression floor gates
# `loops_aborted` at 0 independently of the ratio below.  Bridge resume must
# retain the callee frame's own globals identity: treating its valid pc=0 as a
# failed decode leaves two guards failing on nearly every iteration, while
# guarding the callee namespace through the portal/root frame compiles an
# endless chain of equally failing bridges.  With both frame properties
# preserved the guard-failure count reaches a fixed point instead of growing
# with the iteration count; the 471 it once settled at belongs to a smaller N
# and an older baseline, and what it settles at now is what the committed
# baselines carry.
from fractions import Fraction


# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
#
# The upper bound is what fixes the counters. At 32176 the loop ended while the
# JIT was still converging, so the gated totals recorded how far that had got
# and moved between two runs of one binary -- 923/6 loops against 925/7 here,
# 923/6 against 938/8 on ubuntu and 922 against 923 on windows. Convergence
# completes by 48000 on both native backends, and past it every gated counter is
# independent of N: the loop, bridge and guard-failure values the committed
# baselines carry hold from 48000 through 96000. This sits far enough above that
# point to keep the fixed point on a host that needs a few more iterations to
# reach it.
#
# Seven loops and the third guard failure are the `catch_exception/L` arm being
# compiled: once jd0's staticdata carries the assembler's real opcode ids
# instead of the 255 `op_live` sentinel, an exception the blackhole used to let
# escape is caught. That arm stopped being reached for a while and the counters
# fell back to six loops; it is reached again as of `d6408935b3c`, and all three
# legs agreed on the way back up, which is what the current baselines record.
# Two things about that return are unattributed. The guard failures did not come
# back to the 1007/1012 pair the arm earned historically but to 1005/1009. And
# wasm gained the loop as well -- its baseline moved off six at the same commit
# -- while its guard failures stayed at 1004, so on that backend the arm costs a
# loop and no guard failure.
#
# The earlier reading of this counter -- that the seventh loop was a
# `catch_exception/L` blackhole arm -- does not survive measurement:
# `MAJIT_BH_DEBUG=1` shows all 1002 blackhole entries at `frame=0` across three
# qualnames (`<module>`, `Fraction._add`, `Fraction._div`), never `forward`,
# and both `fbw_blackhole_adopted_*` counters are zero.
#
# The discriminator is the admission, not the fold: `PYRE_FBW_NO_SPECIALIZE=`
# `load_deref` suppresses the trace-time fold and moves no counter, because the
# reclassification happens in the pre-scan that runs before `scan.verdict()`.
# Blocking the inline through any other term of the same admission brings the
# loop back -- wrapping `forward`'s body in `try: ... finally: pass` fails the
# `!has_exception_table` term and reads seven loops again, naming
# `make_forwarder.<locals>.forward` as the extra trace.
#
# DO NOT RE-RECORD `loops_compiled` here on a six. Seven is base-dependent,
# not merely a value some tree happens to produce. Measured across three CI
# runs: main alone at base `5bf59e1f008` reads seven and passes (32565716769);
# a branch adding `RuntimeHelperKind::LoadDeref` to the replay-safe read set reads
# seven on the older base `68a6351bfbf` and passes (32556952922); the *same*
# six commits on `5bf59e1f008` read six and fail (32572034383). On that branch
# the loop that goes missing is `forward`'s own portal trace, which vanishes
# because the admission change inlines the freevar callee -- an improvement,
# and a different loop from the `catch_exception/L` arm above. A six is a
# question about which change interacted; re-recording it pins a number the
# next tracer change moves again.
#
# The committed baselines are seven loops, five bridges, and 1005 dynasm / 1009
# cranelift / 1004 wasm. Neither count answers whether the arm compiles -- the
# arm's trace reads identically under every configuration that moved them, see
# the band note above -- so treat a move in either as the unattributed
# remainder rather than as this fixture's subject.
N = 64000


def make_forwarder():
    def adjust(value):
        return value + Fraction(3, 97)

    def forward(value):
        scaled = value / Fraction(2, 89)
        return adjust(scaled)

    return forward


forward = make_forwarder()
count = 0
for i in range(N):
    value = forward(Fraction(i % 97 + 1, 97))
    if value > 1:
        count += 1

print(count)
