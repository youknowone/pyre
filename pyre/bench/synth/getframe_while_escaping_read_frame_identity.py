# No `max-pypy-ratio`: the loop this fixture DOES compile -- its jitstats
# record `loops_compiled=2` on every backend -- runs too few iterations for
# the generated code to dominate a whole-process measurement. The run
# finishes in a fraction of a second, so a pypy ratio compares two
# interpreters' startup and reads whatever the host's process spawn cost
# happens to be that run. The jitstats baselines gate it.
# The frame-identity read the multi-frame blackhole adopt commits, and the
# regression guard for the path that commits it.
#
# The walk executes the forcing residual CONCRETELY, and an inline push never
# runs the interpreter's call sequence. Before `walker_ec_enter` /
# `walker_ec_leave` published the callee frame on the execution context,
# `ec.topframeref` still named the CALLER while the inlined callee body ran, so
# a `sys._getframe` that is itself the escaping call read the caller's frame at
# walk time and the adopt committed that answer instead of discarding it the way
# the legacy escape/replay path does. Measured then as one wrong iteration per
# multi-frame adopt, in each part:
#
#   part_a  `_gf()`  named `main`, not `leaf`
#   part_b  `_gf(1)` named `<module>`, not `main` -- one level too far up, which
#           is the same error seen through the argument
#
# A `_gf(1)` reading `f_locals` on that shape raised `KeyError` for any caller
# local, for the same reason and not because outer locals go unmaterialized.
#
# Both answers are correct now, and they come from the adopt: with every level
# resumable from the concrete frame it owns, the chain is adopted rather than
# replayed
# (`PYRE_FBW_DEBUG_ABORT=1` prints 10 `BUILT multi-frame` and 10 `adopted
# multi-frame terminal`, with no decline of any kind), so this fixture is once
# again the discriminator for the identity answer rather than a guard on the
# replay that stood in for it. Note the read has to be the ESCAPING call: once
# the escape has happened, a `sys._getframe(1)` executed inside the blackhole was
# always correct, because the chain publishes each level's frame as it runs.
#
# The escape is `leaf_b`'s `f_lasti` read, and it is the only one in the file.
# `sys._getframe` takes no virtualizable force of its own; `rvirtualizable.py
# hook_access_field` places one at each REDIRECTED field access, and `f_lasti`
# reads `last_instr`, one of the five `virtualizable_gen.rs` declares.  The
# `f_code.co_name` both oracles read is not a second escape — `pyframe.rs
# descr_typecheck_fget_f_code` carries no marker and says at its own definition
# why `pycode` needs none — and `leaf_a`'s call is at depth 0, which
# `try_walker_specialize_sys_getframe` answers out of the portal virtualizable
# the walk already holds, so the same read there moves no counter (measured).
# `part_a` therefore keeps its identity oracle over compiled code while the
# adopt this file exists for is reached through `part_b`.
import sys

_gf = sys._getframe

wrong_a = []
wrong_b = []


def leaf_a(x):
    name = _gf().f_code.co_name
    if name != "leaf_a":
        wrong_a.append(name)
    return x + 1


def part_a():
    total = 0
    i = 0
    while i < 30000:
        total = leaf_a(total)
        i = i + 1
    return total


def leaf_b(x):
    fr = _gf(1)
    _ = fr.f_lasti
    name = fr.f_code.co_name
    if name != "part_b":
        wrong_b.append(name)
    return x + 1


def part_b():
    total = 0
    i = 0
    while i < 30000:
        total = leaf_b(total)
        i = i + 1
    return total


print(part_a(), part_b(), len(wrong_a), len(wrong_b))
