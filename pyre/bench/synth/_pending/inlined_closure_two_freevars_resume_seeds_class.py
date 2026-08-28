"""Deterministic witness: a resume into an inlined closure seeds a freevar
register with the `cell` CLASS instead of the caller's cell.

Run with no arguments.  `PYRE_NO_JIT=1`, cpython and pypy print `8997000`;
`pyre-dynasm` and `pyre-cranelift` both print

    TypeError: unsupported operand type(s) for +: 'type' and 'int'

raised inside `inner` at `return x + y`.  Both backends, every run.

## What the value is

Read out of the traceback frame (the readout runs only after the failure, so
it does not perturb what compiled):

    ADDR leaked      = 0x9752a4908      # inner's freevar `x`
    ADDR cell type   = 0x9752a4908      # type(a real cell)   -- SAME OBJECT
    ADDR a real cell = 0x973c6ddd8

`inner`'s `x` is the `cell` **class object**, not a cell and not an int.
`outer`'s own `x` is still `1042` at that moment, so the two are not the same
cell -- the callee was handed something else entirely.

## Where it is injected

`MAJIT_BH_DEBUG=1` prints the blackhole's seeding of the two-frame chain the
guard failure rebuilds, and the mis-seed is visible directly:

    [bh-setpos] jitcode="inner" position=126
    [bh-seed] r2 = 0x9752a4908       <- the `cell` CLASS  (should be x's cell)
    [bh-seed] r5 = 0x974eb0010
    [bh-chain] frame=0 qualname=outer.<locals>.inner position=126 py_pc=Some(4)
    [bh-chain] frame=1 qualname=outer position=932 py_pc=Some(32)
    [bh-seed] r8 = 0x9749a3720       <- outer's x cell
    [bh-seed] r9 = 0x974eb0010       <- outer's y cell

`inner`'s `r5` IS `outer`'s `r9`, so the second freevar is seeded correctly
from the caller.  `inner`'s `r2` should likewise be `outer`'s `r8` and is the
class constant instead.  The jit-summary reports `Guard failures: 1`,
`Total # of bridges: 0` -- one guard failure is all it takes, and the wrong
answer is produced by the blackhole, not by compiled code.

## Why it is silent rather than a crash

`bh_load_deref_value_fn` (`call_jit.rs`) reads

    let value = if !slot.is_null() && is_cell(slot) { w_cell_get(slot) }
                else { slot };

so a non-cell in a freevar slot is passed through unchanged instead of being
rejected.  `pyopcode.py LOAD_DEREF` has no such fall-through -- it does
`cell = self.cells[varindex]; w_value = cell.w_value` -- so the `else` arm is
pyre-only, and it is what turns a mis-seeded register into a wrong answer
several frames away rather than an immediate failure.  Whatever fixes the
seeding, that arm is worth making loud.

## Why the shape is exactly this

Every widening below was measured, and each one stops reproducing, so none of
it is decoration:

    2 cellvars, `inner` reads BOTH, `inner` is CALLED     FAIL
    1 cellvar   (`y` passed as an argument instead)       pass
    2 cellvars, `inner` reads only one of them            pass
    3 cellvars, `inner` reads all three                   pass
    4 cellvars, `inner` reads all four                    pass
    `inner` defined but never called                      pass
    the same loop at module level rather than in `outer`  pass

`while` in place of `for` still fails, `inner` defined before the loop still
fails, and `str` cellvars fail the same way (`'type' and 'str'`), so it is
neither the loop form nor the value type.

The count is a real condition, not an inlining artefact: `PYRE_LOOP_CENSUS=1`
reports `loop outer` for 2, 3 and 4 cellvars alike, `PYRE_FBW_INLINE_DIAG=1`
reports the identical `[inline-resolved] callee=outer.<locals>.inner
nparams=0 has_closure=true` admission for 2 and for 3, and both record exactly
`Guard failures: 1`.  Only the 2-cellvar shape answers wrong.

n must clear the trace threshold: 1000 passes, 1500 and up fail.

`PYRE_FBW_MULTIFRAME_DEPTH` 1, 2, 3 and 7 all reproduce, and so do
`PYRE_FBW_MULTIFRAME` 0 and 1, so unlike gh#1444 this is not the multi-frame
adopt path -- it is the ordinary resume encoding.  gh#1311 is the same family
on the frame register: a non-frame word admitted on a bare `!= 0`.

Adding an `isinstance` check inside `inner` makes it pass, so the assertion
below stays outside the closure and touches nothing the loop compiles.
"""

EXPECTED = 8997000


def outer(n):
    t = 0
    for i in range(n):
        x = i
        y = i

        def inner():
            return x + y

        t += inner()
    return t


got = outer(3000)
if got == EXPECTED:
    print("PASS inlined closure two freevars")
else:
    print("FAIL %r against %r" % (got, EXPECTED))
    raise SystemExit(1)
