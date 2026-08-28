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

## It is a SHIFT, and the condition is the CALLEE's freevar count

Give `x` and `y` distinguishable values and the rebuilt array reads out
directly (`inner` returning `(x, y)` reproduces the same way, so the check can
name each item):

    outer.x = 1042        outer.y = 1001042
    inner.x = <class 'cell'>
    inner.y = 1042                        <- outer.x's VALUE, not outer.y's

So the callee's `locals_cells_stack_w` is rebuilt as
`[cell_class, x_cell, ...]`: one extra leading element, every freevar shifted
by one, and the last cell dropped off the end.

The condition is that the INLINED CALLEE closes over exactly two names -- not
how many cellvars the caller has.  With `outer` holding three or four cellvars
and `inner` closing over just two of them, it still fails identically
(`MISMATCH i=1042: <class 'cell'> 1042`):

    callee closes over 2      FAIL      (caller cellvars 2, 3 or 4 alike)
    callee closes over 1      pass      (`y` passed as an argument instead,
                                         or `inner` reading only one of two)
    callee closes over 3      pass      (n swept 1500..40000, all clean)
    callee closes over 4      pass
    `inner` never called                 pass
    the same loop at module level        pass

`while` in place of `for` still fails, `inner` defined before the loop still
fails (and then the closure is loop-invariant, so the trace is much smaller),
a local inside `inner` still fails, and `str` cellvars fail the same way
(`'type' and 'str'`).  Reading `y + x` rather than `x + y` still corrupts the
FIRST freevar, so it is the slot and not the read order.

The count is not an inlining artefact.  For two and three freevars alike
`PYRE_LOOP_CENSUS=1` reports `loop outer`, `PYRE_FBW_INLINE_DIAG=1` reports the
same `[inline-resolved] callee=outer.<locals>.inner nparams=0
has_closure=true`, both record exactly one guard failure and no bridges, and
both build the analogous virtuals -- `NewArrayClear(2)` + `NewArrayClear(4)`
for two, `NewArrayClear(3)` + `NewArrayClear(5)` for three.  The guard that
fails differs (11 against 25), but sweeping n over 1500..40000 keeps the
three-freevar shape clean, so it is not merely which guard happened to fail.

n must clear the trace threshold: 1000 passes, 1500 and up fail.

## It is not confined to a closure built inside the loop

Two shapes an ordinary program would write fail the same way, both at i=1042:

    a factory's closure, built once outside the loop and called in it
        def make(a, b):
            def add(): return (a, b)
            return add
        -> (<class 'cell'>, 3)          slot 0 = the class, slot 1 = a's value

    a two-name lambda
        g = lambda: (lo, hi)            # co_freevars ('hi', 'lo')
        -> (5, <class 'cell'>)          lo read hi's value, hi read the class

Read against `co_freevars` order both are the same shift as above: slot 0
takes the `cell` class and slot k takes what slot k-1 held.  So decorators,
callbacks and factory-made closures over two names are all in range.

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
