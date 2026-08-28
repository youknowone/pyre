# gh#1444 repro: a function whose every path returns an `int` returns `None`.
#
#     TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
#
# `rec` below returns `1`, `0`, or `rec(...) + 1`.  Never `None`.  The caller
# reports the JIT's answer as a Python-level type error, which is what makes
# this shape easy to dismiss as a mistake in the script.
#
# NOT gated: still reproduces on dynasm and cranelift, backend-identical, and
# passes under `PYRE_NO_JIT=1`.  Kept here so the narrowing below is not lost.
#
# ## What it is (measured 0828, one dynasm binary)
#
# `PYRE_FBW_MULTIFRAME_DEPTH=1` answers correctly; `=2` and the default `7` do
# not.  So the defect is in the depth->=2 multi-frame path, not in any writer of
# `last_instr`: pairing the inline-callee coordinate publish with the forward
# analysis's `valuestackdepth` was tried and changes nothing here.
#
# `PYRE_FBW_DEBUG_ABORT=1` names the event -- one adoption per run:
#
#     [walk-abort] err=LoopBearingCalleeInlineUnsupported { pc: 258,
#                  blackhole_required: true } claimed=true
#     [mf-adopt] frame=0 name="<module>" / 1 "step" / 2,3,4 "rec"
#     [fbw-blackhole] adopted multi-frame terminal depth=5
#
# `MAJIT_LOG=1` shows the terminal is a `DoneWithThisFrameRef`, so neither
# `adopt_blackhole_crn` (the CRN arm) nor the `BailToInterpreter` arm and its
# documented root-coordinate mismatch is the path taken.
#
# ## The trigger is narrow
#
# The callee must be SELF-RECURSIVE and its base case must reach a may-force
# builtin.  Measured over one binary, same driver both loops:
#
#     rec + id(x)      FAIL      rec + hash(x)   pass
#     rec + id(0)      FAIL      rec + abs(x)    pass
#     rec + repr(x)    FAIL      rec + len([])   pass
#     id(x) in the recursive arm rather than the base case: pass
#     no recursion, same id(x): pass
#
# ## Grading
#
# Run it and compare against CPython 3.14's `592304421450`.  BOTH loops are
# load-bearing -- neither alone reproduces at any size -- and so is the plain
# shape: adding `is None` checks changes what compiles, and the instrumented
# variant passes on binaries the plain one fails.
def ck(seq):
    h = 7
    for v in seq:
        h = (h * 1000003 + v) & 0xFFFFFFFFFF
    return h


def rec(x):
    if x <= 0:
        return 1 if id(x) else 0
    return rec(x - 1) + 1


class W:
    def __init__(self):
        self.seen = []

    def step(self, x):
        self.seen.append(x)
        return x * 1000003 + rec(x % 4)


hh = 7
for trial in range(50):
    n = trial % 50
    w = W()
    res = [w.step(i) for i in range(n)]
    hh = (hh * 1000003 + ck(res) + ck(w.seen)) & 0xFFFFFFFFFF
for trial in range(60):
    n = trial % 20
    w = W()
    res = []
    for i in range(n):
        res.append(w.step(i))
    hh = (hh * 1000003 + ck(res) + ck(w.seen)) & 0xFFFFFFFFFF
print(hh)
