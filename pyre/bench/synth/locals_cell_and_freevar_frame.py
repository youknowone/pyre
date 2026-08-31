# `locals()` every iteration of a hot loop, on frames that carry cells and
# freevars.
#
# `fast2locals` binds one key per localsplus slot: the `varnames` band, then
# the pure cellvars, then the freevars. A cell slot holds the `Cell` and its
# key takes `Cell.contents`, so reproducing it is one extra read per slot over
# the plain-fastlocal shape `locals_forced_frame.py` covers. Without that read
# the call stays an opaque residual, which forces the virtualizable and loses
# the loop to ABORT_ESCAPE.
#
# The call is in the loop BODY, not behind a rare branch: a `locals()` reached
# only from a side exit never enters the trace and neither forces nor folds.
#
# Values, not keys: the keys come from the code object and are right whether or
# not the frame was forced, while an unforced virtualizable still holds
# whatever the frame last wrote out. Each total is a sum of differences that
# only comes out right if every mapping reports the CURRENT iteration.
#
# Three bands, one probe each:
#   `pure_cellvar`  - a cellvar that is not also a parameter, so it lives above
#                     `varnames` and is named by `cell_slot_names`.
#   `param_cell`    - a parameter `MAKE_CELL` turned into a cell, so it keeps
#                     its `varnames` slot and its `varnames` name.
#   `freevar`       - read through `COPY_FREE_VARS` from the enclosing scope.
#
# `param_cell` checks the answer, not the compiled path: a frame whose
# PARAMETER is a cell does not enter the JIT at all today, `locals()` or no
# `locals()` -- the same loop with the nested function deleted reports
# `mc_entered=1` and this one reports `0`. That is a separate limit ahead of
# this fold, so the probe holds the band's correctness until it lifts, and only
# the other two say anything about compilation.

N = 100000


def pure_cellvar():
    captured = 0

    def peek():
        return captured

    total = 0
    for i in range(N):
        captured = i * 2
        total += locals()["captured"] - i
    return total, peek()


def param_cell(k):
    def peek():
        return k

    total = 0
    for i in range(N):
        k = i + 7
        d = locals()
        total += d["k"] - d["i"]
    return total, peek()


def make_freevar_probe(step, limit):
    def probe():
        acc = 0
        total = 0
        for i in range(limit):
            acc += step
            d = locals()
            total += d["acc"] + d["step"] - d["i"]
        return total

    return probe


print(pure_cellvar())
print(param_cell(0))
print(make_freevar_probe(3, N)())
