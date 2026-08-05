# A loop-carried accumulator that flips int -> float partway through, so the
# guard at that position fails repeatedly and the bridge grown there closes with
# a virtual state matching no existing target. That makes the optimizer ask for a
# retrace (`compile.py:1085 retrace_needed`), and `retrace_limit` is what lets
# one be BUILT: it defaults to 0 (`rpython/rlib/jit.py:595`), so no other fixture
# in this corpus reaches `compile_retrace` at all.
#
# What this currently covers is the retrace path up to and including its give-up:
# `retrace_needed` -> `cut_retrace_from` -> the unroll pass -> `jump_to_preamble`
# (`unroll.py:156/171`) -> the `compile.py:334` arity give-up. The retrace is not
# assembled, because pyre's start label is the portal entry contract while the
# optimized loop-carried set is wider — see MC_DIAG slot 57. When that contract
# is unified this fixture is what starts exercising a compiled retrace, and its
# jit-stats row is what will say so.
#
# CPython (the oracle) has no `pypyjit`; PyPy and pyre do. Guarding the import
# keeps the output identical across all three while the param only binds where a
# JIT exists. `set_param` rather than an environment variable because the wasm
# guest sees no environment.
try:
    import pypyjit

    pypyjit.set_param("retrace_limit=5")
except ImportError:
    pass


def f(n):
    s = 0
    i = 0
    while i < n:
        if i > 4000:
            s = s + 0.5
        else:
            s = s + 1
        i += 1
    return s


print(f(60000))
