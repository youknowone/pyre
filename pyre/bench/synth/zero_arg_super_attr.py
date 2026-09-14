# pyre-check: skip-cpython
# pyre-check: spec-folds=load_deref,load_super_attr,super_attr_unwrap
# Zero-argument super inside a FOR_ITER body. CPython is skipped because
# even this N is impractical there. spec-folds is the gate: the compiled
# loop still reads a virtualizable frame and is not yet as tight as
# PyPy's virtualized super, so there is no max-pypy-ratio until that
# virtualize lands. N is sized so the census finishes inside check.py's
# 20s budget.
#
# `load_super_attr_descent` still declines at unpublished `w_method_new`.
# Publishing it in a throwaway dynasm binary exposed an unclosed RootScope
# bracket: before `7d57e55a2dc` the shadow stack grew each iteration.  Best of
# three runs per N, in ns/iteration (`fold` disables only the descent):
#
#                  descent           fold          `PYRE_NO_JIT=1`
#         N     before  after     before  after     before  after
#   250,000      3131    265        353    182       1516    860
#   500,000      3445    169        170     91       1315    774
# 1,000,000      4372    119         87     46       1243    730
# 2,000,000      8009     95         45     23       1181    709
#
# The fixed descent is linear (about 0.049s + 71ns/iteration), but remains 4.2x
# slower than the fold at N=2,000,000, so `w_method_new` stays unpublished.
#
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 5_000_000


class Base:
    def val(self):
        return 1


class Child(Base):
    def run(self, n):
        acc = 0
        for _ in range(n):
            acc = acc + super().val()
        return acc


print(Child().run(N))
