# `locals()` called from a function the tracer inlines must report the CALLEE's
# own fastlocals, not the caller's.
#
# `builtin_locals` reaches its frame through `gettopframe_nohidden`
# (`interp_inspect.py:7-11`), and the walker can fold that whole read into
# direct virtualizable reads instead of a residual call.  Inside an inlined
# callee the walker emits those reads against the CALLEE's own virtualizable --
# its inline body carries `getarrayitem_vable_r` / `setarrayitem_vable_r` for
# the callee's slots -- so the fold answers for the right frame.  Nothing
# pinned that before this fixture: the three existing `locals()` fixtures
# (`locals_forced_frame`, `getframe_caller_locals_after_resume`,
# `recursive_forced_frame_kept_stack`) all call it in the portal frame's own
# loop, where the caller's answer and the callee's answer coincide, so the fold
# never fires in a callee there and a fold-on/fold-off comparison over them
# records absent coverage rather than agreement.
#
# The discriminator is the NAME SET, not the values.  A fold that answered from
# the caller's frame would hand back the loop's own names (`i`, `last`,
# `outer`) instead of the callee's (`x`, `y`), and each value would still look
# plausible, so an assertion on values alone can miss it.
#
# Three shapes, because the walker treats them differently:
#
# * `probe_direct` -- one call deep.  All three builtins that share the fold
#   are exercised: `locals()` and `vars()` land on its mapping arm, `dir()` on
#   its sorted-names arm.
# * `probe_nested` -- two calls deep, which is the shape whose inline entries
#   the walker reports as a sub-walk rather than a plain inline body.
# * `probe_closure` -- a callee holding a freevar.  Its locals are not plain
#   fastlocals, so the fold declines and the residual answers; the freevar's
#   name is part of the expected set, which is what catches a fold that took
#   the shape anyway.


def helper_locals(x):
    y = x + 1
    d = locals()
    return sorted(d), d["x"], d["y"]


def helper_vars(x):
    y = x + 1
    d = vars()
    return sorted(d), d["y"]


def helper_dir(x):
    y = x + 1
    return dir()


def probe_direct():
    outer = "caller-only"
    last_locals = None
    last_vars = None
    last_dir = None
    for i in range(100000):
        last_locals = helper_locals(i)
        last_vars = helper_vars(i)
        last_dir = helper_dir(i)
    # `outer` is read after the loop so the caller frame really does carry a
    # name the callee does not; an unread local could be optimized away.
    return outer, last_locals, last_vars, last_dir


def inner_locals(x):
    y = x + 1
    d = locals()
    return sorted(d), d["x"], d["y"]


def outer_helper(a):
    b = a + 2
    return inner_locals(b), b


def probe_nested():
    outer = "caller-only"
    last = None
    for i in range(100000):
        last = outer_helper(i)
    return outer, last


def make_closure(bias):
    def helper(x):
        y = x + bias
        d = locals()
        return sorted(d), d["x"], d["y"]

    return helper


closure_helper = make_closure(1)


def probe_closure():
    outer = "caller-only"
    last = None
    for i in range(100000):
        last = closure_helper(i)
    return outer, last


print(probe_direct())
print(probe_nested())
print(probe_closure())
