# JIT-stress fixture for the malformed-jitcode `raise/r` register read.
#
# `pypyjit.set_param` drops the thresholds to 1 so every lambda below is
# recorded on its first call. In that regime the codewriter produced a jitcode
# for `lambda: [x for x in Boom()]` whose shared exception tail reads Ref
# registers that no op in the jitcode ever writes (the tail is the merge of
# three `catch_exception` handlers, and the value-stack flush it performs
# differs per raise site). The walk then handed `OpRef::NONE` to `raise/r`,
# which reached the backends as `Finish(_)` against an
# `ExitFrameWithExceptionDescrRef` whose fail-arg type is Ref — dynasm panicked
# in `RegisterManager.loc`, cranelift in `resolve_opref`, on the same trace.
#
# Every `show(...)` below is load-bearing: the walk only reaches the malformed
# tail once the earlier sections have accumulated their tracing state. Removing
# any one of them stops the fixture from exercising the path.
#
# CPython (the oracle) has no `pypyjit`; PyPy and pyre do. Guarding the import
# keeps the output identical across all three.
try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


def show(label, fn):
    # Swallowing the exception keeps the fixture's output stable while still
    # driving the raise through the recorded frame.
    try:
        fn()
    except BaseException as e:
        "!%s: %s" % (type(e).__name__, e)


class Boom:
    def __iter__(self):
        return self

    def __next__(self):
        # `self.n` does not exist: every iteration raises AttributeError from
        # inside the comprehension's loop body.
        return self.n


# The first two `exec` payloads run and print; the last two carry a stray
# indent and raise IndentationError, which `show` absorbs. All four are here
# for the tracing state they accumulate.
show("loop_var_leak", lambda: exec("for q in range(3): pass\nprint('  ', q)"))
show("loop_var_empty", lambda: exec("for q2 in []: pass\nprint('  ', 'q2' in dir())"))
show("break_in_finally", lambda: exec("        print('  fin', i)\n"))
show("continue_else", lambda: exec("    print('  else ran')\n"))
show("iter_raises_midloop", lambda: None)
show("midloop_error", lambda: [x for x in Boom()])
print("done")
