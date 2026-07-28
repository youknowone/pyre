# pyre-check: max-pypy-ratio=8
# JIT-stress twin of exception_reraise_tb_depth_hot: `pypyjit.set_param`
# lowers the trace/function thresholds to 1 so trace recording fires on the
# earliest iterations rather than only after the ~1600-iteration warmup. That
# makes the recording pass land on the re-raise sections on every run and every
# backend (the param hook reaches the wasm guest, which sees no environment),
# turning the historically platform-dependent recording-path coverage into a
# deterministic check.
#
# Traceback shape invariants (identical to the non-stress twin): a same-frame
# bare re-raise keeps the original traceback (depth 2, no node at the re-raise
# coordinate), a named re-raise `raise e` attaches its node (depth 3), and a
# `finally` passthrough attaches nothing (depth 2).
# CPython (the oracle) has no `pypyjit`; PyPy and pyre do. Guarding the import
# keeps the output identical across all three while the thresholds only bind
# where a JIT exists.
try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

N = 600


def thrower(i):
    raise KeyError(i)


bare_depths = set()
bare_bad = 0
for i in range(N):
    try:
        try:
            thrower(i)
        except KeyError:
            raise
    except KeyError as e:
        depth = 0
        traceback = e.__traceback__
        while traceback is not None:
            depth += 1
            traceback = traceback.tb_next
        bare_depths.add(depth)
        bare_bad += depth != 2
print("bare_depths =", sorted(bare_depths))
print("bare_bad =", bare_bad)

named_depths = set()
for i in range(N):
    try:
        try:
            thrower(i)
        except KeyError as e:
            raise e
    except KeyError as e2:
        depth = 0
        traceback = e2.__traceback__
        while traceback is not None:
            depth += 1
            traceback = traceback.tb_next
        named_depths.add(depth)
print("named_depths =", sorted(named_depths))

finally_depths = set()
for i in range(N):
    try:
        try:
            thrower(i)
        finally:
            pass
    except KeyError as e3:
        depth = 0
        traceback = e3.__traceback__
        while traceback is not None:
            depth += 1
            traceback = traceback.tb_next
        finally_depths.add(depth)
print("finally_depths =", sorted(finally_depths))
