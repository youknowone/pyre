# pyre-check: max-pypy-ratio=4.7
# The ceiling is fitted to readings, not to a margin over a target.  3.5 was
# fitted while the runners resolved `pypy3.11` to 7.3.23, which ran this
# fixture in 0.23-0.25s against pyre's 0.49-0.51s on ubuntu-cranelift.
# `Pin the CI PyPy oracle to 8.0.0` moved every runner to 8.0.0, whose
# x86-64 speedup halves the oracle to 0.12-0.13s while pyre's own execution
# is unchanged at 0.50-0.52s; the same runner then reads 3.7x and 4.1x.  A
# same-machine A/B of base against branch measured both sides equal with
# identical jit-stats, so what moved is the oracle.  4.7 is the highest
# reading plus 15%.
# A bare re-raise caught in the same frame keeps the original traceback: no
# node is attached at a re-raise coordinate (RaiseWithExplicitTraceback,
# attach_tb=False). The module-level loop's recording iteration runs that
# chain at depth 2. Named re-raise (`raise e`) attaches its node (depth 3),
# and a `finally` passthrough attaches nothing (depth 2).
N = 22000


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
