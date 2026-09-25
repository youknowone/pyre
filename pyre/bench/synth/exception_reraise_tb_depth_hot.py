# pyre-check: max-pypy-ratio=4
# ubuntu cranelift measured 3.7x against a 3.5 gate; the check said that
# needs 0.29x of pypy startup to be noise. Same allowance as exception_reduce.
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
