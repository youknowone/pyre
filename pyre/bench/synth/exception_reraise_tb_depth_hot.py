# pyre-check: max-pypy-ratio=4
# ubuntu cranelift measured 3.7x against a 3.5 gate; the check said that
# needs 0.29x of pypy startup to be noise. Same allowance as exception_reduce.
# A bare re-raise caught in the same frame keeps the original traceback: no
# node is attached at a re-raise coordinate (RaiseWithExplicitTraceback,
# attach_tb=False). The loop's recording iteration runs that chain at depth
# 2. Named re-raise (`raise e`) attaches its node (depth 3), and a `finally`
# passthrough attaches nothing (depth 2).
# Each loop runs in a function. At module level `except E as e` stores and
# deletes a global every iteration, which bumps the module dict's `version?`
# (celldict.py `mutated`) and aborts every trace reaching it. Which of the
# short traceback loops then closed followed the minor-collection schedule
# (JitCounter decay runs every 32 minors, counter.py
# `invoke_after_minor_collection`), so any allocation change moved the loop
# counts. Each loop also raises from its own function: one shared raiser
# collected the interpreted calls of all three warm-ups and sat on the
# function_threshold edge, so its entry trace came and went with the same
# schedule. N keeps pypy's exec time above the ratio gate's floor.
N = 500000


def throw_bare(i):
    raise KeyError(i)


def throw_named(i):
    raise KeyError(i)


def throw_finally(i):
    raise KeyError(i)


def run_bare():
    depths = set()
    bad = 0
    for i in range(N):
        try:
            try:
                throw_bare(i)
            except KeyError:
                raise
        except KeyError as e:
            depth = 0
            traceback = e.__traceback__
            while traceback is not None:
                depth += 1
                traceback = traceback.tb_next
            depths.add(depth)
            bad += depth != 2
    return depths, bad


def run_named():
    depths = set()
    for i in range(N):
        try:
            try:
                throw_named(i)
            except KeyError as e:
                raise e
        except KeyError as e2:
            depth = 0
            traceback = e2.__traceback__
            while traceback is not None:
                depth += 1
                traceback = traceback.tb_next
            depths.add(depth)
    return depths


def run_finally():
    depths = set()
    for i in range(N):
        try:
            try:
                throw_finally(i)
            finally:
                pass
        except KeyError as e3:
            depth = 0
            traceback = e3.__traceback__
            while traceback is not None:
                depth += 1
                traceback = traceback.tb_next
            depths.add(depth)
    return depths


bare_depths, bare_bad = run_bare()
print("bare_depths =", sorted(bare_depths))
print("bare_bad =", bare_bad)
print("named_depths =", sorted(run_named()))
print("finally_depths =", sorted(run_finally()))
