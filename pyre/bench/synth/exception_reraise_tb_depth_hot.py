# pyre-check: max-pypy-ratio=4.3
# pyre-check: max-wasm-ratio=5.1
# leftover=[] extras past the mint abort compile (`compile.py`
# `assert i == len(inputargs)`). The hot re-raise then stays in the
# interpreter: cranelift measured 3.7x vs pypy, so 4.3x is that
# reading plus the same 1.15 headroom as the wasm ratio.
# `_opimpl_recursive_call` looks inside on bridges of a compiled portal
# (`pyjitpl.py`) until `max_unroll_recursion`. wasm compiles each extra
# trace as its own module; ubuntu-24.04 run 34810630624 measured 4.4x
# against dynasm, so 5.1x is that reading plus WASM_RATIO_FIT_HEADROOM.
# A bare re-raise caught in the same frame keeps the original traceback: no
# node is attached at a re-raise coordinate (RaiseWithExplicitTraceback,
# attach_tb=False). The module-level loop makes the recording iteration itself
# execute the re-raise chain, which historically prepended spurious nodes for
# the bare-raise and handler-cleanup coordinates on exactly that iteration
# (depth 4 instead of 2). Named re-raise (`raise e`) must still attach its
# node (depth 3), and a `finally` passthrough attaches nothing (depth 2).
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
