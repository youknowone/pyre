# pyre-check: max-pypy-ratio=26
# pyre-check: jitstats-band=guard_failures=16
# The windows-latest dynasm leg reads 619, 619 and 610 guard
# failures (runs 36192328967, 36254696576, 36316164602) against 605 on macos
# and ubuntu, with the same seven loops and four bridges. The count moves
# between runs of one commit there. On macOS it stays 605 across nursery
# sizes 256KB-16MB, PYPY_GC_MIN / PYPY_GC_MAJOR_COLLECT / PYPY_GC_GROWTH and
# PYTHONHASHSEED, and N = 8000 also reads 605, so the extra failures come
# after iteration 8000 on that host only.
# A frame that raises and catches in one body contributes EXACTLY ONE
# traceback node, and it keeps that count once the loop is compiled.
#
# The raise routes through walk()'s SubRaise arm, which scans the raising
# frame for its own `catch_exception/L` and jumps into the handler.  Both the
# `raise/r` arm and that walk-catch arm can record a node for the same frame,
# so the two have to divide the work: the raise arm emits the node the
# compiled trace runs, the walk-catch arm applies the node for the recording
# pass.  Either one taking both halves gives the frame two nodes; either one
# taking neither drops the frame from the chain.
#
# Surveying over N iterations is what catches it — the pre-compile iterations
# are correct, so a single sample sees nothing and a miscount shows up only as
# a SECOND tuple in the shape set.
#
# `reraise` covers the bare-`raise` spelling, which must preserve the node the
# original raise attached rather than adding one of its own, and the callee
# case pins the two-frame chain the same run has to keep producing.
N = 380953


def chain(traceback):
    out = []
    while traceback is not None:
        out.append((traceback.tb_frame.f_code.co_name, traceback.tb_lineno))
        traceback = traceback.tb_next
    return tuple(out)


def raise_here(i):
    try:
        raise ValueError(i)
    except ValueError as e:
        return chain(e.__traceback__)


def reraise_here(i):
    try:
        try:
            raise KeyError(i)
        except KeyError:
            raise
    except KeyError as e:
        return chain(e.__traceback__)


def callee_raise(i):
    raise TypeError(i)


def caught_from_callee(i):
    try:
        callee_raise(i)
    except TypeError as e:
        return chain(e.__traceback__)


def survey(name, fn):
    seen = set()
    for i in range(N):
        seen.add(fn(i))
    print(name, sorted(seen))


survey("raise_here   ", raise_here)
survey("reraise_here ", reraise_here)
survey("from_callee  ", caught_from_callee)
