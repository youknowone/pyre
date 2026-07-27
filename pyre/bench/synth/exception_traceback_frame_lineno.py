# A traceback that outlives its frame keeps the frame answering for the line it
# stopped on, not the line it was defined at.
#
# `f_lineno` resolves through `offset2lineno(pycode, last_instr)`, and compiled
# code does not run the interpreter's per-opcode `last_instr` store, so the
# value only reaches the frame if the trace publishes it.  A frame the
# function-entry portal compiled starts from the `-1` initialization sentinel,
# which `offset2lineno` answers with the code object's first line, so a missing
# publish reports the `def` line.
#
# Both frame exits have to publish, and the return exit is the one that is easy
# to miss: `catches_here` leaves by RETURN, not by the raise, because it keeps
# running after it catches.  The last line it reaches is the `return`, so that
# is the line its traceback has to report.
#
# The driver decides which route compiles the callee - a `while` loop reaches it
# as a function-entry portal, a `for` loop inlines it into the loop trace - so
# the two are surveyed separately and must agree.  A publish wired into only one
# route shows up as a disagreement between the two lines without the oracle
# having to say anything.
#
# Surveying every iteration rather than sampling the last one is what catches a
# frame that is only sometimes right: the pre-compile iterations are correct, so
# a miss appears as a SECOND tuple in the shape set.
#
# Offsets run from `co_firstlineno` so edits above these functions do not move
# the expected values.
N = 4000


def chain(traceback):
    out = []
    while traceback is not None:
        frame = traceback.tb_frame
        base = frame.f_code.co_firstlineno
        out.append(
            (
                frame.f_code.co_name,
                traceback.tb_lineno - base,
                frame.f_lineno - base,
            )
        )
        traceback = traceback.tb_next
    return tuple(out)


def catches_here(i):
    try:
        raise ValueError(i)
    except ValueError as e:
        return e.__traceback__


def raises_out(i):
    raise KeyError(i)


def catches_callee(i):
    try:
        raises_out(i)
    except KeyError as e:
        return e.__traceback__


def while_same():
    seen = set()
    k = 0
    while k < N:
        seen.add(chain(catches_here(k)))
        k += 1
    return sorted(seen)


def for_same():
    seen = set()
    for k in range(N):
        seen.add(chain(catches_here(k)))
    return sorted(seen)


def while_callee():
    seen = set()
    k = 0
    while k < N:
        seen.add(chain(catches_callee(k)))
        k += 1
    return sorted(seen)


def for_callee():
    seen = set()
    for k in range(N):
        seen.add(chain(catches_callee(k)))
    return sorted(seen)


def kept_alive():
    """Many tracebacks alive at once: a shared or recycled frame collapses."""
    kept = []
    k = 0
    while k < N:
        kept.append(catches_here(k))
        k += 1
    shapes = sorted({chain(t) for t in kept})
    distinct = len({id(t.tb_frame) for t in kept}) == len(kept)
    return shapes, distinct


print("while/same  ", while_same())
print("for/same    ", for_same())
print("while/callee", while_callee())
print("for/callee  ", for_callee())
kept_shapes, kept_distinct = kept_alive()
print("kept        ", kept_shapes)
print("kept_distinct", kept_distinct)
