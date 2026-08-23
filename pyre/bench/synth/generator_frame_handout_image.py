# pyre-check: selfcheck
# Self-checking regression guard for a frame handed to application code through
# `gi_frame` / `cr_frame` instead of `sys._getframe`.
#
# `getframe` (`module/sys/vm.rs`) does two things before it returns the frame:
# `mark_as_escaped`, then `force_frame`.  Its own comment states why the force
# comes first -- it is what makes the reads that follow see the JIT's live
# virtualizable fields, standing in for the `hook_access_field` injection
# upstream performs at each field read.  The generator, coroutine and
# async-generator frame getters (`generator_getter_for` field 2, `typedef.rs`)
# do neither: they hand back the frame pointer as it is.  Nothing pinned that
# the two routes to one frame agree.
#
# The discriminator is a SINGLE source line carrying both reads.  Both name the
# same line by construction, so a disagreement is the non-forcing route reading
# a stale `last_instr` -- a values-only check cannot see it, because each answer
# on its own looks like a plausible line number.
#
# The generator body carries an inner loop with no `yield` in it.  That is
# load-bearing: a loop that yields re-enters through the driver every iteration
# and never becomes a compiled loop, so the frame is never a traced
# virtualizable and the guard would pass vacuously.  `compiled_rounds` below
# fails the fixture if the inner loop stops being entered at all.
import sys

OUTER = 3000
INNER = 300


def gen(holder):
    disagreements = []
    identity_breaks = 0
    rounds = 0
    for _ in range(OUTER):
        for j in range(INNER):
            if j == INNER - 1:
                g = holder[0]
                rounds += 1
                f_handout, f_getframe = g.gi_frame, sys._getframe()
                if f_handout is not f_getframe:
                    identity_breaks += 1
                    continue
                a, b = f_handout.f_lineno, f_getframe.f_lineno
                if a != b:
                    disagreements.append((rounds, a, b))
        yield None
    holder.append((rounds, identity_breaks, disagreements))


def drive_generator():
    holder = []
    g = gen(holder)
    holder.append(g)
    for _ in g:
        pass
    return holder[1]


async def coro_body(holder):
    disagreements = []
    identity_breaks = 0
    rounds = 0
    for _ in range(OUTER):
        for j in range(INNER):
            if j == INNER - 1:
                c = holder[0]
                rounds += 1
                f_handout, f_getframe = c.cr_frame, sys._getframe()
                if f_handout is not f_getframe:
                    identity_breaks += 1
                    continue
                a, b = f_handout.f_lineno, f_getframe.f_lineno
                if a != b:
                    disagreements.append((rounds, a, b))
    return (rounds, identity_breaks, disagreements)


def drive_coroutine():
    holder = []
    c = coro_body(holder)
    holder.append(c)
    # Drive to completion without an event loop: the body never awaits, so the
    # first `send` runs it through and the StopIteration carries the result.
    try:
        c.send(None)
    except StopIteration as stop:
        return stop.value
    return None


def report(label, result):
    if result is None:
        print(f"FAIL {label}: body never ran to completion")
        return 1
    rounds, identity_breaks, disagreements = result
    if rounds != OUTER:
        print(f"FAIL {label}: inner loop entered {rounds} times, expected {OUTER}")
        return 1
    if identity_breaks:
        print(f"FAIL {label}: hand-out and _getframe named different frames "
              f"{identity_breaks} of {rounds} times")
        return 1
    if disagreements:
        print(f"FAIL {label}: f_lineno disagreed on one source line")
        print("  first five (round, handout, getframe):", disagreements[:5])
        return 1
    return 0


def main():
    rc = report("gi_frame", drive_generator())
    rc |= report("cr_frame", drive_coroutine())
    if rc:
        return rc
    print("PASS generator and coroutine frame hand-out image")
    return 0


sys.exit(main())
