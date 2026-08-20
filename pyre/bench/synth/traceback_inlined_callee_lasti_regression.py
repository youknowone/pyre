# pyre-check: selfcheck
# Self-checking regression guard for the coordinate an UNWOUND frame reports
# after an exception passed through it.  Sibling of
# `frame_lineno_mid_replay_regression`, which guards the same coordinate for a
# frame that is still RUNNING; this one guards the frame the exception left.
#
# `dispatch_bytecode` (pyopcode.py) stamps `last_instr` before every opcode, so
# `pyopcode.py:147-148 handle_operation_error` always builds its traceback node
# while the frame holds the instruction that raised, and `tb_frame.f_lasti`
# answers for it forever after.  A frame the walker seeded for an inlined callee
# takes no such store: the walk emits no per-opcode `last_instr` for a frame that
# is not the virtualizable, and `publish_last_instr_at_live_marker` fires only
# for instructions the blackhole actually replays -- a level the exception merely
# passes through replays none.  Such a frame reached the traceback still holding
# the `-1` constructor sentinel (`f_lasti == -2`) or the `pc - 1` resume
# coordinate a walk-end flush left (the position AFTER the call, where the
# interpreter reports the call itself).
#
# The node's own `tb_lasti` was right the whole time -- every recorder resolves
# the coordinate before using it -- so `tb_lasti` is the oracle and the assertion
# is `f_lasti == tb_lasti` on every frame the exception has already left.  The
# head node is excluded: it names the frame running the handler, which is still
# advancing and legitimately reports a later position.
#
# Two exception classes into one hot try/except so the second class compiles as
# an exception-edge bridge, and an inlined intermediate frame between the loop
# and the raise, are both load-bearing: the divergence lives on the seeded
# callee frame, and it took the bridge to reach a recorder that does not stamp.
# Sorting a SET of shapes is what makes it a test -- a per-iteration divergence
# shows up as extra elements rather than as one shifted value.
import sys

N = 60000


def chain(e):
    out = []
    tb = e.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        out.append((frame.f_code.co_name, tb.tb_lasti, frame.f_lasti))
        tb = tb.tb_next
    return tuple(out)


def leaf_two(i):
    if i % 3 == 1:
        raise ValueError(i)
    if i % 3 == 2:
        raise TypeError(i)
    return i


def mid_two(i):
    return leaf_two(i)


def a_bridge_two_classes():
    shapes = set()
    acc = 0
    for i in range(N):
        try:
            acc += mid_two(i)
        except ValueError as e:
            shapes.add(("V",) + chain(e))
        except TypeError as e:
            shapes.add(("T",) + chain(e))
    return shapes


def main():
    shapes = a_bridge_two_classes()
    failures = []

    # `shape[2:]` drops the class tag and the head node: the head names
    # `a_bridge_two_classes`, which is executing the handler this reads from.
    stale = sorted(
        {node for shape in shapes for node in shape[2:] if node[2] != node[1]}
    )
    if stale:
        failures.append(f"unwound frame f_lasti != tb_lasti: {stale}")

    names = sorted(
        (shape[0],) + tuple(node[0] for node in shape[1:]) for shape in shapes
    )
    want = [
        ("T", "a_bridge_two_classes", "mid_two", "leaf_two"),
        ("V", "a_bridge_two_classes", "mid_two", "leaf_two"),
    ]
    if names != want:
        failures.append(f"traceback names: got {names!r}, want {want!r}")

    # One shape per class.  A coordinate that varies across iterations
    # multiplies the shapes even when each value is individually plausible.
    if len(shapes) != 2:
        failures.append(f"expected 2 shapes, got {len(shapes)}: {sorted(shapes)!r}")

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS unwound inlined-callee traceback coordinates")
    return 0


sys.exit(main())
