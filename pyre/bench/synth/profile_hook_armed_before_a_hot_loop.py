# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,root:callee
# The callee's `root:` arm is the premise, not a relaxation: the events this
# fixture counts are owed precisely because compiled code enters the callee past
# `execute_frame`'s bracket.  A callee that stopped reaching the JIT would make
# every count below pass without testing anything.
# A profile hook is installed, and a loop entered afterwards runs a tail long
# enough that a compiled one would have taken it over.
#
# This is the fixture for the gate `eval_with_jit_inner` keeps on profiled
# frames, and it is written to fail the moment that gate is narrowed without
# the reporting to back it up.  Three separate sources owe events here, each
# failing silently and independently:
#
#   * the loop frame's own `call`, from `executioncontext.py call_trace`, and
#     its `return`, from `leave`'s `_trace('leaveframe')`.  `pyframe.py
#     execute_frame` runs the first around the eval loop and the second in a
#     `finally`; pyre's portal sits at `execute_frame` level rather than at
#     `dispatch` where upstream's merge point is, so a portal that served this
#     frame would have to carry both itself;
#   * a callee's `call` / `return`, lost outright when a compiled trace inlines
#     the callee or jumps into its compiled entry, both of which begin at the
#     callee's first bytecode, past that bracket;
#   * a builtin's `c_call` / `c_return`, from `baseobjspace.py call_args`'s
#     profile arm.  Compiled code does not reach that arm at all — measured by
#     probing `call::c_profile_frame`, neither a folded builtin nor a residual
#     one goes through the interpreter's call doors — so this is the row that
#     keeps the gate: `is_being_profiled` is a green, and a profiled trace would
#     have to be RECORDED with the reporting in it.
#
# THE TAIL IS THE TEST for all three.  Each loss is total and permanent rather
# than partial: the moment a compiled trace takes the frame over the events
# stop, and no length of tail brings them back.  Measured with the gate
# narrowed, every count below saturated at the entry threshold — `c_call`
# stopped at 1041 for `len`, `ord`, `divmod`, `hex`, `sorted`, `max` and
# `round` alike — so each is checked at TWO tails and must track the difference
# between them.  A count that saturates fails however large it is.
#
# Only builtins are called here.  A class call would report `c_call` /
# `c_return` for `__new__` and `__init__`, which cpython does not, and that
# divergence is older than any of this (it reproduces at `PYRE_JIT=0`); keeping
# it out of the window leaves this fixture measuring only what it names.
import sys

WARM = 20000  # past the loop threshold (1039) many times over
TAILS = (2500, 10000)
NAME = 'abcdef'


def callee(x):
    return x + 1


def hot(n):
    total = 0
    for _ in range(n):
        total = callee(total) % 1000003
        total += len(NAME)
    return total


def measure(tail):
    counts = {}

    def hook(frame, event, arg):
        key = (event, getattr(arg, '__name__', frame.f_code.co_name))
        counts[key] = counts.get(key, 0) + 1
        return None

    # Compile the loop first, with nothing installed, so the arming below has
    # compiled code to interrupt rather than merely a cold frame to decline.
    hot(WARM)
    sys.setprofile(hook)
    try:
        hot(tail)
    finally:
        sys.setprofile(None)
    return counts


def exact(counts, key, expected, failures):
    got = counts.get(key, 0)
    if got != expected:
        failures.append('%s = %d, expected %d' % (key, got, expected))


def tracks_the_tail(short, long_, key, failures):
    grew = long_.get(key, 0) - short.get(key, 0)
    owed = TAILS[1] - TAILS[0]
    if grew < owed:
        failures.append(
            '%s grew by %d from tail %d to tail %d, owed %d — the count does '
            'not track the tail, so the frame is blind from some point onward'
            % (key, grew, TAILS[0], TAILS[1], owed)
        )


def main():
    failures = []
    short, long_ = (measure(tail) for tail in TAILS)
    for tail, counts in zip(TAILS, (short, long_)):
        # One activation of the loop frame, one of the callee per iteration,
        # one `len` per iteration.
        exact(counts, ('call', 'hot'), 1, failures)
        exact(counts, ('return', 'hot'), 1, failures)
        exact(counts, ('call', 'callee'), tail, failures)
        exact(counts, ('return', 'callee'), tail, failures)
        exact(counts, ('c_call', 'len'), tail, failures)
        exact(counts, ('c_return', 'len'), tail, failures)
    for key in (
        ('call', 'callee'),
        ('return', 'callee'),
        ('c_call', 'len'),
        ('c_return', 'len'),
    ):
        tracks_the_tail(short, long_, key, failures)
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a hot loop keeps reporting to a profile hook')
    return 0


sys.exit(main())
