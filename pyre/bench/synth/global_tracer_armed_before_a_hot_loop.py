# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,root:callee
# The callee's `root:` arm is the premise, not a relaxation: the events this
# fixture counts are owed precisely because compiled code enters the callee past
# `execute_frame`'s bracket.  A callee that stopped reaching the JIT would make
# every count below pass without testing anything.
# A global trace function is installed while a loop is already compiled, and
# the loop keeps running.
#
# `executioncontext.py settrace` does not turn the JIT off — it keeps it on and
# widens `trace_limit` — so pyre's portal must serve a traced frame rather than
# hand every one of them to the plain evaluator.  Three separate paths owe
# events once it does, and each is exercised here because each fails silently
# and independently:
#
#   * the loop frame's own `call` / `return`, which `pyframe.py execute_frame`
#     brackets the eval loop with.  pyre's portal sits at `execute_frame` level
#     rather than at `dispatch` where upstream's merge point is, so the bracket
#     is carried by the portal itself;
#   * a callee's `call` / `return`, which are lost outright when a compiled
#     trace inlines the callee or jumps into its compiled entry, both of which
#     begin at the callee's first bytecode, past the bracket;
#   * a frame that the hook arms local tracing on, whose `line` events come
#     from the dispatch loop's `bytecode_trace` and so cannot come from
#     compiled code at all.
#
# THE TAIL IS THE TEST for all three.  Each loss is total and permanent rather
# than partial: the moment a compiled trace takes the frame over, the events
# stop and no length of tail brings them back.  Measured before the fixes: the
# callee's `call` events stopped at 1043 whatever the tail, and with the
# `call`/`return` half repaired its `line` events stopped at 1239 — the entry
# threshold — equally regardless of tail.  So every count below is checked at
# TWO tails and must track the difference between them; a count that saturates
# fails however large it is.
#
# The `local` arm's hook returns itself, which arms `f_trace` on every frame it
# reports; the `global` arm returns None, which arms nothing and leaves `call`
# as the only event owed.  Both spellings matter: the second is the one that
# still compiles, and the first is the one that must not.
#
# `line` counts are lower bounds, not equalities.  The rule a line event fires
# on is `lastline != lineno or frame.last_instr < d.instr_prev_plus_one`, and
# an implementation whose loop back edge lands lower fires the header a second
# time — cpython 3.14.6 and pypy3 disagree with each other here (4003 against
# 5597 on a 2000-iteration loop) and neither is wrong.  `call` and `return` are
# exact: both references agree on those, per call, at every tail.
import sys

WARM = 20000  # past the loop threshold (1039) many times over
TAILS = (2500, 10000)


def callee(x):
    return x + 1


def hot(n):
    total = 0
    for _ in range(n):
        total = callee(total) % 1000003
    return total


def measure(tail, local):
    counts = {}
    chain = {}

    def hook(frame, event, arg):
        key = (event, frame.f_code.co_name)
        counts[key] = counts.get(key, 0) + 1
        # The caller chain the hook can see, recorded once per key so the
        # forcing it costs is paid a fixed number of times rather than per
        # event.  A frame reported from compiled code is reported by the
        # portal rather than by the eval loop, so what it answers for
        # `f_back` and for `sys._getframe` is a separate question from
        # whether it reports at all.
        if event == 'call' and key not in chain:
            back = frame.f_back
            here = sys._getframe(1)
            chain[key] = (
                back.f_code.co_name if back is not None else '<none>',
                here.f_code.co_name if here is not None else '<none>',
            )
        return hook if local else None

    # Compile the loop first, with nothing installed, so the arming below has
    # compiled code to interrupt rather than merely a cold frame to decline.
    hot(WARM)
    sys.settrace(hook)
    try:
        hot(tail)
    finally:
        sys.settrace(None)
    return counts, chain


def exact(counts, key, expected, arm, failures):
    got = counts.get(key, 0)
    if got != expected:
        failures.append('%s: %s = %d, expected %d' % (arm, key, got, expected))


def at_least(counts, key, floor, arm, failures):
    got = counts.get(key, 0)
    if got < floor:
        failures.append('%s: %s = %d, expected at least %d' % (arm, key, got, floor))


def tracks_the_tail(short, long_, key, arm, failures):
    grew = long_.get(key, 0) - short.get(key, 0)
    owed = TAILS[1] - TAILS[0]
    if grew < owed:
        failures.append(
            '%s: %s grew by %d from tail %d to tail %d, owed %d — the count does '
            'not track the tail, so the frame is blind from some point onward'
            % (arm, key, grew, TAILS[0], TAILS[1], owed)
        )


def names(chain, key, expected, arm, failures):
    got = chain.get(key)
    if got != expected:
        failures.append(
            '%s: the caller chain at %s reads %r, expected %r — a frame reported '
            'from compiled code answers about a chain it never established'
            % (arm, key, got, expected)
        )


def main():
    failures = []
    for local in (False, True):
        arm = 'local' if local else 'global'
        measured = [measure(tail, local) for tail in TAILS]
        short, long_ = (counts for counts, _ in measured)
        for tail, (counts, chain) in zip(TAILS, measured):
            # `hot` is called from `measure`, `callee` from `hot`, and the frame
            # the hook is running on top of is the one it was handed.
            names(chain, ('call', 'hot'), ('measure', 'hot'), arm, failures)
            names(chain, ('call', 'callee'), ('hot', 'callee'), arm, failures)
            # One activation of the loop frame, one of the callee per iteration.
            exact(counts, ('call', 'hot'), 1, arm, failures)
            exact(counts, ('call', 'callee'), tail, arm, failures)
            if local:
                # `return` is delivered through the frame's own `f_trace`, so it
                # is owed exactly where `call` is and only on this arm.
                exact(counts, ('return', 'hot'), 1, arm, failures)
                exact(counts, ('return', 'callee'), tail, arm, failures)
                at_least(counts, ('line', 'callee'), tail, arm, failures)
                at_least(counts, ('line', 'hot'), 2 * tail, arm, failures)
            else:
                # A hook that returns None arms no frame, so nothing but `call`.
                exact(counts, ('return', 'hot'), 0, arm, failures)
                exact(counts, ('line', 'hot'), 0, arm, failures)
        keys = [('call', 'callee')]
        if local:
            keys += [('return', 'callee'), ('line', 'callee'), ('line', 'hot')]
        for key in keys:
            tracks_the_tail(short, long_, key, arm, failures)
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a hot loop keeps reporting to a global trace function')
    return 0


sys.exit(main())
