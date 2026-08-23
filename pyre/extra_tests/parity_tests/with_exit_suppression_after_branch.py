# CPython-suite gap: the suite has no hot loop where a branch arm holds more
# than one suppressing `with`, so nothing exercises a second `__exit__` on a
# compiled path.
# parity-tests reason: this guards the blackhole's `guard_class` result, which
# a resumed exception match reads to pick the handler. With that register left
# unwritten the second block's `__exit__` was never reached and the raise
# escaped the loop.

"""A second suppressing `with` in a branch arm still runs its `__exit__`.

The shape needs both ingredients: a branch the loop takes only sometimes, and
a `with` whose `__exit__` has already handled an exception earlier in the same
arm. One `with`, or two with no branch, stayed correct throughout, so those
are kept here as the boundary.

Counting happens in the loop and the assertions run after it -- an assertion
inside the body reads the counters and stops the loop compiling, which would
leave the compiled path untested.
"""

ROUNDS = 20000


class Suppress:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.log.append(exc_type)
        return exc_type is ValueError


class Reraise:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def two_blocks_in_a_branch():
    """The failing shape: the second block's `__exit__` was skipped."""
    log = []
    cm = Suppress(log)
    taken = 0
    escaped = 0
    for i in range(ROUNDS):
        if i % 100 - 50 > 0:
            taken += 1
        else:
            try:
                with cm:
                    raise ValueError(1)
                with cm:
                    raise ValueError(2)
            except ValueError:
                escaped += 1
    return taken, escaped, len(log)


def three_blocks_in_a_branch():
    log = []
    cm = Suppress(log)
    escaped = 0
    for i in range(ROUNDS):
        if i % 100 - 50 > 0:
            continue
        try:
            with cm:
                raise ValueError(1)
            with cm:
                raise ValueError(2)
            with cm:
                raise ValueError(3)
        except ValueError:
            escaped += 1
    return escaped, len(log)


def branch_arm_hands_back_to_an_outer_handler():
    """A non-suppressing `__exit__` must still reach the enclosing `except`."""
    cm = Reraise()
    caught = 0
    for i in range(ROUNDS):
        if i % 100 - 50 > 0:
            continue
        try:
            with cm:
                raise ValueError(1)
        except ValueError:
            caught += 1
        try:
            with cm:
                raise ValueError(2)
        except ValueError:
            caught += 1
    return caught


def one_block_in_a_branch():
    """Boundary: correct before the fix, kept so a regression narrows."""
    log = []
    cm = Suppress(log)
    escaped = 0
    for i in range(ROUNDS):
        if i % 100 - 50 > 0:
            continue
        try:
            with cm:
                raise ValueError(1)
        except ValueError:
            escaped += 1
    return escaped, len(log)


def two_blocks_without_a_branch():
    """Boundary: correct before the fix, kept so a regression narrows."""
    log = []
    cm = Suppress(log)
    escaped = 0
    for _ in range(ROUNDS):
        try:
            with cm:
                raise ValueError(1)
            with cm:
                raise ValueError(2)
        except ValueError:
            escaped += 1
    return escaped, len(log)


taken, escaped_two, exits_two = two_blocks_in_a_branch()
escaped_three, exits_three = three_blocks_in_a_branch()
outer_caught = branch_arm_hands_back_to_an_outer_handler()
escaped_one, exits_one = one_block_in_a_branch()
escaped_flat, exits_flat = two_blocks_without_a_branch()

not_taken = ROUNDS - taken

assert taken == 9800, f"branch taken {taken}"
assert escaped_two == 0, f"two blocks let {escaped_two} escape"
assert exits_two == 2 * not_taken, f"two blocks ran {exits_two} exits, want {2 * not_taken}"

assert escaped_three == 0, f"three blocks let {escaped_three} escape"
assert exits_three == 3 * not_taken, f"three blocks ran {exits_three} exits"

assert outer_caught == 2 * not_taken, f"outer handler caught {outer_caught}"

assert escaped_one == 0, f"one block let {escaped_one} escape"
assert exits_one == not_taken, f"one block ran {exits_one} exits"

assert escaped_flat == 0, f"flat pair let {escaped_flat} escape"
assert exits_flat == 2 * ROUNDS, f"flat pair ran {exits_flat} exits"

print("OK")
