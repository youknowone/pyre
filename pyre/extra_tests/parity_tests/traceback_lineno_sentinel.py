# pyre-check: pypy-diverges: pins tb_lineno=-1 meaning "resolve from tb_lasti"; pypy3 answers -1 instead of the line
# CPython-suite gap: `test_traceback` builds `TracebackType` objects, but every
# one it builds passes a real line number.  A runtime that stored the fourth
# argument and handed it straight back would pass the whole module.
#
# parity-tests reason: `tb_lineno` is not a stored field.  `-1` means "I have
# not resolved this yet, take the line from tb_lasti", and that sentinel is
# app-level reachable through the constructor, so what a runtime does with it
# is observable.  A runtime carrying a different sentinel -- PyPy uses
# `-sys.maxsize-1` -- answers `-1` here instead of the line, and a runtime that
# resolves eagerly at raise time has nothing to resolve when the constructor
# hands it one.
#
# The rule is not "negative means resolve": `-2` and `0` are stored and read
# back as they are.  Only `-1` resolves, and only against `tb_lasti`: an offset
# that names no instruction answers `None`, and a negative offset answers the
# code object's first line, which is where a frame that has not run an
# instruction sits.
#
# PyPy 7.3.20 fails every `rebuilt` case: its sentinel is `-sys.maxsize-1`, so
# a `-1` handed to the constructor reads back as `-1`.  It also accepts writes
# to `tb_lineno` and `tb_lasti`, which is the other half of the same choice --
# it took the most negative value as its sentinel precisely because a written
# line number could otherwise collide with it.


def raiser():
    raise ValueError('x')


def caller():
    raiser()


def natural():
    """The traceback of a two-frame raise, innermost node first."""
    try:
        caller()
    except ValueError as exc:
        tb = exc.__traceback__
    nodes = []
    while tb is not None:
        nodes.append(tb)
        tb = tb.tb_next
    # `natural`'s own frame is the outermost node; drop it so the fixture does
    # not pin the line the `caller()` call above happens to sit on.
    return nodes[1:]


NODES = natural()
TracebackType = type(NODES[0])


def rebuilt(node, lasti, lineno):
    """`node` reconstructed with a different offset and line number."""
    return TracebackType(None, node.tb_frame, lasti, lineno)


def a_recorded_node_reports_the_line_it_froze_at():
    lines = [node.tb_lineno for node in NODES]
    print('recorded:', lines)
    # The body line of each function, derived rather than written out so the
    # fixture survives an edit to the header above it.
    expected = [caller.__code__.co_firstlineno + 1, raiser.__code__.co_firstlineno + 1]
    assert lines == expected, (lines, expected)


def the_sentinel_resolves_against_tb_lasti():
    # Every node rebuilt with `-1` has to come back with the line it already
    # reported: the offset is unchanged and the sentinel means "use it".
    for node in NODES:
        again = rebuilt(node, node.tb_lasti, -1)
        print('rebuilt:', again.tb_lineno)
        assert again.tb_lineno == node.tb_lineno, (again.tb_lineno, node.tb_lineno)


def an_offset_naming_no_instruction_resolves_to_none():
    node = rebuilt(NODES[0], 1 << 30, -1)
    print('out of range:', node.tb_lineno)
    assert node.tb_lineno is None, node.tb_lineno


def a_negative_offset_resolves_to_the_first_line():
    firstlineno = NODES[0].tb_frame.f_code.co_firstlineno
    for lasti in (-1, -2):
        node = rebuilt(NODES[0], lasti, -1)
        print('negative offset:', node.tb_lineno)
        assert node.tb_lineno == firstlineno, (node.tb_lineno, firstlineno)


def only_tb_next_is_writable():
    # With `tb_lineno` resolving rather than storing, a writable slot would let
    # a `-1` written back read as a computed line -- an answer neither
    # reference gives.  Only `tb_next` takes an assignment.
    node = NODES[0]
    for name, value in (('tb_lineno', 5), ('tb_lasti', 5), ('tb_frame', None)):
        try:
            setattr(node, name, value)
        except AttributeError as exc:
            print(name, type(exc).__name__)
        else:
            print(name, 'writable')
    node.tb_next = None
    print('tb_next:', node.tb_next)


def only_minus_one_is_the_sentinel():
    # `-2` and `0` are stored line numbers, however implausible; the
    # constructor is documented to take them and nothing resolves them.
    for lineno in (-2, 0, 7):
        node = rebuilt(NODES[0], NODES[0].tb_lasti, lineno)
        print('stored:', node.tb_lineno)
        assert node.tb_lineno == lineno, (node.tb_lineno, lineno)


a_recorded_node_reports_the_line_it_froze_at()
the_sentinel_resolves_against_tb_lasti()
an_offset_naming_no_instruction_resolves_to_none()
a_negative_offset_resolves_to_the_first_line()
only_minus_one_is_the_sentinel()
only_tb_next_is_writable()
print('OK')
