# pyre-check: gate=1
# A sequence pattern must not match a string.  All three implementations agree
# on the behaviour -- `pyopcode.py:MATCH_SEQUENCE` rejects unicode, bytes and
# bytearray after the collection flag has already said "sequence" -- but they
# disagree on what `__flags__` publishes: `W_TypeObject.get_flags` reports the
# bit the `_abc` registration set, while the 3.14 surface leaves it clear on
# exactly the three types the opcode excludes.  `__flags__` is
# caller-observable, so the bit follows the spec and only the internal marker
# keeps PyPy's value.
#
# `bench/synth/pypy_type_surface.py` carries no row for these three types for
# that reason, and points here instead.

SEQUENCE = 1 << 5


def is_sequence_pattern(value):
    match value:
        case [*_]:
            return True
        case _:
            return False


for value in ("ab", b"ab", bytearray(b"ab")):
    assert not is_sequence_pattern(value), type(value).__name__
    assert type(value).__flags__ & SEQUENCE == 0, type(value).__name__

for value in ([1, 2], (1, 2), range(2), memoryview(b"ab")):
    assert is_sequence_pattern(value), type(value).__name__

# list and tuple are the two the collection flag is set on directly rather
# than through an `_abc` registration, so they are the pair whose published
# bit is the same everywhere.
assert list.__flags__ & SEQUENCE == SEQUENCE
assert tuple.__flags__ & SEQUENCE == SEQUENCE


# A str subclass inherits the exclusion; the pattern still must not match.
class S(str):
    pass


assert not is_sequence_pattern(S("ab"))
assert S.__flags__ & SEQUENCE == 0
