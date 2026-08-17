# CPython-suite gap: `test_tarfile` reports 15 errors for this, all of them
# `tarfile.EOFHeaderError` escaping `TarFile.next()` even though the frame that
# did catch it agrees `type(e) is EOFHeaderError` and a re-raise there matches
# the clause that just refused it.
# parity-tests reason: the walker folds `CHECK_EXC_MATCH` to the boolean it
# computed while tracing and licenses the fold with a `GuardClass` on the
# exception's `ob_type`.  That is not the field the match read -- it read
# `typedef::type`, i.e. `w_class` -- and it does not separate the classes an
# `except` chain distinguishes: every exception instance carries the one
# `W_BaseException` payload whose vtable is chosen per `ExcKind`, so all the
# Python-level subclasses of one builtin base share it.  A chain traced on one
# of them keeps its folded booleans when another arrives, and the typed clause
# reads `False` for an exception it does match.

WARMUP = 12000
SWITCHED = 400


class ArchiveError(Exception):
    pass


class HeaderError(ArchiveError):
    pass


class EndOfArchive(HeaderError):
    pass


class Corrupt(ArchiveError):
    pass


def read_header(exc):
    # Raised from a callee, as `TarInfo._frombuf` is: the handler chain sees a
    # class the frame naming the clauses never mentions.
    raise exc


def step(exc):
    try:
        read_header(exc)
    except EndOfArchive:
        return "end"
    except HeaderError:
        return "header"
    except Exception:
        return "fallback"
    return "none"


# Distinct instances, so which class arrives is carried by the object rather
# than by a branch the trace would guard and side-exit on.
warm = [Corrupt("corrupt member") for _ in range(WARMUP)]
switched = [EndOfArchive("end of archive") for _ in range(SWITCHED)]
again = [Corrupt("corrupt member") for _ in range(WARMUP)]

counts = {"end": 0, "header": 0, "fallback": 0, "none": 0}

for exc in warm:
    counts[step(exc)] += 1
for exc in switched:
    counts[step(exc)] += 1
for exc in again:
    counts[step(exc)] += 1

print(sorted(counts.items()))
assert counts["end"] == SWITCHED, counts
assert counts["fallback"] == 2 * WARMUP, counts
assert counts["header"] == 0, counts
assert counts["none"] == 0, counts
print("OK")
