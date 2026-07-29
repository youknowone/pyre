import itertools
import sys


# Every one of these sizes its storage from a Python-supplied count, so an
# infallible reservation would abort the process rather than unwind.  CPython
# guards the same calls in test_itertools (test_combinations_overflow and
# friends), but those are @support.bigaddrspacetest and skip unless the suite
# is run with -M, so nothing exercises them in an ordinary run.
for label, call in [
    ("batched", lambda: next(itertools.batched([], sys.maxsize))),
    ("tee", lambda: itertools.tee([], sys.maxsize)),
    ("combinations", lambda: itertools.combinations([], sys.maxsize)),
    (
        "combinations_with_replacement",
        lambda: itertools.combinations_with_replacement([], sys.maxsize),
    ),
]:
    try:
        call()
    except MemoryError:
        pass
    else:
        raise AssertionError(f"{label} accepted an unsatisfiable size")

# product multiplies the pool count by `repeat`, so it rejects the size before
# reaching an allocation at all.
try:
    itertools.product([1], repeat=sys.maxsize)
except OverflowError as error:
    assert "repeat argument too large" in str(error)
else:
    raise AssertionError("product accepted an unsatisfiable repeat count")

print("OK")
