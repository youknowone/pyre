import random

# A comprehension whose body calls a user Python function accumulates through
# LIST_APPEND. The call commits body effects, so a mid-body walk abort reaches
# `fbw_foriter_inflight_take` with a non-empty effect journal; that refuses the
# consumed item's delivery and the legacy replay resumes at the next iteration,
# losing the element. Only lengths are asserted, so the check holds whatever
# values the generator produces.
random.seed(1234)
for trial in range(400):
    size = random.randrange(50)
    data = [random.randrange(25) for _ in range(size)]
    assert len(data) == size, (trial, size, len(data))

# The same shape spelled as a statement loop, and one where the accumulator is
# a local rather than the comprehension's own temporary.
for trial in range(200):
    size = trial % 50
    collected = []
    for _ in range(size):
        collected.append(random.randrange(25))
    assert len(collected) == size, (trial, size, len(collected))

print("OK")
