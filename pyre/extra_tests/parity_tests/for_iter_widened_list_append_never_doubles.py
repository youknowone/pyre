# CPython-suite gap: no CPython test observes how often a comprehension body runs.
# parity-tests reason: this pins the widened LIST_APPEND admission as exactly-once.

# `LIST_APPEND` bodies are admitted whatever the body does. A mid-body walk
# abort must therefore neither drop the consumed item nor re-run the body over
# it. Count the calls as well as the elements: a drop shows up in the length, a
# double-apply only in the counter. The conditional makes half the elements
# take the call arm, so the loop guard-fails on the branch and the abort paths
# are the ones under test.

calls = [0]


def uf(x):
    calls[0] += 1
    return x


expected = list(range(500))
for trial in range(200):
    calls[0] = 0
    out = [uf(x) if x < 250 else x for x in range(500)]
    assert len(out) == 500, (trial, len(out))
    assert calls[0] == 250, (trial, calls[0])
    assert out == expected, trial

# The same shape where the accumulator is a named local rather than the
# comprehension's own temporary.
for trial in range(200):
    calls[0] = 0
    collected = []
    for x in range(500):
        collected.append(uf(x) if x < 250 else x)
    assert len(collected) == 500, (trial, len(collected))
    assert calls[0] == 250, (trial, calls[0])
    assert collected == expected, trial

print("OK")
