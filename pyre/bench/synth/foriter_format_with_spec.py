# pyre-check: max-pypy-ratio=25
# `FORMAT_WITH_SPEC` in a `for` body. `f"{v:spec}"` with a non-empty spec
# lowers to one `format_with_spec` residual that pushes a fresh string, the
# same shape as the `FORMAT_SIMPLE` that `f"{v}"` emits; the body scan admitted
# only the latter, so a loop formatting with a spec stayed interpreted.
# Both spellings appear below so the two arms are exercised in one body.
# Output verified against CPython/PyPy.
N = 200000


def main():
    total = 0
    hits = 0
    for i in range(N):
        k = i % 97
        padded = f"{k:>5}"
        plain = f"{k}"
        total += len(padded) + len(plain)
        if padded.endswith("96"):
            hits += 1
    print(total, hits, f"{N - 1:>8}", f"{-0.0:z.2f}")


main()
