# pyre-check: max-pypy-ratio=44
# pyre-check: min-pypy-ratio=4.65
# An inlined list comprehension whose LIST_APPEND element is a non-empty nested
# list (`[[i] …]` / `[[i, i + 1] …]`). The #171 fold virtualizes the inner list,
# whose separately allocated backing block (NewArray / NewArrayClear) carries no
# jitcode-liveness color. Once the trace-time single-executor forks were retired
# the append body no longer runs under a speculative-replay sub-walk, so the
# backing block is bound at every guard-exit deopt without an extra resume-data
# root.
#
# Acceptance repro for that fold: it must print the same total on all three
# backends (dynasm / cranelift / wasm).


def single_comp(n):
    return [[i] for i in range(n)]


def pair_comp(n):
    return [[i, i + 1] for i in range(n)]


def main():
    total = 0
    k = 0
    while k < 500:
        xs = single_comp(1000)
        total += len(xs)
        total += xs[-1][0]
        ys = pair_comp(1000)
        total += len(ys)
        total += ys[-1][1]
        k += 1
    print(total)


main()
