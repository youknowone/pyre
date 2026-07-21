# An inlined list comprehension whose LIST_APPEND element is a non-empty nested
# list (`[[i] …]` / `[[i, i + 1] …]`). The #171 fold virtualizes the inner list,
# whose separately allocated backing block (NewArray / NewArrayClear) carries no
# jitcode-liveness color. Once the trace-time single-executor forks were retired
# the append body no longer runs under a speculative-replay sub-walk, so the
# backing block is bound at every guard-exit deopt without an extra resume-data
# root; the shape is admitted by default (PYRE_NESTED_LIST_FOLD_VIRT, default-on).
#
# Acceptance repro for that fold: it must print the same total on all three
# backends (dynasm / cranelift / wasm). Set PYRE_NESTED_LIST_FOLD_VIRT=0 to fall
# back to the `for_iter_bodies_all_jit_safe` interpreter decline (native only —
# the wasm guest cannot read the env var).


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
