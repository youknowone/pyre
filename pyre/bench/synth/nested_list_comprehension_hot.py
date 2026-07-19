# An inlined list comprehension whose LIST_APPEND element is a non-empty nested
# list (`[[i] …]` / `[[i, i + 1] …]`). The #171 fold virtualizes the inner list,
# but its separately allocated backing block (NewArray / NewArrayClear) carries
# no jitcode-liveness color, so it is not rooted in the append commit sub-walk's
# guard-exit resume data and a deopt resolves it to a null OpRef.
#
# Held back behind the DEFAULT-OFF PYRE_NESTED_LIST_FOLD_VIRT gate: while the
# gate is off `for_iter_bodies_all_jit_safe` declines the shape, so this runs in
# the interpreter and prints the correct total. It is the acceptance repro for
# the recursive virtual-list-forcing rooting work — with the gate on it must
# still print the same total on all three backends (dynasm / cranelift / wasm)
# once the rooting lands.


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
