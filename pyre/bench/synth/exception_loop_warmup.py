# pyre-check: max-pypy-ratio=6.5
# pyre-check: max-wasm-ratio=7.4
# Ubuntu run 34705838874: wasm/dynasm 6.4x. The oracle compiles 1 loop / 2
# bridges (`handle_fail` / `must_compile`); pyre matches. wasm materializes
# each as its own module. 7.4x is 6.4x plus WASM_RATIO_FIT_HEADROOM (15%).
# Warm-up-then-raise: the loop compiles clean, then `i >= 2000 and i % 19 == 0`
# starts raising. That guard is not in the recorded loop, so the first
# failures blackhole (`resume_in_blackhole`) until `must_compile` attaches
# the exception-edge bridge. Benches that raise from iteration 1 do not
# exercise this warmup-then-bridge path.
N = 47040000


def nested(n):
    acc = 0
    i = 0
    while i < n:
        v = 1
        try:
            try:
                if i >= 2000 and i % 19 == 0:
                    acc //= 0
                v = 2
            finally:
                v += 10
        except ZeroDivisionError:
            v += 100
        acc = (acc + v + i % 3) % 1000003
        i += 1
    return acc


def main():
    print(nested(N))


main()
