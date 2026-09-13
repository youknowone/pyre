# pyre-check: max-pypy-ratio=7
# pyre-check: max-wasm-ratio=5.6
# Ubuntu run 34705838874: wasm/dynasm 4.8x. The oracle compiles 1 loop / 3
# bridges; pyre matches. wasm materializes each as its own module. 5.6x is
# 4.8x plus WASM_RATIO_FIT_HEADROOM (15%).
def loop_with_two_backedges(n):
    high = 0
    for i in range(n):
        value = i % 7
        high = value if value > high else high
        if i % 45 == 0:
            high = 0
    return high


result = loop_with_two_backedges(10952000)
assert result == 6
print(result)
