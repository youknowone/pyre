# pyre-check: max-pypy-ratio=7
# pyre-check: max-wasm-ratio=5.0
# ubuntu-24.04 run 34737668364 measured 4.3x wasm/dynasm (0.80s / 0.19s).
# 5.0x is that reading plus WASM_RATIO_FIT_HEADROOM (15%).
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
