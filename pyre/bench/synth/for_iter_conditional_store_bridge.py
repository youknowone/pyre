# pyre-check: max-pypy-ratio=7
# pyre-check: max-wasm-ratio=7.3
# leftover-empty vable tails now GETFIELD the live/baked field list
# (`compile.py patch_new_loop_to_load_virtualizable_fields`). dynasm
# turns those into native loads; wasm emits them as guest ops. darwin-arm64
# measured 6.3x against dynasm after rebase onto origin/main; 7.3x is
# that reading plus WASM_RATIO_FIT_HEADROOM (15%).
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
