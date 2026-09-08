# pyre-check: max-wasm-ratio=5.5
# rustpython-common now compiles the xz-core lzma engine on every target,
# including the wasm guest.  That grows the module the wasm backend compiles
# per trace; fib_recursive is already bound by CALL_ASSEMBLER returns and
# sat on the 4x default ceiling.  ubuntu-24.04 measured 4.7x after the lzma
# share, so 5.5x is that reading plus WASM_RATIO_FIT_HEADROOM (15%).
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

for i in range(35):
    print(fib(i))
