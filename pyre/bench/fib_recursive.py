# The 5.5x wasm/dynasm allowance was ubuntu-24.04's 4.7x after the lzma
# share, plus WASM_RATIO_FIT_HEADROOM. Three later ubuntu-24.04 runs
# (35491772991, 35547723188, 35592344587) read 3.1x, 2.9x, 2.8x, all
# inside 4.0 with that headroom, so the allowance is gone.
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

for i in range(35):
    print(fib(i))
