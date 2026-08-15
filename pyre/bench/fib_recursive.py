# pyre-check: max-wasm-ratio=3.6
# Reported 3.1x on ubuntu-24.04, and under the gate on the run before it.

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

for i in range(35):
    print(fib(i))
