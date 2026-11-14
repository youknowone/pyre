# pyre-check: max-pypy-ratio=11.8
#
# Floor is 11.8 / 4 = 2.95x. CI (native legs, runs 41939478616 and
# 41964689814) measured cranelift 6.8x-7.4x (ubuntu) and 5.3x-5.6x
# (macOS arm64), dynasm 4.5x-6.4x (ubuntu, windows) and 4.4x-4.7x (macOS
# arm64); local macOS arm64 dynasm 4.0x-4.1x. Ceiling = 1.6 x the largest.
# Before the recursive call folded to CALL_ASSEMBLER here, the same
# workload ran about 8x slower than that.
#
# A self-recursive function defined and driven in code executed with a
# plain `{}` as globals.  Its `LOAD_GLOBAL` cannot fold to a constant, so
# the walker runs the globals lookup as a residual; an executed residual
# earlier in the body must not stop the recursive call at the inline bound
# from becoming a `CALL_ASSEMBLER`.
#
# Without that, every recursive call past the bound re-entered the portal
# through a residual call, ~60x slower than the same code under a module
# dict.
SRC = '''
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

total = 0
for i in range(30):
    total += fib(i)
'''


def main():
    g = {}
    exec(compile(SRC, "<fib>", "exec"), g)
    print(g["total"])


main()
