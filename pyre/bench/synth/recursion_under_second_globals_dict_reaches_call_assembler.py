# pyre-check: max-pypy-ratio=5.4
#
# Floor is 5.4 / 4 = 1.35x. CI (native legs, runs 41939478616 and
# 41964689814) measured cranelift 3.0x-3.4x (ubuntu) and 2.4x-2.6x
# (macOS arm64), dynasm 2.4x-3.3x (ubuntu, windows) and 2.2x-2.4x (macOS
# arm64); local macOS arm64 dynasm 2.0x-2.1x. Ceiling = 1.6 x the largest.
# Before the fix the workload ran 13-28s, far past this ceiling.
#
# One code object is exec'd first under a module dict, publishing
# PyCode.w_globals, then under a plain dict. A function whose code
# already recorded a different first globals used to residualize every
# call as opaque bh_call_fn; the inline / CALL_ASSEMBLER frame now
# carries FrameDebugData.w_globals the way PyFrame.__init__ does.
# `ll_portal_runner` consults warmstate at every portal entry, including
# mid-body greens after CALL_ASSEMBLER. ubuntu/macOS/windows dynasm,
# cranelift and wasm then record three loops, 17 bridges and 3993 guard
# failures. PyPy records one loop for these two namespaces (five bridges
# in the measured oracle).
import types

SRC = """
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

total = 0
for i in range(29):
    total += fib(i)
"""


def main():
    code = compile(SRC, "<fib>", "exec")
    g1 = types.ModuleType("m").__dict__
    exec(code, g1)
    print(g1["total"])
    g2 = {}
    exec(code, g2)
    print(g2["total"])
    print("PASS")


main()
