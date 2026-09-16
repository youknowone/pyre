# pyre-check: max-wasm-ratio=8.8
# Almost nothing but cross-loop JUMP. After the wasm32 stack-residency
# fix, peeled loops compile instead of declining, so every JUMP pays
# wasm's module-local br / jitframe-slot path where dynasm remaps
# LABEL values in registers. leftover-empty vable tails also GETFIELD
# the live/baked field list (`compile.py patch_new_loop_to_load_virtualizable_fields`).
# After rebase onto origin/main, darwin-arm64 measured 7.6x against
# dynasm; 8.8x is that reading plus WASM_RATIO_FIT_HEADROOM (15%).
# The 4x default ceiling is not raised.
# Fannkuch-Redux benchmark (The Computer Language Benchmarks Game)
# Ported for pyre: while-loop only, no range/list/enumerate

DEFAULT_ARG = 9

def fannkuch(n):
    p = [0] * n
    q = [0] * n
    s = [0] * n
    i = 0
    while i < n:
        p[i] = i
        q[i] = i
        s[i] = i
        i = i + 1

    maxflips = 0
    checksum = 0
    sign = 1

    while True:
        # count flips on copy q
        q0 = p[0]
        if q0 != 0:
            i = 1
            while i < n:
                q[i] = p[i]
                i = i + 1
            flips = 1
            while True:
                qq = q[q0]
                if qq == 0:
                    break
                q[q0] = q0
                if q0 >= 3:
                    i = 1
                    j = q0 - 1
                    while i < j:
                        t = q[i]
                        q[i] = q[j]
                        q[j] = t
                        i = i + 1
                        j = j - 1
                q0 = qq
                flips = flips + 1
            if flips > maxflips:
                maxflips = flips
            checksum = checksum + sign * flips

        # next permutation
        if sign == 1:
            t = p[0]
            p[0] = p[1]
            p[1] = t
            sign = -1
        else:
            t = p[1]
            p[1] = p[2]
            p[2] = t
            sign = 1
            i = 2
            while i < n:
                sx = s[i]
                if sx != 0:
                    s[i] = sx - 1
                    break
                if i == n - 1:
                    print(checksum)
                    return maxflips
                s[i] = i
                # rotate p[0..i+1]
                t = p[0]
                j = 0
                while j < i + 1:
                    p[j] = p[j + 1]
                    j = j + 1
                p[i + 1] = t
                i = i + 1


for i in range(3, 10):
    print(fannkuch(DEFAULT_ARG))
