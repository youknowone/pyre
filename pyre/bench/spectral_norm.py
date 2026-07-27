# Spectral Norm benchmark (The Computer Language Benchmarks Game)
# Computes the spectral norm of an infinite matrix A
# where A[i][j] = 1/((i+j)(i+j+1)/2+i+1)

def multiply_Av(n, v, result):
    i = 0
    while i < n:
        s = 0.0
        j = 0
        while j < n:
            ij = i + j
            s = s + v[j] / (ij * (ij + 1) // 2 + i + 1)
            j = j + 1
        result[i] = s
        i = i + 1

def multiply_Atv(n, v, result):
    i = 0
    while i < n:
        s = 0.0
        j = 0
        while j < n:
            ij = j + i
            s = s + v[j] / (ij * (ij + 1) // 2 + j + 1)
            j = j + 1
        result[i] = s
        i = i + 1

def multiply_AtAv(n, v, result, tmp):
    multiply_Av(n, v, tmp)
    multiply_Atv(n, tmp, result)

n = 700
u = [1.0] * n
v = [0.0] * n
tmp = [0.0] * n

i = 0
# 30 rather than the reference 10: check.py gates on execution-only time
# (run minus the measured empty-program startup), and at 10 the pypy run is
# 0.026s against a 0.011s startup, so the denominator is the residue of two
# similar measurements and the ratio stops tracking pyre. 30 puts pypy at
# 0.077s, ~7x its startup.
while i < 30:
    multiply_AtAv(n, u, v, tmp)
    multiply_AtAv(n, v, u, tmp)
    i = i + 1

vBv = 0.0
vv = 0.0
i = 0
while i < n:
    vBv = vBv + u[i] * v[i]
    vv = vv + v[i] * v[i]
    i = i + 1

print((vBv / vv) ** 0.5)
