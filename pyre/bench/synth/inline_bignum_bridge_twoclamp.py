# A guard-failure bridge re-enters an inlined helper containing exact-integer
# residual arithmetic.  The independent recurrence makes the clamp guard take
# a different pattern while the bignum hash keeps the helper residual live.
N = 1200
BIGP = 18446744073709551629
BASE = 1000003
P = 9223372036854775783


def mix(h, v):
    return (h * BASE + (v % BIGP)) % BIGP


h = 0
y = -7
z = 5
for i in range(N):
    z = z * 5 + (i & 7) - 3
    if z > 100000000000000000000 or z < -100000000000000000000:
        z = z % P
    y = y * 3 - (i & 15)
    if y > 100000000000000000000 or y < -100000000000000000000:
        y = y % P
    h = mix(h, y)

print(h)
