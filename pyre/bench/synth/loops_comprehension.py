# pyre-check: max-pypy-ratio=30
# pyre-check: min-pypy-ratio=2.52
# Merged synth parity smoke suite: independent feature-level hot loops, each
# kept verbatim from its former standalone file with module-level names prefixed
# by the source name. Bug-repro / resume / kept-stack tests are NOT merged (they
# stay isolated so a miscompile is not diluted). check.py runs every *.py here.

# ── comprehensions ──
comprehensions__N = 70000

def comprehensions__main():
    i = 0
    acc = 0
    while i < comprehensions__N:
        xs = [j * j for j in range(8) if j + i & 1 == 0]
        d = {j: j + i for j in xs}
        acc = acc + sum(xs) + len(d)
        i = i + 1
    print(acc)
comprehensions__main()

# ── for_range_loop ──
for_range_loop__N = 600000

def for_range_loop__main():
    acc = 0
    for i in range(for_range_loop__N):
        acc = acc + i % 19
    for i in range(10, 200000, 3):
        acc = acc - i % 11
    print(acc)
for_range_loop__main()

# ── while_nested_break_continue ──
while_nested_break_continue__N = 60000

def while_nested_break_continue__main():
    i = 0
    acc = 0
    while i < while_nested_break_continue__N:
        j = 0
        while j < 12:
            if j == 5:
                j = j + 1
                continue
            if j == 10:
                break
            acc = acc + (i + j) % 23
            j = j + 1
        i = i + 1
    print(acc)
while_nested_break_continue__main()

# ── nested_foriter_sum ──
def nested_foriter_sum__main():
    total = 0
    n = 0
    while n < 20000:
        for x in [1, 2, 3, 4, 5]:
            for y in range(x):
                total += y
        n += 1
    return total
nested_foriter_sum__result = nested_foriter_sum__main()
print(nested_foriter_sum__result)
