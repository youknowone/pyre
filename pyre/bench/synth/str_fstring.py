# pyre-check: max-pypy-ratio=45
# Merged synth parity smoke suite: independent feature-level hot loops, each
# kept verbatim from its former standalone file with module-level names prefixed
# by the source name. Bug-repro / resume / kept-stack tests are NOT merged (they
# stay isolated so a miscompile is not diluted). check.py runs every *.py here.

# ── fstring_simple ──
fstring_simple__N = 200000

def fstring_simple__main():
    acc = 0
    i = 0
    while i < fstring_simple__N:
        s = f'{i}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
fstring_simple__main()

# ── fstring_multi ──
fstring_multi__N = 200000

def fstring_multi__main():
    acc = 0
    i = 0
    while i < fstring_multi__N:
        s = f'{i}-{i}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
fstring_multi__main()

# ── fstring_spec ──
fstring_spec__N = 200000

def fstring_spec__main():
    acc = 0
    i = 0
    while i < fstring_spec__N:
        s = f'{i:05d}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
fstring_spec__main()

# ── convert_value ──
convert_value__N = 200000

def convert_value__main():
    acc = 0
    i = 0
    while i < convert_value__N:
        s = f'{i!r}-{i!s}-{i!a}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
convert_value__main()

# ── string_ops ──
string_ops__N = 250000

def string_ops__main():
    words = ['alpha', 'beta', 'gamma', 'delta']
    i = 0
    acc = 0
    while i < string_ops__N:
        s = words[i & 3]
        t = s + ':' + str(i & 255)
        if t.startswith('a') or t.endswith('7'):
            acc = acc + len(t)
        else:
            acc = acc - len(s)
        i = i + 1
    print(acc)
string_ops__main()

# ── bytes_ops ──
bytes_ops__N = 350000

def bytes_ops__main():
    data = b'abcdefghijklmnopqrstuvwxyz'
    i = 0
    acc = 0
    while i < bytes_ops__N:
        b = data[i % len(data)]
        piece = data[i & 7:(i & 7) + 5]
        acc = acc + b + len(piece)
        i = i + 1
    print(acc)
bytes_ops__main()
