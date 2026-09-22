fstring_spec__N = 100000

def fstring_spec__main():
    acc = 0
    i = 0
    while i < fstring_spec__N:
        s = f'{i:05d}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
fstring_spec__main()

