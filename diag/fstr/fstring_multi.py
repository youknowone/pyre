fstring_multi__N = 100000

def fstring_multi__main():
    acc = 0
    i = 0
    while i < fstring_multi__N:
        s = f'{i}-{i}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
fstring_multi__main()

