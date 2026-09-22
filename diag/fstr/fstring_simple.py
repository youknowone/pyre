fstring_simple__N = 100000

def fstring_simple__main():
    acc = 0
    i = 0
    while i < fstring_simple__N:
        s = f'{i}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
fstring_simple__main()

