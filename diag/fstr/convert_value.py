convert_value__N = 100000

def convert_value__main():
    acc = 0
    i = 0
    while i < convert_value__N:
        s = f'{i!r}-{i!s}-{i!a}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
convert_value__main()

