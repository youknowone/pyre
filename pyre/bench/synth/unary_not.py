N = 200000


def main():
    acc = 0
    i = 0
    while i < N:
        acc = acc + (not (i & 1))
        i = i + 1
    print(acc)


main()
