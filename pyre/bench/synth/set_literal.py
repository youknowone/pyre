N = 300000


def main():
    acc = 0
    i = 0
    while i < N:
        s = {i, i + 1, i + 2}
        acc = acc + len(s)
        i = i + 1
    print(acc)


main()
