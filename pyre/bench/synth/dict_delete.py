N = 300000


def main():
    i = 0
    acc = 0
    while i < N:
        d = {0: i, 1: i + 1, 2: i + 2}
        del d[1]
        acc = acc + d[0] + d[2] + len(d)
        i = i + 1
    print(acc)


main()
