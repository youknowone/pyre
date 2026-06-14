N = 200000


def main():
    acc = 0
    i = 0
    while i < N:
        s = f"{i}"
        acc = acc + len(s)
        i = i + 1
    print(acc)


main()
