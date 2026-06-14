N = 300000


class C:
    pass


def main():
    o = C()
    acc = 0
    i = 0
    while i < N:
        o.x = i
        acc = acc + o.x
        del o.x
        i = i + 1
    print(acc)


main()
