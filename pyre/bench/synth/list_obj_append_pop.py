N = 700000


def main():
    xs = []
    i = 0
    acc = 0
    s = "x"
    t = "y"
    while i < N:
        xs.append(s)
        if len(xs) > 32:
            xs.pop(0)
        if (i & 7) == 0:
            xs.append(t)
            xs.pop()
        i = i + 1
    print(len(xs) + acc)


main()
