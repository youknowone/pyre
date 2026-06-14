N = 200000


def main():
    acc = 0
    i = 0
    while i < N:
        import math
        if math.pi > 3:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
