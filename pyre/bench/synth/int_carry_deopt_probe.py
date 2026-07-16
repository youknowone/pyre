# Loop-carried int accumulator that deopts mid-loop via a type-flip guard.
N = 200000


def main():
    acc = 0
    i = 0
    while i < N:
        x = i % 5
        if i == 123457:
            x = "s"          # forces a guard failure / bridge with acc live
        if isinstance(x, int):
            acc = acc + x
        i = i + 1
    print(acc)


main()
