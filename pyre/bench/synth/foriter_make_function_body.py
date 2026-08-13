# MAKE_FUNCTION plus SET_FUNCTION_ATTRIBUTE in a hot FOR_ITER body. The default
# value forces the companion attribute initializer onto the definition path.
N = 20000


def main():
    total = 0
    for i in range(N):

        def add(value=i):
            return value + 1

        total += add()
    print(total)


main()
# Expected: 200010000
