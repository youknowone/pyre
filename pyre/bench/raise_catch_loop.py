def main():
    s = 0
    i = 0
    # Keep the timed loop long enough that every native backend's execution
    # time exceeds its empty-process startup cost by the benchmark gate's
    # health margin; otherwise startup subtraction dominates the ratio.
    while i < 500000000:
        try:
            if i % 7 == 0:
                raise ValueError("v")
            s = s + 1
        except ValueError:
            s = s + 2
        i = i + 1
    print(s)

main()
