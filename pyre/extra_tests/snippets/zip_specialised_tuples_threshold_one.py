# pyre-check: gate=1
# A 2-loop traced at threshold 1 iterates zip() of an arity-2 int tuple and an
# arity-2 object tuple. Both are specialised layouts with no items array.
try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


def main():
    tails = (3, 5)
    counts = ({"a": 1}, {"b": 2})
    seen = []
    for tail, count in zip(tails, counts):
        seen.append((tail, sorted(count)))
    assert seen == [(3, ["a"]), (5, ["b"])], seen
    print("PASS")


if __name__ == "__main__":
    main()
