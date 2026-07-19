# `int(s, 0)` reports the offending literal verbatim, keeping any surrounding
# whitespace, rather than the internally trimmed value.
def main():
    for s in ("   ", "  x  ", " 0b12 "):
        try:
            int(s, 0)
        except ValueError as e:
            print(str(e))


main()
