# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# `int(s, 0)` reports the offending literal verbatim, keeping any surrounding
# whitespace, rather than the internally trimmed value.
def main():
    for s in ("   ", "  x  ", " 0b12 "):
        try:
            int(s, 0)
        except ValueError as e:
            print(str(e))


main()
