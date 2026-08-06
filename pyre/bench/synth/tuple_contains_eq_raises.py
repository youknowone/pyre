# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# containment on a 2-tuple compares elements with `__eq__`; an exception it
# raises propagates rather than being swallowed into a False result.
def main():
    class E:
        def __eq__(self, o):
            raise ValueError("boom")

        def __hash__(self):
            return 1

    for probe in ("(E(), 2)", "(E(), 2, 3)"):
        try:
            print(1 in eval(probe))
        except ValueError as e:
            print("propagated", e)


main()
