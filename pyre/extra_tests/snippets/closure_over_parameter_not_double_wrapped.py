# pyre-check: gate=1
# An argument slot promoted to a cellvar (captured by an inner
# function) must wrap to a single cell, not a cell-of-cell, so the
# closure reads the value rather than an inner cell.
def make_adder(n):
    def add(x):
        return x + n
    return add
result = make_adder(10)(5)

assert result == 15
