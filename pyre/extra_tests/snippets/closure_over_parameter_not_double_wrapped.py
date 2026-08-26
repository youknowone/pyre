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

# A cell passed as the captured argument is still an ordinary Python value.
# The new closure must therefore contain a distinct outer cell whose contents
# are the argument cell, rather than mistaking that argument for its own
# closure container.  PyPy `PyFrame.init_cells` gets this from its separate
# argument/cell slots; pyre's unified locals-plus slot must preserve it too.
def external_cell():
    value = 42

    def inner():
        return value

    return inner.__closure__[0]


cell_ext = external_cell()
def capture(arg):
    def read():
        return arg

    return read


read = capture(cell_ext)
cell_closure = read.__closure__[0]
assert read() is cell_ext
assert cell_closure is not cell_ext
