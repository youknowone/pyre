# pyre-check: max-pypy-ratio=20
# A fresh empty list promoted by append has a trace-allocated typed backing
# block. The following subscript may revisit that symbolic block through
# W_ListObject.int_items.block before compiled execution gives it a real pointer.


def f(n):
    acc = 0
    for i in range(n):
        l = []
        l.append(i & 1)
        acc += l[0]
    return acc


print(f(300000))
