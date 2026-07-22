# pyre-check: max-pypy-ratio=30
# A hot call whose positional argument is a literal (`build(1000)`) into a
# callee that grows a list over `range(n)`. The list's realloc-boundary append
# declines the FBW fold, so a guard resumes the caller frame at the CALL site.
# The literal argument is loaded into the walker's unboxed int bank, so the
# Ref-only operand mirror left its stack slot a NONE hole; `stack_sync` then
# omitted it and the single-frame bridge rebuilt the argument slot NULL. The
# re-executed CALL passed NULL, so the callee's parameter `n` reconstructed
# unbound -> `UnboundLocalError`. The comprehension and explicit-append forms
# and int / object element strategies each exercise the same caller-frame
# resume; every form must reach `n` (never a dropped element or an unbound
# local).


def comp(n):
    return [i for i in range(n)]


def obj_comp(n):
    return [(i, i) for i in range(n)]


def build(n):
    r = []
    for i in range(n):
        r.append(i)
    return r


total = 0
k = 0
while k < 400:
    total += len(comp(1000))
    total += len(obj_comp(1000))
    total += len(build(1000))
    k += 1
print(total)
