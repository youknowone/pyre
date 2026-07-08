# Adversarial companion to kept_stack_deep_var_shortcircuit: the deep
# operand-stack Variables g(i)/h(i) are held across a `p or q` short-circuit
# guard EXACTLY as in that bench, but g and h each MUTATE a shared global list
# (a non-journaled STORE_SUBSCR-class heap effect committed inside a user
# frame).  A FOR_ITER trace that consumes the iterator, aborts on the deep
# kept-stack guard, and then DELIVERS the in-flight item would re-run the body
# and DOUBLE the mutation.  The `log` length must equal the iteration count
# exactly (2 mutations per iteration): a doubled delivery over-counts, a
# dropped iteration under-counts.
log = []


def g(i):
    log.append(i)
    return i * 2 + 1


def h(i):
    log.append(-i)
    return i * 3 - 1


def f(n):
    s = 0
    for i in range(n):
        p = (i % 4) != 0
        q = (i % 5) != 0
        t = (g(i), h(i), (g(i) if (p or q) else h(i)))
        s += t[0] - t[1] + t[2]
    return s


r = f(40000)
print(r)
print(len(log))
