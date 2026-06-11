# Bound list methods flowing as values (`m = xs.append; m(i)`) dispatch the
# JIT's W_MethodObject method-form arms, which get NO concrete execution
# during tracing — their heap effect must stay deferred (no dm143 advance),
# unlike the builtin-form `xs.append(i)` shape which mutates concretely
# during trace and is advanced past the traced iteration.
N = 200000


def main():
    xs = []
    m_append = xs.append
    i = 0
    while i < N:
        m_append(i)
        i = i + 1

    acc = 0
    m_pop = xs.pop
    j = 0
    while j < N // 4:
        acc = acc + m_pop()
        j = j + 1

    k = 0
    while k < N // 16:
        acc = acc + m_pop(0)
        k = k + 1

    m_rev = xs.reverse
    m_rev()
    print(len(xs), acc, xs[0], xs[len(xs) - 1])


main()
