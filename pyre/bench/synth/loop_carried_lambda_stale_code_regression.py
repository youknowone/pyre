# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=f
# A compiled loop must not keep using a lambda that a loop-carried local
# no longer holds.  `f` peels the inner `for x in "ab"` loop while the
# pair iterator still yields `(add, +)`, then a bridge unpacks `(sub, -)`
# and jumps back to that peeled label.  The peeled body inlines `numberop`
# from a heap fact about the label argument (`Function.code` is the add
# lambda).  Every entry to the label has to re-establish that fact;
# exporting an unforced short-preamble `PreambleOp` as its Const payload
# lets the "next x" bridge inherit it without a `Function.code` guard, so
# the add lambda runs for the sub lambda.
#
#     target/release/pyre-dynasm pyre/bench/synth/loop_carried_lambda_stale_code_regression.py
#
# Correct: fails == 0.  The pre-fix binary reports 3832.


def add(p, q):
    r = {}
    for k in p:
        r[k] = p[k] + q[k]
    return r


def sub(p, q):
    r = {}
    for k in p:
        r[k] = p[k] - q[k]
    return r


def f():
    fails = 0
    p = {"a": 3, "b": 1}
    q = {"a": 1, "b": 2}
    for i in range(3000):
        for counterop, numberop in [
            (add, lambda x, y: x + y),
            (sub, lambda x, y: x - y),
        ]:
            result = counterop(p, q)
            for x in "ab":
                if numberop(p[x], q[x]) != result[x]:
                    fails += 1
    return fails


fails = f()
if fails == 0:
    print("PASS loop-carried lambda keeps matching Function.code")
else:
    print(f"FAIL stale lambda: {fails} mismatches")
    raise SystemExit(1)
