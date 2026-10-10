# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# A young send/throw payload is `w_arg_or_err` in `execute_frame` from
# `_invoke_execute_frame` through `call_trace` and `resume_execute_frame`.
# The generator must receive the same object that was sent or thrown.
def echo():
    v = None
    while True:
        try:
            v = yield v
        except ValueError as e:
            v = e.args[0]


def run():
    g = echo()
    g.send(None)
    ok = 0
    n = 0
    while n < 80:
        payload = [n, n + 1, n + 2]
        got = g.send(payload)
        if got is payload and got[0] == n and got[2] == n + 2:
            ok = ok + 1
        thrown = [n + 100, n + 101]
        got = g.throw(ValueError(thrown))
        if got is thrown and got[0] == n + 100 and got[1] == n + 101:
            ok = ok + 1
        n = n + 1
    return ok


result = run()
assert result == 160, result
print("PASS")
