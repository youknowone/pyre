# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# A bridge that resumes inside an inlined callee must keep running that callee
# on the frame the parent trace entered.
#
# `same` is inlined into `main`'s loop.  Switching `nat` from a list to a str
# fails the class guard inside `same`, and the bridge resumes `same` there.
# Its resume data carries `same`'s frame red, and the still-open
# `virtual_ref` scope names that frame, so `ExecutionContext.topframeref` does
# too.  When the bridge ran `same` on a second, freshly built frame instead,
# the traceback recorded the fresh frame, whose `f_back` read as `None`, while
# the raising callee's `f_back` named the other copy.  `traceback.clear_frames`
# then cleared frames that were still running.


def callee(start, stop):
    i = start
    while i < stop:
        i += 1
    raise ValueError


SEEN = {}


def census(e):
    tb = e.__traceback__
    caught = tb.tb_frame
    raised = tb.tb_next.tb_frame
    back = caught.f_back
    key = (
        raised.f_back is caught,
        back.f_code.co_name if back is not None else None,
    )
    SEEN[key] = SEEN.get(key, 0) + 1


def same(nat, letter, start, stop):
    try:
        nat.index(letter, start, stop)
    except ValueError:
        pass
    else:
        return
    try:
        callee(start, stop)
    except ValueError as e:
        census(e)


def main():
    for ty in list, str:
        nat = ty("abracadabra")
        for letter in "abcdrz":
            for start in range(-3, 14):
                for stop in range(-3, 14):
                    same(nat, letter, start, stop)


main()
assert SEEN == {(True, "main"): 2636}, SEEN
print("PASS")
