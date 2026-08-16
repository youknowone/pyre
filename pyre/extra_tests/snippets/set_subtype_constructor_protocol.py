# pyre-check: gate=1
class S(set):
    pass
s = S([1, 2, 3])
manual = set()
set.__init__(manual, [4, 5])
is_subtype = type(s) is S
result = len(s)
manual_result = len(manual)

assert result == 3
assert manual_result == 2
assert is_subtype is True
