"""Python 3.14 ``__sizeof__`` surface for dict and set-like types."""

for typ in (dict, set, frozenset):
    assert "__sizeof__" in typ.__dict__
    assert typ.__sizeof__.__text_signature__ == "($self, /)"

assert dict().__sizeof__() == 48
assert {0: None}.__sizeof__() == 208
assert {str(i): None for i in range(6)}.__sizeof__() == 256
assert dict.fromkeys(range(11)).__sizeof__() == 616

for typ in (set, frozenset):
    assert typ().__sizeof__() == 200
    assert typ(range(4)).__sizeof__() == 200
    assert typ(range(5)).__sizeof__() == 712
    assert typ(range(19)).__sizeof__() == 2248

print("dict/set __sizeof__ Python 3.14 parity: ok")
