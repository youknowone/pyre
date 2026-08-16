# pyre-check: gate=1
class F(frozenset):
    pass
seed = frozenset([1, 2])
same = frozenset(seed) is seed
sub = F([1, 2, 3])
is_subtype = type(sub) is F
result = len(sub)

assert result == 3
assert same is True
assert is_subtype is True
