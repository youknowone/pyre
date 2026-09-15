# pyre-check: max-pypy-ratio=5.0
# A MutableCell builtins slot unwraps through generated `unwrap_cell`.
# Until that helper's `ob_type` compares fold on the baked cell pointer,
# the compiled loop is a few ns/iter slower than pypy (dynasm 3.4x,
# cranelift 4.4x on a fast pypy). The recovery is that fold, not a
# hand-written getfield.
# pyre-check: spec-folds=builtin_len_descent
# pyre-check: skip-cpython
# A module-scope LOAD_NAME whose name misses the module dict resolves through
# the frame's builtin module.  The builtins cell folds under the module dict's
# version? (so a later global binding shadows the builtin) and the builtins
# dict's own version?.  The second loop proves that invalidation is seen: the
# census reads `builtin_len_descent` consulted twice and fired once, the decline being
# the shadowed call.

N = 225000000
M = 4000000
s = "xx"

total = 0
for i in range(N):
    total = total + len(s)
print(total)

len = lambda x: 100
shadowed = 0
for i in range(M):
    shadowed = shadowed + len(s)
print(shadowed)
