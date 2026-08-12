# pyre-check: no-cpython
import gc


message = "runtime-exception-group-message-" + str(id(gc))
leaf_message = "runtime-exception-group-leaf-" + str(id(message))
group = ExceptionGroup(message, [ValueError(leaf_message)])

ordinary_str = str(group)
direct_str = ExceptionGroup.__str__(group)
ordinary_repr = repr(group)
direct_repr = ExceptionGroup.__repr__(group)

assert ordinary_str == direct_str == message + " (1 sub-exception)"
assert ordinary_repr == direct_repr
assert message in ordinary_repr
assert leaf_message in ordinary_repr

for result in (ordinary_str, direct_str, ordinary_repr, direct_repr):
    assert any(obj is result for obj in gc.get_objects())

print("exception group render results are collectable")
