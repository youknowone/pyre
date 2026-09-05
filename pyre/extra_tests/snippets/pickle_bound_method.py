# Builtin getattr of a method stays residual. Folding it into a virtual
# Method lets pickle's REDUCE store that Method on the unpickler stack.
import pickle


class C:
    def m(self):
        return 42


o = C()
for proto in range(pickle.HIGHEST_PROTOCOL + 1):
    assert pickle.loads(pickle.dumps(o.m, proto))() == 42
    assert pickle.loads(pickle.dumps(C.m, proto))(o) == 42
print("pickle_bound_method OK")
