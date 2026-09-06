# pyre-check: gate=1
# Both descriptor kinds must retain their interpreter-visible header and
# payload when their bound method is virtualized, returned, or stored.
import gc


class C:
    def m(self):
        return 42


def exercise():
    obj = C()
    values = []
    python_method_type = type(obj.m)
    builtin_method_type = type(values.append)
    escaped = []
    i = 0
    while i < 4000:
        method = getattr(obj, "m")
        append = getattr(values, "append")
        assert type(method) is python_method_type
        assert type(append) is builtin_method_type
        assert method.__self__ is obj
        assert append.__self__ is values
        assert append.__module__ is None
        append(method())
        escaped.append((method, append))
        if i % 64 == 0:
            gc.collect()
            for old_method, old_append in escaped:
                assert old_method() == 42
                assert type(old_append) is builtin_method_type
                assert old_append.__module__ is None
                assert old_append.__self__ is values
            escaped.clear()
        i += 1
    assert len(values) == 4000
    assert sum(values) == 168000


exercise()
print("getattr_bound_method OK")
