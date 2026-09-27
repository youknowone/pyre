# pyre-check: pypy-diverges: structseqtype.__new__ asserts on a base, and sys.thread_info's repr prints the `name` field where the type name goes ("pthread(...)")
import os
import sys
import time

# A structseq type is not an acceptable base type.
for tp in (os.stat_result, time.struct_time, type(sys.flags)):
    try:
        class Derived(tp):
            pass
    except TypeError as e:
        assert "not an acceptable base type" in str(e), e
    else:
        raise AssertionError(f"{tp!r} accepted as a base")

# Refusing a structseq base leaves tuple itself subclassable.
class T(tuple):
    pass
assert T((1, 2)) == (1, 2)

# A type with a field called `name` still reports its own name.
assert repr(sys.thread_info).startswith("sys.thread_info(name="), repr(sys.thread_info)
assert repr(time.gmtime(0)).startswith("time.struct_time(tm_year=1970"), repr(time.gmtime(0))

print("OK")
