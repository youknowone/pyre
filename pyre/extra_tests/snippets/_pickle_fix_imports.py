# `fix_imports` gates the `_compat_pickle` name remap that protocols < 3 apply
# between Python 2 and Python 3 global names. It defaults to True and is a
# no-op at protocol >= 3.
import io
import _pickle
import pickle


def dumps(obj, proto, **kw):
    buf = io.BytesIO()
    _pickle.Pickler(buf, proto, **kw).dump(obj)
    return buf.getvalue()


def loads(data, **kw):
    return _pickle.Unpickler(io.BytesIO(data), **kw).load()


# ── dump: proto 2 remaps `builtins` -> `__builtin__` unless fix_imports=False ──
assert dumps(len, 2) == b"\x80\x02c__builtin__\nlen\nq\x00."
assert dumps(len, 2, fix_imports=True) == b"\x80\x02c__builtin__\nlen\nq\x00."
assert dumps(len, 2, fix_imports=False) == b"\x80\x02cbuiltins\nlen\nq\x00."
# protocol 3 writes the Python 3 name verbatim regardless of fix_imports.
assert dumps(len, 3) == b"\x80\x03cbuiltins\nlen\nq\x00."
assert dumps(len, 3, fix_imports=False) == b"\x80\x03cbuiltins\nlen\nq\x00."

# Module-level _pickle.dumps mirrors the class behavior.
assert _pickle.dumps(len, 2) == b"\x80\x02c__builtin__\nlen\nq\x00."
assert _pickle.dumps(len, 2, fix_imports=False) == b"\x80\x02cbuiltins\nlen\nq\x00."

# ── load: fix_imports gates the forward (py2 -> py3) remap ──
py2_stream = dumps(len, 2, fix_imports=True)   # __builtin__\nlen
py3_stream = dumps(len, 2, fix_imports=False)  # builtins\nlen
assert loads(py2_stream, fix_imports=True) is len
assert loads(py2_stream) is len  # default True
assert loads(py3_stream, fix_imports=False) is len
assert _pickle.loads(py2_stream) is len
assert _pickle.loads(py3_stream, fix_imports=False) is len

# A Python 2 module name with fix_imports=False is resolved literally, so the
# nonexistent `__builtin__` module fails to import.
try:
    loads(py2_stream, fix_imports=False)
    raise AssertionError("expected an import failure")
except ImportError:
    pass

print("_pickle_fix_imports OK")
