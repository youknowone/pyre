# Follow-up coverage for the interp-level `_pickle` accelerator: legacy
# string opcodes with the encoding/errors decode path, DUP, LONG4 byte-count
# validation, copyreg extension codes (EXT1/EXT2/EXT4), reducer_override,
# dispatch_table, fast mode, the Unpickler.find_class override hook, and the
# dump-time global-resolution verification. Behaviors are pinned to CPython
# 3.14.
import io
import copyreg
import pickle
import _pickle


def loads(data, **kw):
    return _pickle.Unpickler(io.BytesIO(data), **kw).load()


def dumps(obj, proto):
    buf = io.BytesIO()
    _pickle.Pickler(buf, proto).dump(obj)
    return buf.getvalue()


# --- legacy STRING / BINSTRING / SHORT_BINSTRING (Python 2 wire) ------------
# STRING: 'S' + repr-quoted text + newline, decoded with the unpickler's
# encoding (default "ASCII").
assert loads(b"S'abc'\n.") == "abc"
assert loads(b'S"abc"\n.') == "abc"
# encoding="bytes" keeps the raw bytes instead of decoding.
assert loads(b"S'abc'\n.", encoding="bytes") == b"abc"
# escape sequences inside the quoted argument.
assert loads(b"S'a\\nb'\n.") == "a\nb"
assert loads(b"S'\\x41'\n.") == "A"
# SHORT_BINSTRING: 'U' + 1-byte length + bytes.
assert loads(b"U\x03abc.") == "abc"
assert loads(b"U\x03abc.", encoding="bytes") == b"abc"
# BINSTRING: 'T' + 4-byte little-endian length + bytes.
assert loads(b"T\x03\x00\x00\x00abc.") == "abc"
# an unquoted STRING argument is rejected.
try:
    loads(b"Sabc\n.")
    raise AssertionError("unquoted STRING accepted")
except pickle.UnpicklingError:
    pass

# --- DUP duplicates the top of stack (same object) -------------------------
# MARK EMPTY_LIST DUP TUPLE STOP -> ([], []) where both entries are identical.
g = loads(b"(]2t.")
assert g == ([], [])
assert g[0] is g[1]

# --- LONG4 negative byte count is rejected ---------------------------------
try:
    loads(b"\x8b\xff\xff\xff\xff.")
    raise AssertionError("negative LONG4 accepted")
except pickle.UnpicklingError as e:
    assert "negative byte count" in str(e), e


# --- copyreg extension codes (EXT1 / EXT2 / EXT4) --------------------------
class Ext1:
    pass


class Ext2:
    pass


class Ext4:
    pass


# Register module/name -> code so save_global emits an EXT opcode and the
# unpickler resolves it back through copyreg + find_class.
mod = __name__
for cls, code in ((Ext1, 0xF0), (Ext2, 0x1234), (Ext4, 0x12345)):
    copyreg.add_extension(mod, cls.__name__, code)
try:
    for cls in (Ext1, Ext2, Ext4):
        assert loads(dumps(cls, 2)) is cls, cls.__name__
        # EXT opcodes are a protocol >= 2 feature.
        assert loads(dumps(cls, 4)) is cls, cls.__name__
finally:
    for cls, code in ((Ext1, 0xF0), (Ext2, 0x1234), (Ext4, 0x12345)):
        copyreg.remove_extension(mod, cls.__name__, code)


# --- reducer_override on a Pickler subclass --------------------------------
class Wrapped:
    def __init__(self, v):
        self.v = v


def rebuild_wrapped(v):
    return Wrapped(v)


class OverridePickler(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, Wrapped):
            # +100 marks that the override (not the default __dict__ reduce) ran.
            return (rebuild_wrapped, (obj.v + 100,))
        return NotImplemented


buf = io.BytesIO()
OverridePickler(buf, 2).dump(Wrapped(5))
assert pickle.loads(buf.getvalue()).v == 105


# --- dispatch_table (per-pickler reduce override) --------------------------
class Boxed:
    def __init__(self, x):
        self.x = x


def reduce_boxed(obj):
    return (Boxed, (obj.x,))


buf = io.BytesIO()
p = pickle.Pickler(buf, 2)
# Unset by default: reading it raises AttributeError (T_OBJECT_EX member).
try:
    p.dispatch_table
    raise AssertionError("dispatch_table readable when unset")
except AttributeError:
    pass
p.dispatch_table = {Boxed: reduce_boxed}
p.dump(Boxed(7))
assert pickle.loads(buf.getvalue()).x == 7


# --- fast mode disables the memo (no shared identity) ----------------------
shared = [1, 2, 3]
buf = io.BytesIO()
pf = pickle.Pickler(buf, 2)
assert pf.fast == 0
pf.fast = 1
assert pf.fast == 1
pf.dump([shared, shared])
fast_g = pickle.loads(buf.getvalue())
assert fast_g == [[1, 2, 3], [1, 2, 3]]
assert fast_g[0] is not fast_g[1]
# without fast mode the memo preserves identity.
buf = io.BytesIO()
pickle.Pickler(buf, 2).dump([shared, shared])
slow_g = pickle.loads(buf.getvalue())
assert slow_g[0] is slow_g[1]


# --- Unpickler.find_class override + super() -------------------------------
class GuardedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("builtins", "eval"):
            raise pickle.UnpicklingError("eval blocked")
        return super().find_class(module, name)


# proto >= 4 keeps the module name as "builtins" (no fix_imports remap).
try:
    GuardedUnpickler(io.BytesIO(pickle.dumps(eval, 4))).load()
    raise AssertionError("eval was not blocked")
except pickle.UnpicklingError as e:
    assert "eval blocked" in str(e), e
assert GuardedUnpickler(io.BytesIO(pickle.dumps(len, 4))).load() is len


# --- dump-time verification of global resolution ---------------------------
# A function-local class cannot be referenced by a dotted path.
def make_local():
    class Local:
        pass

    return Local


try:
    dumps(make_local(), 2)
    raise AssertionError("local class pickled")
except pickle.PicklingError:
    pass


# An object whose name resolves to a different object is rejected.
class Shadow:
    pass


_real_shadow = Shadow
globals()["Shadow"] = "not the class"
try:
    dumps(_real_shadow, 2)
    raise AssertionError("shadowed class pickled")
except pickle.PicklingError as e:
    assert "not the same object" in str(e) or "not found" in str(e), e
finally:
    globals()["Shadow"] = _real_shadow


print("_pickle_followups OK")
