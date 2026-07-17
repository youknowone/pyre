# _pickle.Pickler/Unpickler accept their constructor arguments positionally and
# by keyword: tp_new allocates and ignores them, __init__ validates and stores
# them. `pickle.Pickler` binds to the C accelerator where it exists and to the
# pure-Python class otherwise; both agree on the observable results asserted
# here (protocol given positionally or by keyword, fix_imports, out-of-band
# buffers via buffer_callback / the Unpickler buffers argument, and the
# protocol<5 buffer_callback rejection).
import io
import pickle
from pickle import Pickler, Unpickler, PickleBuffer


def roundtrip(obj, protocol):
    data = io.BytesIO()
    Pickler(data, protocol).dump(obj)
    data.seek(0)
    return Unpickler(data).load()


def roundtrip_kw(obj, protocol, fix_imports):
    data = io.BytesIO()
    Pickler(data, protocol=protocol, fix_imports=fix_imports).dump(obj)
    data.seek(0)
    return Unpickler(data, fix_imports=fix_imports).load()


def oob_roundtrip(obj, protocol):
    buffers = []
    data = io.BytesIO()
    Pickler(data, protocol, buffer_callback=buffers.append).dump(obj)
    data.seek(0)
    restored = Unpickler(data, buffers=iter(buffers)).load()
    return bytes(restored), [bytes(b) for b in buffers]


def warm(n):
    acc = 0
    for i in range(n):
        acc += roundtrip(i % 7, 2)
        acc += len(roundtrip([i % 3, i % 5], 4))
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__)


def main():
    print("warm", warm(15000))
    m("proto_positional", lambda: roundtrip("abc", 2))
    m("proto_positional_hi", lambda: roundtrip(("x", "y", "z"), 5))
    m("proto_keyword", lambda: roundtrip_kw({"a": 1, "b": 2}, 4, True))
    m("fix_imports_false", lambda: roundtrip_kw([1, 2, 3], 2, False))
    m("oob_bytes", lambda: oob_roundtrip(PickleBuffer(b"hello"), 5))
    m("oob_bytearray", lambda: oob_roundtrip(PickleBuffer(bytearray(b"world")), 5))
    # buffer_callback requires protocol >= 5
    m("cb_low_proto", lambda: Pickler(io.BytesIO(), 4, buffer_callback=lambda b: None))
    # file must expose a write attribute
    m("no_write_attr", lambda: Pickler(object(), 2))


main()
