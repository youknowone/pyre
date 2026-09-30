# CPython-suite gap: no test checks that BytesIO/StringIO instance attributes
# live on the typed payload rather than a generic instance dict.
# parity-tests reason: W_IOBase.getdict stores attributes in w_dict.

"""Typed _io payloads keep instance attributes on W_IOBase.w_dict."""

import copy
import io
import pickle


def check(label, stream):
    stream.x = 1
    print("get", label, stream.x)
    print("vars-before-del", label, sorted(vars(stream).items()))
    print("contains", label, "x" in stream.__dict__)
    print("identity", label, stream.__dict__ is stream.__dict__)
    del stream.x
    print("after-del", label, "x" in stream.__dict__, sorted(vars(stream).items()))
    try:
        stream.__dict__ = {}
    except Exception:
        print("assign", label, "raises")
    else:
        print("assign", label, "accepted")


class B(io.BytesIO):
    pass


class S(io.StringIO):
    pass


check("BytesIO", io.BytesIO())
check("StringIO", io.StringIO())
check("TextIOWrapper", io.TextIOWrapper(io.BytesIO()))
check("BufferedReader", io.BufferedReader(io.BytesIO()))
check("BufferedWriter", io.BufferedWriter(io.BytesIO()))
check("BufferedRandom", io.BufferedRandom(io.BytesIO()))
check("BufferedRWPair", io.BufferedRWPair(io.BytesIO(), io.BytesIO()))
check("B", B())
check("S", S())

raw = io.BytesIO(b"abc")
raw.x = 7
copied = copy.copy(raw)
print("copy BytesIO", copied.x, copied.getvalue(), type(copied).__name__)
restored = pickle.loads(pickle.dumps(raw))
print("pickle BytesIO", restored.x, restored.getvalue(), type(restored).__name__)

text = io.StringIO("abc")
text.x = 7
copied = copy.copy(text)
print("copy StringIO", copied.x, copied.getvalue(), type(copied).__name__)
restored = pickle.loads(pickle.dumps(text))
print("pickle StringIO", restored.x, restored.getvalue(), type(restored).__name__)

fresh = io.BytesIO()
print("fresh-id", fresh.__dict__ is fresh.__dict__, sorted(vars(fresh).items()))
print("OK")
