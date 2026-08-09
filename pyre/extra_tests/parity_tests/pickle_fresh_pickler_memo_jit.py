import io
import pickle


# A fresh pure-Python Pickler must write the object before it can emit a memo
# reference.  The JIT used to trace BytesIO.write(), abort while descending
# into its void-returning check_closed(), and replay Pickler.dump() after the
# first execution had already populated memo.  The replay then emitted only a
# BINGET for a memo entry absent from the output stream.
for proto in range(pickle.HIGHEST_PROTOCOL + 1):
    for n in range(100):
        payload = bytes((n & 0xFF, proto))
        stream = io.BytesIO()
        pickler = pickle._Pickler(stream, protocol=proto)
        pickler.dump(payload)
        assert pickle.loads(stream.getvalue()) == payload

print("OK")
