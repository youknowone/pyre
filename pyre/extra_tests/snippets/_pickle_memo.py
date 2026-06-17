# A reused Pickler keeps its memo across dump() calls (until clear_memo),
# so a second dump of the same object emits a back-reference rather than a
# fresh copy, and the two unpickled objects share identity. clear_memo()
# resets that. The Unpickler memo persists symmetrically across load() calls.
import io
import _pickle
import pickle

for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
    shared = ["x", "y"]
    buf = io.BytesIO()
    p = _pickle.Pickler(buf, proto)
    p.dump(shared)
    len1 = buf.tell()
    p.dump(shared)  # second dump should reference the memoized list
    len2 = buf.tell() - len1
    # The back-referenced second pickle is much smaller than the first.
    assert len2 < len1, (proto, len1, len2)

    # Load both from the same stream; identity is preserved across dumps.
    buf.seek(0)
    up = _pickle.Unpickler(buf)
    a = up.load()
    b = up.load()
    assert a is b, (proto, "cross-dump identity lost")
    assert a == ["x", "y"], (proto, a)

    # clear_memo() makes the next dump independent (full copy again).
    buf2 = io.BytesIO()
    p2 = _pickle.Pickler(buf2, proto)
    p2.dump(shared)
    n1 = buf2.tell()
    p2.clear_memo()
    p2.dump(shared)
    n2 = buf2.tell() - n1
    # After clear_memo the object is serialized in full again.
    assert n2 == n1, (proto, n1, n2)
