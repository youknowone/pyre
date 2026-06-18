# Pickler.memo / Unpickler.memo expose a live proxy of the memo (a fresh
# PicklerMemoProxy / UnpicklerMemoProxy on every access). The proxy's copy()
# snapshots the memo as a dict and clear() empties it; the property is settable
# from a matching proxy or a dict, but not deletable. Pinned to CPython 3.14.
import io
import _pickle
import pickle


def err(fn):
    try:
        fn()
        return None
    except Exception as e:  # noqa: BLE001
        return e


def dump(obj, proto):
    buf = io.BytesIO()
    _pickle.Pickler(buf, proto).dump(obj)
    return buf.getvalue()


# ── Pickler.memo proxy ────────────────────────────────────────────────────
buf = io.BytesIO()
p = _pickle.Pickler(buf, 5)
p.dump(["a", "b"])

proxy = p.memo
assert type(proxy).__name__ == "PicklerMemoProxy", type(proxy).__name__
# A fresh proxy is handed back on each access.
assert p.memo is not p.memo

snap = proxy.copy()
assert isinstance(snap, dict) and len(snap) > 0
# Values are (memo_index, obj) 2-tuples; the key is id(obj).
for key, value in snap.items():
    assert isinstance(value, tuple) and len(value) == 2, value
    assert key == id(value[1])
# Indices are a contiguous 0..n-1 set.
assert {idx for idx, _ in snap.values()} == set(range(len(snap)))

# clear() empties the pickler's memo.
proxy.clear()
assert p.memo.copy() == {}

# ── Pickler.memo setter ───────────────────────────────────────────────────
# From a proxy: the same objects are memoized, at the same indices.
src = _pickle.Pickler(io.BytesIO(), 5)
shared = ["a", "b"]
src.dump(shared)
src_snap = src.memo.copy()
dst = _pickle.Pickler(io.BytesIO(), 5)
dst.memo = src.memo
dst_snap = dst.memo.copy()
src_by_idx = {idx: obj for idx, obj in src_snap.values()}
dst_by_idx = {idx: obj for idx, obj in dst_snap.values()}
assert set(src_by_idx) == set(dst_by_idx)
for i in src_by_idx:
    assert dst_by_idx[i] is src_by_idx[i]

# From a {id: (index, obj)} dict.
o = ["restored"]
p2 = _pickle.Pickler(io.BytesIO(), 5)
p2.memo = {id(o): (0, o)}
vals = list(p2.memo.copy().values())
assert len(vals) == 1
(idx, obj) = vals[0]
assert idx == 0 and obj is o

# Wrong type / bad value shape / deletion are rejected.
e = err(lambda: setattr(p2, "memo", [1, 2]))
assert isinstance(e, TypeError) and "PicklerMemoProxy object or dict" in str(e), e
e = err(lambda: setattr(p2, "memo", {1: 2}))
assert isinstance(e, TypeError) and "2-item tuples" in str(e), e
e = err(lambda: delattr(p2, "memo"))
assert isinstance(e, TypeError) and "deletion is not supported" in str(e), e


# ── Unpickler.memo proxy ──────────────────────────────────────────────────
data = dump(["x", "y", ["x"]], 5)
u = _pickle.Unpickler(io.BytesIO(data))
u.load()

uproxy = u.memo
assert type(uproxy).__name__ == "UnpicklerMemoProxy", type(uproxy).__name__
assert u.memo is not u.memo

usnap = uproxy.copy()
assert isinstance(usnap, dict) and len(usnap) > 0
# index -> object; keys are integers.
assert all(isinstance(k, int) for k in usnap)

uproxy.clear()
assert u.memo.copy() == {}

# ── Unpickler.memo setter ─────────────────────────────────────────────────
# From a proxy: the index->object mapping is copied.
u1 = _pickle.Unpickler(io.BytesIO(data))
u1.load()
u1_snap = u1.memo.copy()
u2 = _pickle.Unpickler(io.BytesIO(data))
u2.memo = u1.memo
u2_snap = u2.memo.copy()
assert u2_snap.keys() == u1_snap.keys()
for k in u1_snap:
    assert u2_snap[k] is u1_snap[k]

# From a plain dict: the keys are validated, but the resulting memo is empty
# (the entries are written into the memo that is then replaced wholesale).
u3 = _pickle.Unpickler(io.BytesIO(data))
u3.memo = {0: "a", 1: "b"}
assert u3.memo.copy() == {}

# Wrong type / bad keys / deletion are rejected.
e = err(lambda: setattr(u3, "memo", [1, 2]))
assert isinstance(e, TypeError) and "UnpicklerMemoProxy object or dict" in str(e), e
e = err(lambda: setattr(u3, "memo", {"k": 1}))
assert isinstance(e, TypeError) and "memo key must be integers" in str(e), e
e = err(lambda: setattr(u3, "memo", {-1: 1}))
assert isinstance(e, ValueError) and "positive integers" in str(e), e
e = err(lambda: delattr(u3, "memo"))
assert isinstance(e, TypeError) and "deletion is not supported" in str(e), e

print("_pickle_memo_proxy OK")
