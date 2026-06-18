# Pickler.memo / Unpickler.memo expose a live proxy of the memo (a fresh
# PicklerMemoProxy / UnpicklerMemoProxy on every access). pickletester already
# covers memo priming/clearing round-trips (test_priming_*_memo,
# test_clear_pickler_memo); what it does NOT pin and we keep here:
#   - a fresh proxy instance is handed back on every attribute access,
#   - the exact setter TypeError/ValueError messages and that deletion raises,
#   - assigning a plain dict to Unpickler.memo yields an EMPTY memo (a quirk:
#     entries are written into a memo that is then replaced wholesale).
# Pinned to CPython 3.14.
import io
import _pickle


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

assert type(p.memo).__name__ == "PicklerMemoProxy", type(p.memo).__name__
# A fresh proxy is handed back on each access.
assert p.memo is not p.memo

# clear() empties the pickler's memo.
p.memo.clear()
assert p.memo.copy() == {}

# Wrong type / bad value shape / deletion are rejected (the dict setter path
# validates the {id: (index, obj)} shape).
e = err(lambda: setattr(p, "memo", [1, 2]))
assert isinstance(e, TypeError) and "PicklerMemoProxy object or dict" in str(e), e
e = err(lambda: setattr(p, "memo", {1: 2}))
assert isinstance(e, TypeError) and "2-item tuples" in str(e), e
e = err(lambda: delattr(p, "memo"))
assert isinstance(e, TypeError) and "deletion is not supported" in str(e), e


# ── Unpickler.memo proxy ──────────────────────────────────────────────────
data = dump(["x", "y", ["x"]], 5)
u = _pickle.Unpickler(io.BytesIO(data))
u.load()

assert type(u.memo).__name__ == "UnpicklerMemoProxy", type(u.memo).__name__
assert u.memo is not u.memo

u.memo.clear()
assert u.memo.copy() == {}

# Assigning a plain dict validates the keys but yields an EMPTY memo: the
# entries are written into a memo that is then replaced wholesale.
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
