# pyre-check: no-cpython

import gc
import pickle


text = b"pickle runtime string"
payloads = [
    b"\x80\x04\x8c" + bytes([len(text)]) + text + b".",
    b"\x80\x04X" + len(text).to_bytes(4, "little") + text + b".",
    b"\x80\x04\x8d" + len(text).to_bytes(8, "little") + text + b".",
]

for payload in payloads:
    result = pickle.loads(payload)
    assert result == "pickle runtime string"
    assert any(obj is result for obj in gc.get_objects())

print("pickle unicode results are collectable")
