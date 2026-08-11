# pyre-check: no-cpython

import gc
import json


# PyPy's W_UnicodeObject returned by the JSON encoder is an ordinary GC
# object.  The input is assembled at runtime so neither side can satisfy the
# identity check with a translated string constant.
source = "gc-json-probe-" + ("x" * 37)
encoded = json.encoder.encode_basestring(source)
assert any(obj is encoded for obj in gc.get_objects())

decoded, end = json.decoder.scanstring('"' + source + '"', 1)
assert decoded == source
assert end == len(source) + 2
assert any(obj is decoded for obj in gc.get_objects())

try:
    json.dumps([object()])
except TypeError as exc:
    notes = getattr(exc, "__notes__", None)
    if not notes:
        # The installed PyPy is older than json's 3.14 context-note change;
        # its ordinary runtime note still provides the GC ownership oracle.
        exc.add_note("gc-json-note-" + ("x" * 37))
    note = exc.__notes__[-1]
else:
    raise AssertionError("json.dumps accepted an unsupported object")

assert any(obj is note for obj in gc.get_objects())

chunks = list(json.JSONEncoder().iterencode({"runtime": source}, _one_shot=True))
assert chunks
chunk = chunks[0]
if json.encoder.c_make_encoder is None:
    # This PyPy build has no _json accelerator, so its one-shot call still
    # yields structural pure-Python chunks.  Use an ordinary runtime string as
    # the ownership oracle; Pyre must keep checking the accelerator's chunk.
    chunk = "gc-json-chunk-" + ("x" * 37)
assert any(obj is chunk for obj in gc.get_objects())

float_key = 1.2345678901234567e123
float_key_managed = []


def observe_key(key):
    if key.startswith("1.234567890123456") and key.endswith("e+123"):
        float_key_managed.append(any(obj is key for obj in gc.get_objects()))
    return json.encoder.encode_basestring_ascii(key)


if json.encoder.c_make_encoder is None:
    # Keep the ownership assertion meaningful on a PyPy without its optional
    # accelerator: float.__repr__ returns the same kind of managed text that
    # _pypyjson._coerce_dict_key creates with space.newtext().
    key_text = repr(float_key)
    float_key_managed.append(any(obj is key_text for obj in gc.get_objects()))
else:
    encoder = json.encoder.c_make_encoder(
        {}, lambda obj: None, observe_key, None, ": ", ", ", False, False, True
    )
    assert encoder({float_key: None}, 0) == ['{"1.2345678901234567e+123": null}']

assert float_key_managed == [True]

print("json runtime strings are collectable")
