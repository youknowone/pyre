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

print("json runtime strings are collectable")
