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

print("json codec strings are collectable")
