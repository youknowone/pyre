# pyre-check: no-cpython

import gc
import unicodedata


cases = [
    ("NFC", "e\u0301 runtime", "é runtime"),
    ("NFD", "é runtime", "e\u0301 runtime"),
    ("NFKC", "\ufb03 runtime", "ffi runtime"),
    ("NFKD", "\ufb03 runtime", "ffi runtime"),
]

for form, source, expected in cases:
    result = unicodedata.normalize(form, source)
    assert result == expected
    assert any(obj is result for obj in gc.get_objects())

ascii_source = "".join(["ascii", " runtime"])
ascii_result = unicodedata.normalize("NFC", ascii_source)
assert ascii_result is ascii_source
assert any(obj is ascii_result for obj in gc.get_objects())


class Text(str):
    pass


subclass_result = unicodedata.normalize("NFC", Text("subclass runtime"))
assert type(subclass_result) is str
assert subclass_result == "subclass runtime"
assert any(obj is subclass_result for obj in gc.get_objects())

print("unicodedata normalize results are collectable")
