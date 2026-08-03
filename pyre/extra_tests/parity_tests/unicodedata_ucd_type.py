"""`unicodedata.UCD` is an interpreter-owned type, not a constructible one.

`interp_ucd.py:311 UCD.typedef` declares no `__new__`, so the databases the
module exports are the only instances; the type rejects both direct
construction and use as a base.
"""

import unicodedata

ucd_type = type(unicodedata.ucd_3_2_0)
assert ucd_type.__name__ == "UCD", ucd_type.__name__

try:
    ucd_type()
except TypeError as error:
    assert "UCD" in str(error), error
else:
    raise AssertionError("UCD must not be constructible")

try:

    class Sub(ucd_type):
        pass

except TypeError as error:
    assert str(error) == "type 'unicodedata.UCD' is not an acceptable base type", error
else:
    raise AssertionError("UCD must not be an acceptable base type")

# The exported instances keep working, and each carries its own database.
assert unicodedata.ucd_3_2_0.unidata_version == "3.2.0"
assert unicodedata.ucd_3_2_0.category("A") == "Lu"
assert unicodedata.name("A") == "LATIN CAPITAL LETTER A"

print("OK")
