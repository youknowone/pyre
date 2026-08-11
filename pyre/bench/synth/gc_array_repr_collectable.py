# pyre-check: no-cpython

import array
import gc


integer_rendered = array.array.__repr__(array.array("i", [1, 2, 3]))
unicode_rendered = array.array.__repr__(array.array("u", "abc"))

assert integer_rendered == "array('i', [1, 2, 3])"
assert unicode_rendered == "array('u', 'abc')"
assert any(obj is integer_rendered for obj in gc.get_objects())
assert any(obj is unicode_rendered for obj in gc.get_objects())

print("array reprs are collectable")
