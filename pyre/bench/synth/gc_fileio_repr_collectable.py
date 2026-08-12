# pyre-check: no-cpython
# pyre-check: skip-backends=wasm
import _io
import gc


file = _io.FileIO(__file__, "r")
name_direct = _io.FileIO.__repr__(file)
name_ordinary = repr(file)

assert name_direct == name_ordinary
assert " name=" in name_direct
assert any(obj is name_direct for obj in gc.get_objects())
assert any(obj is name_ordinary for obj in gc.get_objects())

del file.name
fd_direct = _io.FileIO.__repr__(file)
fd_ordinary = repr(file)

assert fd_direct == fd_ordinary
assert " fd=" in fd_direct
assert any(obj is fd_direct for obj in gc.get_objects())
assert any(obj is fd_ordinary for obj in gc.get_objects())

file.close()
closed_direct = _io.FileIO.__repr__(file)
closed_ordinary = repr(file)

assert closed_direct == closed_ordinary
assert closed_direct == "<_io.FileIO [closed]>"
assert any(obj is closed_direct for obj in gc.get_objects())
assert any(obj is closed_ordinary for obj in gc.get_objects())

print("FileIO repr results are collectable")
