# pyre-check: no-cpython
import gc
import os
import sys


# inspector.py:HeapDumper writes signed native words, with the root/non-root
# marker between the two graph phases. The translation-time type tables use
# one text line per member and expose the header-id mapping separately.
typeids_z = gc.get_typeids_z()
typeids_list = gc.get_typeids_list()
assert isinstance(typeids_z, bytes)
assert len(typeids_z) > 2
assert isinstance(typeids_list, list) and len(typeids_list) > 1
assert isinstance(typeids_list[0], int)
assert isinstance(typeids_list[-1], int)

dump_path = "/tmp/pyre-gc-dump-%d.bin" % os.getpid()
dump_file = open(dump_path, "wb")
assert gc.dump_rpy_heap(dump_file) is None
dump_file.close()
with open(dump_path, "rb") as dump_file:
    dump_data = dump_file.read()
word_size = 8 if sys.maxsize > 2**32 else 4
marker = b"\x00" * (word_size * 3) + (-1).to_bytes(
    word_size, sys.byteorder, signed=True
)
assert len(dump_data) % word_size == 0
assert marker in dump_data
os.unlink(dump_path)

print("OK")
