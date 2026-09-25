# cpyext-fixture: cpyext_foreign_block
# cpyext-expect: cpyext-foreign-block-ok

# A block the object allocator never handed out belongs to its extension.
#
# cffi allocates every `__CDataOwn` with plain `malloc` and frees it with
# `free`, so a block arrives holding whatever was there before and stops
# existing when its deallocator returns.  Both ends were read: `PyObject_Init`
# released the type it found in an uninitialised header, and the resurrection
# check read the count out of a block that had been freed.

import gc

import cpyext_foreign_block as m

# 0 is what `calloc` leaves behind, 0x7f what `malloc` may.
m.make(0)
m.make(0x7f)
for _ in range(3):
    gc.collect()

assert m.released_yet(), 'the deallocator has not run'
first = m.released_intact()
assert first == -1, 'byte %d of the released block was written' % first

print('cpyext-foreign-block-ok')
