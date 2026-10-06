# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=<module>
# Int subclass instances with a first STORE_ATTR in a hot loop.
#
# `w_int_subclass_new` is nursery-born. The walker fold records
# `NewWithVtable` plus `_mapdict_init_empty(terminator)` (`mapdict.py`
# `user_setup`) while the concrete box still sits at map 0 until
# `tag_subclass_instance` / `ensure_mapdict_initialized`. A collection
# between those stores leaves `opimpl_getfield_gc_i` reading a live
# terminator against a heapcache word of 0
# (`_opimpl_getfield_gc_any_pureornot`).
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


class Int(int):
    pass


acc = 0
for i in range(20000):
    obj = Int(i % 500)
    obj.pair = (i % 500, i % 4)
    acc += obj.pair[0]
if acc != 4990000:
    raise AssertionError(acc)
print("PASS int subclass setattr")
