# CPython-suite gap: module tests omit non-string keys and mixed-key order
# across PyPy's ModuleDictStrategy exit.
# parity-tests reason: this directly targets the PyPy/pyre module-dict storage
# transition and its unified object-strategy operations.

"""One ModuleDictStrategy exit preserves non-str CRUD and unified ordering."""

builtins = __builtins__
mapping = builtins.__dict__ if hasattr(builtins, "__dict__") else builtins
assert isinstance(mapping, dict)

keys = ("__parity_module_a", -99887766, "__parity_module_b", None)
later = (-99887765, "__parity_module_c")
try:
    # The integer triggers the strategy switch between two string inserts.
    for key, value in zip(keys, (1, "one", 3, "none")):
        mapping[key] = value
    assert list(mapping.items())[-4:] == [
        (keys[0], 1),
        (keys[1], "one"),
        (keys[2], 3),
        (keys[3], "none"),
    ]

    mapping[keys[1]] = "ONE"
    assert mapping[keys[1]] == "ONE"
    assert mapping.pop(keys[1]) == "ONE"
    assert keys[1] not in mapping
    del mapping[keys[3]]
    assert keys[3] not in mapping

    # Later inserts share one order regardless of key type; popitem is LIFO.
    mapping[later[0]] = 4
    mapping[later[1]] = 5
    assert mapping.popitem() == (later[1], 5)
    assert mapping.popitem() == (later[0], 4)
    assert mapping.popitem() == (keys[2], 3)
    assert mapping.popitem() == (keys[0], 1)
finally:
    for key in keys + later:
        mapping.pop(key, None)

print("OK")
