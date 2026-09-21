# pyre-check: gate=1
"""A builtin function keeps its `__module__` across a major collection.

`struct.unpack` is an interp-level carrier the collector cannot trace into,
and `_struct` is a module the collector owns, so none of the tables the root
walk visits names the carrier.  The `__module__` string stamped into it at
module-install time is then reachable only through the carrier's own census;
without that census a major collection sweeps the string while the carrier
keeps answering with its address.

A swept cell keeps its bytes until something else claims it, so reading the
attribute is not enough on its own -- the short strings below are what puts
the freed cell back in use, and the fingerprint is what notices.  Naming the
string itself would root it and answer nothing, so only the two integers
survive the setup.
"""
import gc
import struct

MB = 1024 * 1024


def fingerprint(text):
    """Describe a string without keeping a reference to it."""
    return len(text), sum(map(ord, text))


def a_builtin_keeps_its_module_across_a_major_collection():
    watched = [
        (func, fingerprint(func.__module__))
        for func in (struct.pack, struct.unpack, struct.calcsize)
    ]
    for _ in range(3):
        junk = [bytes(MB) for _ in range(48)]
        del junk
        gc.collect()
        gc.collect()
        reuse = ['x' * (i % 40) for i in range(40000)]
        gc.collect()
        del reuse
        for func, before in watched:
            assert fingerprint(func.__module__) == before, before


a_builtin_keeps_its_module_across_a_major_collection()
print('OK')
