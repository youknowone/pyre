# CPython-suite gap: exercise pattern and original-text reclamation together.
# parity-tests reason: managed SRE patterns must not leave immortal root slots.

# pypy/module/_sre/interp_sre.py SRE_Pattern__new__ relies on the app-level
# cache and ordinary object ownership. Purging the cache must release the
# pattern once the last application reference dies. A unique literal is used
# because a named-group pattern currently stays reachable after purge — that
# is a separate leak, not the immortal address table this file removes.
import gc
import re
import weakref

class PatternText(str):
    pass

text = PatternText("unique_sre_gc_lifetime_pattern")
pattern = re.compile(text)
pid = id(pattern)
# CPython does not track `re.Pattern` in `gc.get_objects`. pyre does, and a
# live weakref currently keeps that type allocated, so the two oracles are
# mutually exclusive.
tracked = any(id(obj) == pid for obj in gc.get_objects())
refs = None if tracked else (weakref.ref(text), weakref.ref(pattern))
re.purge()
del text, pattern
gc.collect()
gc.collect()
if tracked:
    assert not any(id(obj) == pid for obj in gc.get_objects())
else:
    assert refs[0]() is None
    assert refs[1]() is None
print("OK")
