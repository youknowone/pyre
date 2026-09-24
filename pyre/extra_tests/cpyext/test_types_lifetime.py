# cpyext-fixture: cpyext_types
# cpyext-expect: cpyext-lifetime-ok

# End-to-end check for C-defined types: `PyType_Ready`, the slot wrappers,
# `tp_methods`/`tp_members`/`tp_getset` descriptors, inheritance through
# `tp_base` and `PyErr_NewExceptionWithDoc`.

import gc
import weakref
import cpyext_types as m

# ── an instance nothing holds is deallocated ───────────────────────────
before = m.owner_deallocs()
for _ in range(16):
    m.Owner()
gc.collect()
after = m.owner_deallocs()
assert after > before, f'no Owner was deallocated: {before} -> {after}'

# ── a C reference roots its object ─────────────────────────────────────
class Node:
    pass

node = Node()
ref = weakref.ref(node)
owner = m.Owner(node)
del node
gc.collect()
gc.collect()
assert ref() is not None, 'a C reference did not keep its object alive'

# ── and releasing it lets the object go ────────────────────────────────
before = m.owner_deallocs()
del owner
gc.collect()
after = m.owner_deallocs()
assert after > before, f'the last Owner was not deallocated: {before} -> {after}'
gc.collect()
assert ref() is None, 'releasing the C reference did not let the object go'

# ── and gc.collect() runs the deallocators before it returns ───────────
# Read from C, so no bytecode runs between the collection and the count: a
# drain left to the async action would not have happened yet.  Automatic
# collection is off across the setup and the count is read again once it is
# built, so a collection nobody asked for cannot be what moves the number.
was_enabled = gc.isenabled()
gc.disable()
try:
    before = m.owner_deallocs()
    for _ in range(16):
        m.Owner()
    settled = m.owner_deallocs()
    assert settled == before, f'an Owner was deallocated before the explicit collect: {before} -> {settled}'
    during = m.collect_then_owner_deallocs()
finally:
    if was_enabled:
        gc.enable()
assert during > settled, f'gc.collect() returned before tp_dealloc ran: {settled} -> {during}'

# ── a member declaring Py_AUDIT_READ reports every read ────────────────
# Last, because a hook can never be removed once it is installed.
import sys

seen = []
sys.addaudithook(
    lambda event, args: seen.append(args[1]) if event == 'object.__getattr__' else None
)
audited = m.Extra(1)
audited.set(5, 2.5)
assert audited.tag == 5, audited.tag
assert seen == [], seen
assert audited.weight == 2.5, audited.weight
assert seen == ['weight'], seen

print('cpyext-lifetime-ok')
