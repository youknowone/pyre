# cpyext-fixture: cpyext_cycles
# cpyext-expect: cpyext-cycles-collected-ok

# End-to-end check for the cyclic-collection protocol: a cycle that runs
# through a reference only C holds is collected, and nothing such a reference
# is the only one to is collected early.
#
# Both directions matter and they fail independently. Judging a block on its
# references alone collects the first script and frees live objects in the
# second; keeping everything a block references alive passes the second and
# leaks the first.
#
# Two kinds of such a reference: a C field, reported by `tp_traverse`, and the
# borrow the layer owns on a container's behalf, which lives in a side table
# no traverse can reach.

import gc
import cpyext_cycles as m

def collect():
    for _ in range(3):
        gc.collect()

class Holder:
    pass

# ── no cycle: the block dies on its own ────────────────────────────────
node = m.Node()
node.ref = Holder()
assert m.alive() == 1
del node
collect()
assert m.alive() == 0, m.alive()

# ── out through the C field and back through an attribute ──────────────
node = m.Node()
holder = Holder()
node.ref = holder
holder.back = node
assert m.alive() == 1
del node, holder
collect()
assert m.alive() == 0, m.alive()

# ── two blocks holding each other ──────────────────────────────────────
first, second = m.Node(), m.Node()
first.ref = second
second.ref = first
assert m.alive() == 2
del first, second
collect()
assert m.alive() == 0, m.alive()

# ── a block holding itself ─────────────────────────────────────────────
node = m.Node()
node.ref = node
del node
collect()
assert m.alive() == 0, m.alive()

# ── a cycle through a borrowed reference, no C type involved ───────────
# The borrow is a reference the layer owns on the container's behalf, so it
# roots the item, the item's attribute roots the container, and the container
# is what would release the borrow.
import weakref

class Watchable(list):
    pass

items = Watchable()
holder = Holder()
items.append(holder)
holder.back = items
m.peek_list(items)
watch = weakref.ref(holder)
del items, holder
collect()
assert watch() is None, 'a list cycle through a borrow was not collected'

mapping = {}
holder = Holder()
mapping['key'] = holder
holder.back = mapping
assert m.peek_dict(mapping, 'key')
watch = weakref.ref(holder)
del mapping, holder
collect()
assert watch() is None, 'a dict cycle through a borrow was not collected'

# ── a finalizer is handed what its block still references ──────────────
# `finalize_garbage` before `delete_garbage`: the block is dying and nothing
# outside it names either field, so the call can only be made while the
# collection that found it dead is still keeping both.
seen = []


class Payload:
    pass


payload = Payload()
doomed = m.Doomed(seen.append, payload)
del doomed
collect()
assert seen == [payload], seen

# ── and once, however many collections follow ──────────────────────────
collect()
assert seen == [payload], seen

print('cpyext-cycles-collected-ok')
