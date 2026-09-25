# cpyext-fixture: cpyext_cycles
# cpyext-expect: cpyext-cycles-survives-ok

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
    def __init__(self, tag):
        self.tag = tag

# ── the C field is the only reference, and the block is alive ──────────
node = m.Node()
node.ref = Holder('one')
collect()
assert getattr(node.ref, 'tag', None) == 'one', node.ref

# ── the same, reached only by the collections allocation forces ────────
churned = m.Node()
churned.ref = Holder('two')
junk = None
for index in range(200000):
    junk = [index, junk] if index % 1000 == 0 else [index]
assert getattr(churned.ref, 'tag', None) == 'two', churned.ref

# ── two blocks in a chain, only the head named ─────────────────────────
head, tail = m.Node(), m.Node()
head.ref = tail
tail.ref = Holder('three')
del tail
collect()
assert getattr(head.ref.ref, 'tag', None) == 'three', head.ref.ref
assert m.alive() == 4, m.alive()

# ── and the whole chain goes when the head does ────────────────────────
del head
collect()
assert m.alive() == 2, m.alive()

# ── a C global roots a block, and the block roots what it references ───
pinned = m.Node()
pinned.ref = Holder('four')
m.pin(pinned)
del pinned
collect()
assert type(m.pinned_ref()).__name__ == 'Node'
assert getattr(m.pinned_ref().ref, 'tag', None) == 'four', m.pinned_ref().ref

# ── a cycle a C global roots stays whole, and goes when it lets go ─────
cycle = m.Node()
holder = Holder('five')
cycle.ref = holder
holder.back = cycle
m.pin(cycle)
del cycle, holder
collect()
assert m.pinned_ref().ref.back is m.pinned_ref()
m.pin(None)
collect()
assert m.alive() == 2, m.alive()

del node, churned
collect()
assert m.alive() == 0, m.alive()

# ── a borrowed reference stays usable while its container lives ─────────
class Watchable(list):
    pass

items = Watchable([Holder('borrowed')])
m.hold_item(items)
collect()
assert m.held_item().tag == 'borrowed', m.held_item()
assert m.held_item() is items[0]

# The layer holds the borrow until the container's mirror dies, so an item the
# container no longer names is still readable.
del items[0]
collect()
assert getattr(m.held_item(), 'tag', None) == 'borrowed', m.held_item()

# A borrow on a container reached only through another container.
inner = Watchable([Holder('inner')])
outer = Watchable([inner])
m.hold_item(outer)
del inner
collect()
assert m.held_item()[0].tag == 'inner', m.held_item()

print('cpyext-cycles-survives-ok')
