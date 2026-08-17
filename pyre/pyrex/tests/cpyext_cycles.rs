//! End-to-end check for the cyclic-collection protocol: a cycle that runs
//! through a C field is collected, and nothing a C field is the only reference
//! to is collected early.
//!
//! The two halves fail independently. Judging a block on its references alone
//! collects the first script and frees live objects in the second; keeping
//! every referenced block alive passes the second and leaks the first.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const COLLECTED_SCRIPT: &str = r#"
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

print('cpyext-cycles-collected-ok')
"#;

const SURVIVES_SCRIPT: &str = r#"
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

print('cpyext-cycles-survives-ok')
"#;

#[test]
fn a_cycle_through_a_c_field_is_collected() {
    let fixtures = Fixtures::new("cpyext-cycles-collected");
    fixtures.compile("cpyext_cycles");
    fixtures.expect_ok(COLLECTED_SCRIPT, &[], "cpyext-cycles-collected-ok");
}

#[test]
fn what_only_a_c_field_references_outlives_a_collection() {
    let fixtures = Fixtures::new("cpyext-cycles-survives");
    fixtures.compile("cpyext_cycles");
    fixtures.expect_ok(SURVIVES_SCRIPT, &[], "cpyext-cycles-survives-ok");
}
