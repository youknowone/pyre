//! A build-time jitcode table and the one descr pool its bodies index.
//!
//! `CodeWriter.make_jitcodes()` (codewriter.py:89) produces two lists that
//! only mean anything together: `all_jitcodes`, and the single shared
//! `Assembler.descrs` (assembler.py:23) that every `d`/`j` argcode in every
//! body indexes. A host that runs the codewriter at build time and embeds the
//! two serialized lists in its binary has to join them back into the runtime
//! shapes — `Arc<JitCode>` shells and a `RuntimeBhDescr` pool. This type is
//! that join, and it is the only place the two lists meet.
//!
//! **What the join has to preserve.** A `BhDescr::JitCode` slot names its
//! callee by an `all_jitcodes` index, and the object that index resolves to
//! must be *the* object the table holds at it — `codewriter.py:80
//! all_jitcodes[jitcode.index] is jitcode`, an identity, not an equality.
//! Minting a second shell per pool slot satisfies every read that only wants a
//! body and breaks every read that asks whether two references name the same
//! jitcode: an `Arc::ptr_eq` dedup while flattening a registry sees one callee
//! twice, and an index stamped on one shell is read back off the other.
//!
//! **Why there is no cycle to break.** The shells carry an EMPTY per-jitcode
//! `exec.descrs` and resolve every operand through this pool as the
//! process-global fallback ([`JitCode::descr_at`],
//! [`init_global_build_descr_pool`]). So the pool holds jitcodes and the
//! jitcodes hold nothing — a `BC_INLINE_CALL` chain of any depth resolves
//! against one table. Copying the pool into each shell instead makes the depth
//! the copy was taken at the depth that resolves: the callees inside a copied
//! pool are shells of their own, and whatever pool they were given is the one
//! their own operands read.

use std::sync::Arc;

use super::{CanonicalBhDescr, CanonicalJitCode, JitCode, RuntimeBhDescr, RuntimeDescrTable};

/// A materialized build-time `all_jitcodes` + `Assembler.descrs` pair.
///
/// Process-lifetime by construction: [`Self::materialize`] leaks both lists,
/// because the pool is installed as the global `descr_at` fallback, whose
/// entries are handed out as `&'static`.
pub struct EmbeddedJitCodeTable {
    jitcodes: &'static [Arc<JitCode>],
    descrs: &'static [RuntimeBhDescr],
}

// SAFETY: `materialize` builds only the `Descr` and `JitCode` variants. The
// two variants that carry raw pointers — `Call`'s `JitCallTarget` and
// `AssemblerToken` — are never constructed here, so nothing in either list is
// a pointer this type could hand across a thread. A future arm that mints one
// invalidates this, which is why the constructor is the only writer.
unsafe impl Send for EmbeddedJitCodeTable {}
unsafe impl Sync for EmbeddedJitCodeTable {}

impl EmbeddedJitCodeTable {
    /// Join the two serialized lists into runtime shells and their pool.
    ///
    /// `canonical` must be `all_jitcodes` in allocation order, which
    /// `codewriter.py:68 jitcode.index = index` makes the same thing as
    /// "indexed by `jitcode.index`" — asserted here, since every `j` operand
    /// in the pool is an index into it and a list that drifted from its own
    /// indices resolves silently to the wrong callee.
    pub fn materialize(
        canonical: &[Arc<CanonicalJitCode>],
        descrs: Vec<CanonicalBhDescr>,
    ) -> &'static Self {
        let jitcodes: &'static [Arc<JitCode>] = Box::leak(
            canonical
                .iter()
                .enumerate()
                .map(|(index, core)| {
                    assert_eq!(
                        core.try_index(),
                        Some(index),
                        "jitcode {:?} sits at position {index} but names index {:?}; \
                         every `j` operand indexes this list",
                        core.name,
                        core.try_index(),
                    );
                    Arc::new(JitCode::from_canonical((**core).clone()))
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let pool: &'static [RuntimeBhDescr] = Box::leak(
            descrs
                .into_iter()
                .map(|descr| match descr {
                    // The callee is the table's own entry, cloned — an
                    // `Arc::clone`, so `ptr_eq` against the table holds.
                    CanonicalBhDescr::JitCode { jitcode_index, .. } => {
                        RuntimeBhDescr::JitCode(Arc::clone(&jitcodes[jitcode_index]))
                    }
                    // Every other variant is an ordinary `d` descr and carries
                    // through unchanged: the runtime pool element and the
                    // build-time one are the same type.
                    other => RuntimeBhDescr::Descr(Box::new(other)),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        Box::leak(Box::new(Self {
            jitcodes,
            descrs: pool,
        }))
    }

    /// `metainterp_sd.jitcodes` (warmspot.py:281-282) — the flat registry
    /// `resume.py:1338-1340` indexes by a frame's `jitcode_pos`.
    pub fn jitcodes(&self) -> &'static [Arc<JitCode>] {
        self.jitcodes
    }

    /// The shared pool, in `Assembler.descrs` order.
    pub fn descrs(&self) -> &'static [RuntimeBhDescr] {
        self.descrs
    }

    /// The jitcode an `all_jitcodes` index names.
    pub fn by_index(&self, index: usize) -> Option<&'static Arc<JitCode>> {
        self.jitcodes.get(index)
    }

    /// The jitcode a graph leaf name names, or `None`.
    ///
    /// A name is not unique — the codewriter derives it from the graph leaf,
    /// and two distinct graphs can end on the same one — so this returns the
    /// first match and is only a lookup for callers that hold a name and
    /// nothing better. Anything that can carry an index should use
    /// [`Self::by_index`], which is what the operands themselves do.
    pub fn by_name(&self, name: &str) -> Option<&'static Arc<JitCode>> {
        self.jitcodes.iter().find(|jitcode| jitcode.name() == name)
    }

    /// Install this pool as the process-global `descr_at` fallback.
    ///
    /// Idempotent through [`init_global_build_descr_pool`]: the first table
    /// wins. Until this runs, a shell's operands resolve against its own empty
    /// `exec.descrs` and every lookup returns `None`.
    pub fn install_as_global_pool(&'static self) {
        super::init_global_build_descr_pool(self);
    }
}

impl RuntimeDescrTable for EmbeddedJitCodeTable {
    fn get(&self, index: usize) -> Option<&'static RuntimeBhDescr> {
        self.descrs.get(index)
    }

    fn len(&self) -> usize {
        self.descrs.len()
    }

    fn jitcodes(&self) -> &'static [Arc<JitCode>] {
        self.jitcodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two jitcodes, the second reachable from the first through a `j` slot.
    fn fixture() -> (Vec<Arc<CanonicalJitCode>>, Vec<CanonicalBhDescr>) {
        let caller = Arc::new(CanonicalJitCode::new("caller"));
        caller.set_index(0);
        caller.set_body(Default::default());
        let callee = Arc::new(CanonicalJitCode::new("callee"));
        callee.set_index(1);
        callee.set_body(Default::default());
        (
            vec![caller, callee],
            vec![CanonicalBhDescr::JitCode {
                jitcode_index: 1,
                fnaddr: 0,
                calldescr: Default::default(),
            }],
        )
    }

    /// The subject: a `j` slot resolves to the table's own entry, not to a
    /// second shell that merely carries the same body.
    #[test]
    fn a_jitcode_slot_resolves_to_the_table_entry_itself() {
        let (canonical, descrs) = fixture();
        let table = EmbeddedJitCodeTable::materialize(&canonical, descrs);
        let from_slot = table.descrs()[0]
            .as_jitcode()
            .expect("the `j` slot must resolve to a jitcode");
        let from_table = table.by_index(1).expect("index 1 is in the table");
        assert!(
            Arc::ptr_eq(from_slot, from_table),
            "`all_jitcodes[jitcode.index] is jitcode` (codewriter.py:80) is an \
             identity: a second shell with the same body passes every read of \
             the body and fails every `Arc::ptr_eq`",
        );
    }

    /// The shells hold no pool of their own, so nothing is stored twice and
    /// depth cannot decide what resolves.
    #[test]
    fn a_shell_carries_no_pool_of_its_own() {
        let (canonical, descrs) = fixture();
        let table = EmbeddedJitCodeTable::materialize(&canonical, descrs);
        for jitcode in table.jitcodes() {
            assert!(
                jitcode.exec.descrs.is_empty(),
                "a build-time shell resolves through the global pool; a \
                 per-shell copy makes its callees' operands read whichever \
                 pool those callees were handed",
            );
        }
    }

    /// A list whose positions disagree with its own `jitcode.index` stamps
    /// cannot be indexed by a `j` operand, and says so at materialization
    /// rather than resolving to the wrong callee.
    #[test]
    #[should_panic(expected = "names index")]
    fn a_table_out_of_order_with_its_indices_is_refused() {
        let (mut canonical, descrs) = fixture();
        canonical.swap(0, 1);
        let _ = EmbeddedJitCodeTable::materialize(&canonical, descrs);
    }
}
