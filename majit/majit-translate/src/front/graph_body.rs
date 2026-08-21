//! On-demand construction of a funcobj's lowered body.
//!
//! `translator.py buildflowgraph(func)` builds a function's flow graph
//! from the function object when a consumer first asks for it, and
//! `description.py FunctionDesc.cachedgraph` is that consumer: a
//! cache miss builds, a hit returns.  Nothing in upstream builds every
//! reachable function's graph up front — `translator.graphs` *is* the set
//! that demand has already reached.
//!
//! Pyre's funcobjs are Charon-extracted `FunDecl`s rather than Python
//! function objects, so the analogue of "build the flow graph from the
//! function object" is "lower the `FunDecl` from the LLBC it came from".
//! That requires the LLBC set to stay alive past the whole-program build,
//! which is what a [`GraphBodyProvider`] owns.

use std::collections::HashMap;
use std::sync::OnceLock;

use majit_charon_reader::Llbc;

use crate::front::mir::{self, LowerError};
use crate::model::{FunctionGraph, ValueType};

/// Where a funcobj's body comes from: the LLBC that carries it and the
/// Charon `def_id` that indexes it there (`Llbc::fn_by_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GraphBodySource {
    pub llbc_index: u32,
    pub def_id: u64,
}

/// Owns the extracted LLBC set and everything else the lowering reads, so
/// a funcobj's body can be built after the whole-program pass has run.
///
/// The three `HostStaticAddrs` tables are held owned because
/// [`crate::HostStaticAddrs`] borrows its slices from the caller's frame;
/// the borrowed view is rebuilt per [`GraphBodyProvider::build`] call,
/// which only a demanded body pays for.
pub(crate) struct GraphBodyProvider {
    llbcs: Vec<Llbc>,
    /// Per-LLBC struct field-attribute map, the same one the whole-program
    /// loop lowered that LLBC's decls with.  `derive_program_metadata` is a
    /// pure function of the LLBC, so recovering it here reproduces the
    /// map exactly; it is computed on the first body demanded from each
    /// LLBC rather than for every LLBC up front.
    struct_field_attrs: Vec<OnceLock<HashMap<String, Vec<(String, ValueType)>>>>,
    pytypes: Vec<(String, i64)>,
    refs: Vec<(String, i64)>,
    int_values: Vec<(String, i64)>,
}

impl GraphBodyProvider {
    pub(crate) fn new(llbcs: Vec<Llbc>, static_addrs: crate::HostStaticAddrs<'_>) -> Self {
        let own = |rows: &[(&str, i64)]| -> Vec<(String, i64)> {
            rows.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
        };
        let struct_field_attrs = llbcs.iter().map(|_| OnceLock::new()).collect();
        Self {
            llbcs,
            struct_field_attrs,
            pytypes: own(static_addrs.pytypes),
            refs: own(static_addrs.refs),
            int_values: own(static_addrs.int_values),
        }
    }

    /// Locate the funcobj whose Charon `name_path()` is `name_path`, if
    /// exactly one carries it.
    ///
    /// `bookkeeper.py getdesc(pyobj)` keys a descriptor by the function
    /// object itself, so two distinct functions are never conflated even
    /// when they render the same name.  A name path carries no such
    /// identity, and nothing stops two extracted `FunDecl`s from sharing
    /// one, so an ambiguous name resolves to nothing rather than to an
    /// arbitrary one of the candidates: binding the wrong body is silent,
    /// while a miss is not.  Identity on the demand path is
    /// [`GraphBodySource`], recorded when the funcobj was registered.
    ///
    /// Linear over the corpus, so it is a registration-time helper (and
    /// the test seam), not a per-demand lookup.
    pub(crate) fn source_for_name_path(&self, name_path: &str) -> Option<GraphBodySource> {
        let mut found = None;
        for (i, llbc) in self.llbcs.iter().enumerate() {
            for fd in llbc.iter_local_fns() {
                if fd.item_meta.name_path() != name_path {
                    continue;
                }
                if found.is_some() {
                    return None;
                }
                found = Some(GraphBodySource {
                    llbc_index: i as u32,
                    def_id: fd.def_id,
                });
            }
        }
        found
    }

    /// Lower the funcobj `src` names, reproducing what the whole-program
    /// loop produced for it.
    pub(crate) fn build(&self, src: GraphBodySource) -> Result<FunctionGraph, LowerError> {
        let idx = src.llbc_index as usize;
        let llbc = self
            .llbcs
            .get(idx)
            .ok_or_else(|| LowerError::Unsupported(format!("llbc index {idx} out of range")))?;
        let fd = llbc.fn_by_id(src.def_id).ok_or_else(|| {
            LowerError::Unsupported(format!("no FunDecl for def_id {}", src.def_id))
        })?;
        let attrs = self.struct_field_attrs[idx].get_or_init(|| mir::struct_field_attrs_of(llbc));
        let pytypes = borrowed(&self.pytypes);
        let refs = borrowed(&self.refs);
        let int_values = borrowed(&self.int_values);
        mir::lower_fun_decl_with_static_addrs_and_attrs(
            llbc,
            fd,
            crate::HostStaticAddrs {
                pytypes: &pytypes,
                refs: &refs,
                int_values: &int_values,
            },
            attrs,
        )
    }
}

fn borrowed(rows: &[(String, i64)]) -> Vec<(&str, i64)> {
    rows.iter().map(|(k, v)| (k.as_str(), *v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../charon-corpus/corpus.ullbc");

    /// Rewrite every `id: N` in `s` to the position at which that id was
    /// first seen across the graph, so two lowerings that differ only in
    /// which range of the process-global `NEXT_VAR_ID` they drew from
    /// render the same.  Aliasing is preserved: one id maps to one slot,
    /// so "these two operands are the same variable" still shows up.
    fn normalize_ids(s: &str, seen: &mut HashMap<u64, usize>) -> String {
        const KEY: &str = "id: ";
        let mut out = String::with_capacity(s.len());
        let mut rest = s;
        while let Some(at) = rest.find(KEY) {
            let (before, after) = rest.split_at(at + KEY.len());
            out.push_str(before);
            let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
            match digits.parse::<u64>() {
                Ok(id) => {
                    let next = seen.len();
                    let slot = *seen.entry(id).or_insert(next);
                    out.push_str(&format!("#{slot}"));
                    rest = &after[digits.len()..];
                }
                Err(_) => rest = after,
            }
        }
        out.push_str(rest);
        out
    }

    /// An id-independent shape of a lowered graph.  Variables carry ids
    /// from process-global counters, so two lowerings of one funcobj are
    /// never `==`; what must match is the structure the codewriter reads.
    fn shape(
        g: &FunctionGraph,
    ) -> (
        String,
        Option<String>,
        usize,
        Vec<(usize, usize)>,
        Vec<String>,
    ) {
        let mut seen: HashMap<u64, usize> = HashMap::new();
        let mut ops = Vec::new();
        let mut blocks = Vec::new();
        for b in &g.blocks {
            blocks.push((b.inputargs.len(), b.operations.len()));
            for arg in &b.inputargs {
                ops.push(normalize_ids(&format!("inputarg {arg:?}"), &mut seen));
            }
            for op in &b.operations {
                ops.push(normalize_ids(&format!("{:?}", op.kind), &mut seen));
            }
        }
        (
            g.name.clone(),
            g.return_type.clone(),
            g.blocks.len(),
            blocks,
            ops,
        )
    }

    /// A body built through the provider is the body the whole-program
    /// loop built: same graph shape, from the same `FunDecl`, for every
    /// funcobj in the corpus that lowers at all.
    ///
    /// Also asserts every lowerable funcobj's name path is unique in the
    /// corpus: `source_for_name_path` resolves an ambiguous name to
    /// `None`, so a duplicate surfaces here as a lookup miss.
    #[test]
    fn provider_reproduces_the_eagerly_lowered_body() {
        let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
        let attrs = mir::struct_field_attrs_of(&llbc);
        let mut eager: Vec<(String, _)> = Vec::new();
        for fd in llbc.iter_local_fns() {
            if fd.unstructured().is_none() || fd.is_global_initializer.is_some() {
                continue;
            }
            if let Ok(g) = mir::lower_fun_decl_with_static_addrs_and_attrs(
                &llbc,
                fd,
                crate::HostStaticAddrs::default(),
                &attrs,
            ) {
                eager.push((fd.item_meta.name_path(), shape(&g)));
            }
        }
        assert!(!eager.is_empty(), "corpus fixture lowered no bodies at all");

        let provider = GraphBodyProvider::new(vec![llbc], crate::HostStaticAddrs::default());
        for (name_path, want) in &eager {
            let src = provider
                .source_for_name_path(name_path)
                .unwrap_or_else(|| panic!("no unique GraphBodySource for {name_path}"));
            let got = provider
                .build(src)
                .unwrap_or_else(|e| panic!("provider failed to build {name_path}: {e}"));
            assert_eq!(&shape(&got), want, "{name_path}");
        }
    }
}
