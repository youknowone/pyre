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

use std::collections::HashSet;

use majit_charon_reader::Llbc;

use crate::front::mir::{self, CrateLowering, CrateLoweringState, DeclBuildError, LowerError};
use crate::front::semantic::{SemanticFunction, SemanticProgram};

/// Where a funcobj's body comes from: the LLBC that carries it and the
/// Charon `def_id` that indexes it there (`Llbc::fn_by_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "no funcobj records its body source yet")
)]
pub(crate) struct GraphBodySource {
    pub llbc_index: u32,
    pub def_id: u64,
}

/// Owns the extracted LLBC set and everything else the lowering reads, so
/// a funcobj's body can be built after the whole-program pass has run.
///
/// The three `HostStaticAddrs` tables and the error-carrier spec are held
/// owned because [`crate::HostStaticAddrs`] borrows all of them from the
/// caller's frame; the borrowed view is rebuilt per lowering call.
pub(crate) struct GraphBodyProvider {
    crates: Vec<ProvidedCrate>,
    jitdriver_receiver_roots: Vec<String>,
    pytypes: Vec<(String, i64)>,
    pytypes_by_struct: Vec<(String, i64)>,
    refs: Vec<(String, i64)>,
    int_values: Vec<(String, i64)>,
    error_carrier: OwnedErrorCarrierSpec,
    scalar_field_stores: Vec<OwnedScalarFieldStore>,
}

/// One lowered crate: its artefact and the lowering state its decls were
/// lowered with.
struct ProvidedCrate {
    llbc: Llbc,
    state: CrateLoweringState,
}

/// Owned mirror of [`crate::ErrorCarrierSpec`], held for the same reason as
/// the three tables beside it: the spec borrows from the caller's frame, and
/// a demanded body has to lower with the one the whole-program pass used.
#[derive(Debug, Default)]
struct OwnedErrorCarrierSpec {
    carrier_path: String,
    carrier_wrappers: Vec<String>,
    to_exc_object: Option<Vec<String>>,
    from_exc_object: Option<(String, String)>,
}

/// Owned mirror of [`crate::ScalarFieldStore`], held for the reason the
/// carrier spec above it is: a demanded body has to lower the declared
/// boundaries the whole-program pass lowered, and the declaration borrows
/// from the caller's frame.
#[derive(Debug)]
struct OwnedScalarFieldStore {
    function_path: String,
    owner_root: String,
    field: String,
    bank: crate::ScalarBank,
}

impl OwnedScalarFieldStore {
    fn own(store: &crate::ScalarFieldStore<'_>) -> Self {
        Self {
            function_path: store.function_path.to_string(),
            owner_root: store.owner_root.to_string(),
            field: store.field.to_string(),
            bank: store.bank,
        }
    }

    fn borrowed(&self) -> crate::ScalarFieldStore<'_> {
        crate::ScalarFieldStore {
            function_path: &self.function_path,
            owner_root: &self.owner_root,
            field: &self.field,
            bank: self.bank,
        }
    }
}

impl OwnedErrorCarrierSpec {
    fn own(spec: crate::ErrorCarrierSpec<'_>) -> Self {
        let own_path = |segments: &[&str]| -> Vec<String> {
            segments.iter().map(|s| (*s).to_string()).collect()
        };
        Self {
            carrier_path: spec.carrier_path.to_string(),
            carrier_wrappers: own_path(spec.carrier_wrappers),
            to_exc_object: spec.to_exc_object.map(own_path),
            from_exc_object: spec
                .from_exc_object
                .map(|(receiver, method)| (receiver.to_string(), method.to_string())),
        }
    }
}

impl GraphBodyProvider {
    pub(crate) fn new(
        static_addrs: crate::HostStaticAddrs<'_>,
        jitdriver_receiver_roots: &[String],
    ) -> Self {
        let own = |rows: &[(&str, i64)]| -> Vec<(String, i64)> {
            rows.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
        };
        Self {
            crates: Vec::new(),
            jitdriver_receiver_roots: jitdriver_receiver_roots.to_vec(),
            pytypes: own(static_addrs.pytypes),
            pytypes_by_struct: own(static_addrs.pytypes_by_struct),
            refs: own(static_addrs.refs),
            int_values: own(static_addrs.int_values),
            error_carrier: OwnedErrorCarrierSpec::own(static_addrs.error_carrier),
            scalar_field_stores: static_addrs
                .scalar_field_stores
                .iter()
                .map(OwnedScalarFieldStore::own)
                .collect(),
        }
    }

    /// Lower one already-linked artefact and keep it. The caller applied
    /// `discover_transparent_scalar_kinds` and `discover_foldable_const_lits`
    /// across the whole set first, and `cross_tombstoned_leaves` is the
    /// duplicate-leaf verdict across that set.
    pub(crate) fn lower_prelinked_crate(
        &mut self,
        llbc: Llbc,
        module_paths: &[&str],
        cross_tombstoned_leaves: &HashSet<String>,
    ) -> SemanticProgram {
        let module_filter = mir::normalize_module_filter(module_paths);
        let paint_tombstones = mir::prelink_crate(&llbc, cross_tombstoned_leaves);
        let mut state = CrateLoweringState::new(&llbc, &paint_tombstones);
        let functions = self.with_static_addrs(|static_addrs| {
            CrateLowering::new(&llbc, static_addrs, &self.jitdriver_receiver_roots, &state)
                .lower_all(module_filter.as_ref(), None)
        });
        let mut program = state.finish(functions);
        mir::harden_duplicate_leaf_metadata(
            &mut program.struct_fields,
            &mut program.struct_origins,
            &mut program.enum_variant_by_discriminant,
            Some(&program.struct_ids),
        );
        self.crates.push(ProvidedCrate { llbc, state });
        program
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
    #[cfg(test)]
    pub(crate) fn source_for_name_path(&self, name_path: &str) -> Option<GraphBodySource> {
        let mut found = None;
        for (i, krate) in self.crates.iter().enumerate() {
            for fd in krate.llbc.iter_local_fns() {
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

    /// Build the funcobj `src` names with the state its crate was lowered
    /// with, reproducing what the whole-program loop produced for it. Which
    /// funcobjs exist is decided when they are registered, so no membership
    /// gate runs here, and a body that does not lower answers its own error.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "no funcobj records its body source yet")
    )]
    pub(crate) fn build(&self, src: GraphBodySource) -> Result<SemanticFunction, LowerError> {
        let idx = src.llbc_index as usize;
        let krate = self
            .crates
            .get(idx)
            .ok_or_else(|| LowerError::Unsupported(format!("llbc index {idx} out of range")))?;
        let fd = krate.llbc.fn_by_id(src.def_id).ok_or_else(|| {
            LowerError::Unsupported(format!("no FunDecl for def_id {}", src.def_id))
        })?;
        self.with_static_addrs(|static_addrs| {
            CrateLowering::new(
                &krate.llbc,
                static_addrs,
                &self.jitdriver_receiver_roots,
                &krate.state,
            )
            .build_decl(fd)
        })
        .map_err(|e| match e {
            DeclBuildError::NoBody => LowerError::Unsupported(format!(
                "{}: no Unstructured body",
                fd.item_meta.name_path()
            )),
            DeclBuildError::Lower { error, .. } => error,
        })
    }

    /// Run `f` with the borrowed [`crate::HostStaticAddrs`] view of the
    /// owned tables.
    fn with_static_addrs<R>(&self, f: impl FnOnce(crate::HostStaticAddrs<'_>) -> R) -> R {
        let pytypes = borrowed(&self.pytypes);
        let pytypes_by_struct = borrowed(&self.pytypes_by_struct);
        let refs = borrowed(&self.refs);
        let int_values = borrowed(&self.int_values);
        let carrier = &self.error_carrier;
        let carrier_wrappers = borrowed_segments(&carrier.carrier_wrappers);
        let scalar_field_stores: Vec<crate::ScalarFieldStore<'_>> = self
            .scalar_field_stores
            .iter()
            .map(OwnedScalarFieldStore::borrowed)
            .collect();
        let to_exc_object = carrier.to_exc_object.as_deref().map(borrowed_segments);
        f(crate::HostStaticAddrs {
            pytypes: &pytypes,
            pytypes_by_struct: &pytypes_by_struct,
            refs: &refs,
            int_values: &int_values,
            error_carrier: crate::ErrorCarrierSpec {
                carrier_path: &carrier.carrier_path,
                carrier_wrappers: &carrier_wrappers,
                to_exc_object: to_exc_object.as_deref(),
                from_exc_object: carrier
                    .from_exc_object
                    .as_ref()
                    .map(|(receiver, method)| (receiver.as_str(), method.as_str())),
            },
            scalar_field_stores: &scalar_field_stores,
        })
    }
}

fn borrowed(rows: &[(String, i64)]) -> Vec<(&str, i64)> {
    rows.iter().map(|(k, v)| (k.as_str(), *v)).collect()
}

fn borrowed_segments(segments: &[String]) -> Vec<&str> {
    segments.iter().map(String::as_str).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::model::FunctionGraph;

    const CORPUS: &str = crate::runtime_names::artifacts::CHARON_CORPUS_ULLBC;

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
    /// Also asserts every lowered funcobj's name path is unique in the
    /// corpus: `source_for_name_path` resolves an ambiguous name to
    /// `None`, so a duplicate surfaces here as a lookup miss.
    #[test]
    fn provider_reproduces_the_eagerly_lowered_body() {
        let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
        let mut provider = GraphBodyProvider::new(crate::HostStaticAddrs::default(), &[]);
        let program = provider.lower_prelinked_crate(llbc, &[], &HashSet::new());
        let mut compared = 0;
        for f in &program.functions {
            let Some(fd) = f
                .fun_decl_id
                .and_then(|id| provider.crates[0].llbc.fn_by_id(id))
            else {
                continue;
            };
            let name_path = fd.item_meta.name_path();
            let src = provider
                .source_for_name_path(&name_path)
                .unwrap_or_else(|| panic!("no unique GraphBodySource for {name_path}"));
            let got = provider
                .build(src)
                .unwrap_or_else(|e| panic!("provider failed to build {name_path}: {e}"));
            // A clause specialization shares its generic's `FunDecl` under
            // its own name.
            if got.name != f.name {
                continue;
            }
            assert_eq!(shape(got.graph()), shape(f.graph()), "{name_path}");
            compared += 1;
        }
        assert!(compared > 0, "corpus fixture lowered no bodies at all");
    }
}
