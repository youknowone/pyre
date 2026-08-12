//! Front-end B census over cel's Charon LLBC.
//!
//! Ported from the `test_aheui_census.rs` that lived here until
//! `f1f4ffbe358` moved it to `aheui/aheui-jit/tests/`. Nothing has ever run
//! this pipeline over `cel.ullbc` — the moved harness censuses aheui, and the
//! only other caller of any analyze entry point is `pyre-jit-trace/build.rs`.
//! This is a measurement, not an acceptance gate: it prints what front-end B
//! can and cannot digest and asserts nothing about the numbers.
//!
//! Two probes, deliberately independent:
//!
//! * `cel_census_call_sites` lowers **every** local body with
//!   `lower_fun_decl` and classifies every call site it produces. It needs no
//!   portal, so its coverage is the whole artefact rather than one BFS
//!   closure. This is the probe that answers which call sites are walls.
//! * `cel_census_pipeline_*` runs the production analyzer from a portal seed,
//!   which additionally exercises the codewriter and annotator. Coverage is
//!   the portal's graph closure only.
//!
//! ```sh
//! cargo test --release -p majit-translate --test test_cel_census \
//!     -- --nocapture --test-threads=1
//! ```
//!
//! No `--ignored` here, and the omission is deliberate: the `ignore` below is
//! `cfg_attr(debug_assertions, …)`, so in a release build these tests are not
//! ignored at all and `--ignored` selects *nothing* — "running 0 tests;
//! 3 filtered out", which reads like a pass. The template this was ported from
//! documents the `--ignored` form and has the same hole.
//!
//! `--test-threads=1` is not decoration: each pipeline invocation re-seeds
//! process-global registries (`STRUCT_ORIGIN_REGISTRY`, …), so two probes
//! running concurrently race on them.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use majit_charon_reader::Llbc;
use majit_translate::front::mir::{LowerError, lower_fun_decl};
use majit_translate::{
    AnalyzeConfig, CallPath, CallTarget, HostStaticAddrs, JitDriverSpec, OpKind, PipelineConfig,
};

/// `cel-jit/build/llbc/cel.ullbc`, or `CEL_CENSUS_LLBC`. Returns `None` when
/// absent so the test skips instead of failing in a checkout that has never
/// run the extractor.
fn cel_llbc_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CEL_CENSUS_LLBC") {
        return Some(PathBuf::from(p));
    }
    let path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../cel-jit/build/llbc/cel.ullbc"
    ));
    path.exists().then_some(path)
}

fn skip_note() {
    eprintln!(
        "skipping: cel.ullbc missing — run `python3 scripts/extract-llbc.py cel` \
         in cel-jit, or set CEL_CENSUS_LLBC"
    );
}

fn bump(counts: &mut BTreeMap<&'static str, usize>, key: &'static str) {
    *counts.entry(key).or_default() += 1;
}

/// Lower every local body and classify every call site in the result.
///
/// The wall classes are named as such: a wall stops the graph, whereas an
/// ordinary residual call is a normal lowering that the codewriter continues
/// past. `__dyn_call` is documented at `front/mir.rs:16542` as "not a
/// lowering, it is a placeholder: an unregistered synthetic path that stops
/// whatever graph reaches it".
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: lowers the whole cel LLBC; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_call_sites() {
    let Some(path) = cel_llbc_path() else {
        skip_note();
        return;
    };
    let llbc = Llbc::load(&path).expect("load cel llbc");

    let mut fns_total = 0usize;
    let mut fns_bodyless = 0usize;
    let mut fns_failed: BTreeMap<String, usize> = BTreeMap::new();
    let mut graphs = Vec::new();
    // Leaf names of bodies that lowered. Used only to split FunctionPath call
    // sites into "callee has a graph here" and "callee does not"; leaf-keyed,
    // so the ambiguity count below is reported alongside it rather than
    // silently absorbed.
    let mut lowered_leaves: BTreeMap<String, usize> = BTreeMap::new();

    for fd in llbc.iter_local_fns() {
        if fd.is_global_initializer.is_some() {
            continue;
        }
        fns_total += 1;
        if fd.unstructured().is_none() {
            fns_bodyless += 1;
            continue;
        }
        match lower_fun_decl(&llbc, fd) {
            Ok(graph) => {
                if let Some(leaf) = graph.name.rsplit("::").next() {
                    *lowered_leaves.entry(leaf.to_string()).or_default() += 1;
                }
                graphs.push(graph);
            }
            Err(err) => {
                let class = match &err {
                    LowerError::FunctionNotFound(_) => "FunctionNotFound".to_string(),
                    LowerError::Schema(_) => "Schema".to_string(),
                    // Keep the leading clause only: the tail carries block and
                    // local numbers, which would make every failure unique and
                    // turn the histogram into a list.
                    LowerError::Unsupported(msg) => {
                        let head: String = msg.chars().take(60).collect();
                        format!("Unsupported: {head}")
                    }
                };
                *fns_failed.entry(class).or_default() += 1;
            }
        }
    }

    let mut sites = BTreeMap::new();
    let mut residual_callees: BTreeMap<String, usize> = BTreeMap::new();
    let mut indirect_sites: BTreeMap<String, usize> = BTreeMap::new();
    // Wall sites named by the graph that owns them, not merely counted. The
    // relocation A/B/A over `PYRE_FNPTR_INDIRECT` claims the *same* sites move
    // between the two wall classes; a matching count of 6 in both arms is
    // consistent with that and also with six different sites, and only the
    // names discriminate. Same reason the 17 `Indirect` sites are enumerated
    // below rather than compared to #104's 17 by cardinality.
    let mut wall_owners: BTreeMap<String, usize> = BTreeMap::new();
    for graph in &graphs {
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::Call { target, .. } => match target {
                        CallTarget::FunctionPath { segments } => {
                            let leaf = segments.last().map(String::as_str).unwrap_or("");
                            if leaf == "__dyn_call" {
                                bump(&mut sites, "WALL  call __dyn_call (graph-stopping)");
                                *wall_owners.entry(graph.name.clone()).or_default() += 1;
                            } else if lowered_leaves.contains_key(leaf) {
                                bump(&mut sites, "      call FunctionPath, callee lowered here");
                            } else {
                                bump(&mut sites, "      call FunctionPath, no local graph");
                                *residual_callees.entry(leaf.to_string()).or_default() += 1;
                            }
                        }
                        CallTarget::Method { .. } => {
                            bump(&mut sites, "      call Method (receiver dispatch)")
                        }
                        CallTarget::SyntheticTransparentCtor { .. } => bump(
                            &mut sites,
                            "      call SyntheticTransparentCtor (enum shell)",
                        ),
                        CallTarget::Indirect {
                            trait_root,
                            method_name,
                        } => {
                            bump(&mut sites, "      call Indirect (vtable arm)");
                            // Enumerated, not just counted: #104 reports a set of
                            // the same size bound by a receiver-agnostic
                            // unique-impl-by-name fallback. Lowering to `Indirect`
                            // and being mis-bound afterwards are different stages
                            // and can both be true, so the two sets have to be
                            // compared member by member rather than by cardinality.
                            *indirect_sites
                                .entry(format!("{trait_root}::{method_name}"))
                                .or_default() += 1;
                        }
                        CallTarget::UnsupportedExpr => {
                            bump(&mut sites, "WALL  call UnsupportedExpr");
                            *wall_owners.entry(graph.name.clone()).or_default() += 1;
                        }
                    },
                    OpKind::IndirectCall { graphs, .. } => match graphs {
                        Some(candidates) if !candidates.is_empty() => {
                            bump(&mut sites, "      indirect-call, candidate graphs present")
                        }
                        Some(_) => bump(&mut sites, "      indirect-call, empty candidate list"),
                        None => {
                            bump(&mut sites, "WALL  indirect-call, graphs=None");
                            *wall_owners.entry(graph.name.clone()).or_default() += 1;
                        }
                    },
                    _ => {}
                }
            }
        }
    }

    let ambiguous_leaves = lowered_leaves.values().filter(|n| **n > 1).count();
    let total_sites: usize = sites.values().sum();
    let walls: usize = sites
        .iter()
        .filter(|(k, _)| k.starts_with("WALL"))
        .map(|(_, n)| *n)
        .sum();

    eprintln!("=== cel front-end B census: {} ===", path.display());
    // Reported by the process about itself, so an A/B/A log is self-attesting
    // about process identity. It matters for exactly one claim: the relocation
    // A/B/A over `PYRE_FNPTR_INDIRECT` is only immune to in-process
    // memoization if the legs are distinct processes, and three identical pids
    // across three legs would say the opposite. `lower_fun_decl` holds no
    // cross-call cache today (it re-derives `derive_program_metadata` per
    // body), so distinct pids make the argument twice over rather than resting
    // on the absence of a cache anyone could add later.
    eprintln!(
        "pid {}  PYRE_FNPTR_INDIRECT={}",
        std::process::id(),
        std::env::var("PYRE_FNPTR_INDIRECT").unwrap_or_else(|_| "<unset>".into())
    );
    eprintln!("local fns visited        {fns_total}");
    eprintln!("  no body (opaque)       {fns_bodyless}");
    eprintln!("  lowered                {}", graphs.len());
    eprintln!(
        "  refused                {}",
        fns_failed.values().sum::<usize>()
    );
    for (class, n) in &fns_failed {
        eprintln!("      {n:6}  {class}");
    }
    eprintln!("call sites in lowered graphs  {total_sites}");
    for (class, n) in &sites {
        eprintln!("      {n:6}  {class}");
    }
    eprintln!("walls                    {walls} of {total_sites}");
    eprintln!(
        "leaf-keyed control: {ambiguous_leaves} of {} lowered leaves are owned by >1 body, \
         so the FunctionPath split above is approximate by that much",
        lowered_leaves.len()
    );
    eprintln!("wall sites, by owning graph:");
    for (name, n) in &wall_owners {
        eprintln!("      {n:6}  {name}");
    }
    eprintln!("vtable (Indirect) sites, by trait::method:");
    for (name, n) in &indirect_sites {
        eprintln!("      {n:6}  {name}");
    }
    let mut top: Vec<(&String, &usize)> = residual_callees.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    eprintln!("top callees with no local graph:");
    for (name, n) in top.into_iter().take(20) {
        eprintln!("      {n:6}  {name}");
    }
}

/// Opnames in a `JitCode::dump()`, one per assembled instruction.
///
/// `format_assembler` (`codewriter/format.rs:112-`) writes one line per
/// `FlatOp`, optionally prefixed by a `%4d  ` bytecode position, and every
/// non-label arm opens with the opname. Labels are `L<n>:` and are the only
/// lines that are not instructions, so they are the only thing filtered.
fn dump_opnames(dump: &str) -> impl Iterator<Item = &str> {
    dump.lines().filter_map(|line| {
        let mut tokens = line.split_whitespace();
        let first = tokens.next()?;
        // The position prefix is a bare integer; the opname is the next token.
        let name = if first.bytes().all(|b| b.is_ascii_digit()) {
            tokens.next()?
        } else {
            first
        };
        (!name.ends_with(':')).then_some(name)
    })
}

/// Print the design-of-record §1 cells for one portal.
///
/// STOP: The point of this function is that §1 was **measured but not
/// reproducible** (#107): the numbers came from an ad-hoc harness that was
/// never committed, so the cells could be read but not re-derived. Nothing
/// here asserts — it prints, and the doc is annotated with what it prints.
///
/// WARNING: Two of §1's rows are *not* re-derivable from what the pipeline returns,
/// and are deliberately absent rather than approximated:
///
/// * the ULLBC `Drop` / `Call` terminator / fn counts are over the portal's
///   ULLBC closure, which this result does not carry — a whole-artefact count
///   would be a different denominator wearing the same name;
/// * the "real-computation ops" percentage depends on which opnames count as
///   `binop` / `arrayread` / `arraywrite`, and **that definition was never
///   recorded**. So the full opname histogram is printed instead: any family
///   split is re-derivable from it, including one that disagrees with mine.
fn section1_cells(label: &str, result: &majit_translate::pipeline::ProgramPipelineResult) {
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for jitcode in &result.jitcodes {
        for name in dump_opnames(&jitcode.dump()) {
            *hist.entry(name.to_string()).or_default() += 1;
        }
    }
    let ops: usize = hist.values().sum();
    // §1's `ops` cell does not say whether it counts the two pseudo-ops that
    // `format_assembler` emits alongside real instructions: `-live-`
    // (`FlatOp::Live`, a liveness marker) and `---` (`FlatOp::EndOfBlock`).
    // Both are in `ssarepr.insns` and neither is an instruction anyone
    // executes, so the cell is ambiguous by three ways of counting. Print all
    // three rather than pick one — the doc's number can then be matched
    // against whichever rule reproduces it instead of being declared drift.
    let live_markers = hist.get("-live-").copied().unwrap_or(0);
    let block_markers = hist.get("---").copied().unwrap_or(0);
    let count_prefix = |p: &str| -> usize {
        hist.iter()
            .filter(|(k, _)| k.starts_with(p))
            .map(|(_, n)| n)
            .sum()
    };
    let exact = |k: &str| -> usize { hist.get(k).copied().unwrap_or(0) };

    eprintln!("--- §1 cells [{label}] ---");
    eprintln!("  jitcodes                  {}", result.jitcodes.len());
    eprintln!("  ops (all dump lines)      {ops}");
    eprintln!(
        "  ops less `-live-`         {}      <- the rule that reproduces §1",
        ops - live_markers
    );
    eprintln!(
        "  ops less `-live-` + `---` {}",
        ops - live_markers - block_markers
    );
    eprintln!(
        "  residual_call* : inline_call*   {} : {}",
        count_prefix("residual_call"),
        count_prefix("inline_call")
    );
    eprintln!("  guard_class               {}", exact("guard_class"));
    // STOP: SKILL.md:762-764: the op prints as `vtablemethodptr` in the dump while
    // the insns-table key is `vtable_method_ptr/rd>i`. Matching only the latter
    // against a dump silently reads 0 — a whole flag was once filed on that
    // zero. Both spellings are printed so the trap cannot be re-entered.
    eprintln!(
        "  vtablemethodptr (dump)    {}   [insns-table spelling `vtable_method_ptr`: {}]",
        exact("vtablemethodptr"),
        exact("vtable_method_ptr")
    );
    eprintln!(
        "  new / new_with_vtable     {} / {}",
        exact("new"),
        exact("new_with_vtable")
    );
    eprintln!(
        "  indirectcalltarget_indices  {}",
        result.indirectcalltarget_indices.len()
    );
    eprintln!("  opname histogram ({} distinct):", hist.len());
    let mut rows: Vec<(&String, &usize)> = hist.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (name, n) in rows {
        eprintln!("      {n:6}  {name}");
    }
}

fn run_pipeline_census(label: &str, portal: CallPath) {
    let Some(path) = cel_llbc_path() else {
        skip_note();
        return;
    };
    // SAFETY: serialized test binary (`--test-threads=1`); set before the
    // pipeline reads it and before any worker spawns.
    unsafe { std::env::set_var("PYRE_MIR_FRONTEND_LLBC", &path) };

    let config = AnalyzeConfig {
        pipeline: PipelineConfig {
            transform: Default::default(),
            jit_drivers: vec![JitDriverSpec {
                portal,
                greens: Vec::new(),
                reds: Vec::new(),
                autoreds: false,
                virtualizables: Vec::new(),
                red_types: Vec::new(),
            }],
            register_trait_families: Vec::new(),
        },
    };

    // A panic here is expected output, not a bug to be silenced. The pipeline
    // is designed to fail loud on a shape it cannot digest, and it prints its
    // census histograms before reaching that point — so catching the unwind
    // and reporting the message is what makes this a measurement at all.
    // Anyone "fixing" this into a quiet fallback destroys the exact signal the
    // harness exists to collect.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        majit_translate::analyze_multiple_pipeline_with_modules(
            &[],
            &config,
            None,
            &|_, _| None,
            &[],
            HostStaticAddrs::default(),
        )
    }));

    match outcome {
        Ok(result) => {
            eprintln!("=== cel pipeline census [{label}]: completed ===");
            eprintln!("jitcodes emitted: {}", result.jitcodes.len());
            let names: BTreeSet<String> = result
                .jitcodes_by_path
                .keys()
                .map(|k| k.canonical_key())
                .collect();
            eprintln!("jitcode paths ({}): {names:#?}", names.len());
            let mut insns: Vec<&String> = result.insns.keys().collect();
            insns.sort_unstable();
            eprintln!("insn vocabulary ({}): {insns:?}", insns.len());
            section1_cells(label, &result);
        }
        Err(err) => {
            let msg = err
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| err.downcast_ref::<&str>().copied())
                .unwrap_or("<non-string panic>");
            eprintln!("=== cel pipeline census [{label}]: panicked ===");
            eprintln!("panic: {msg}");
        }
    }
}

/// The design-of-record P0.b "typed VM control" column names
/// `clean_interp_seeded_f` (`cel/src/majit/bytecode.rs`). Seeding there is the
/// closest thing to a reproduction of that column that exists.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_typed_vm() {
    run_pipeline_census(
        "clean_interp_seeded_f",
        CallPath::from_segments(["majit", "bytecode", "float_bank", "clean_interp_seeded_f"]),
    );
}

/// `run_mainloop_f` (`cel/src/majit/bytecode.rs`). §1 pins this portal harder
/// than any other — it is the one the design document gives *both* artefact
/// shapes for (`SKILL.md:749-757`): 16 jitcodes / 2048 ops / 263 : 20 /
/// `new_ops` 4 whole-crate, against 13 / 1873 / 239 : 20 / 0 scoped. Run it
/// against `cel-portals.ullbc`; the whole-crate column is the control that
/// proves the divergence is scope rather than provenance, not a target.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_mainloop() {
    run_pipeline_census(
        "run_mainloop_f",
        CallPath::from_segments(["majit", "bytecode", "float_bank", "run_mainloop_f"]),
    );
}

/// `cel::vm::eval`, the bytecode VM entry. A free function, which matters:
/// `register_configured_jitdrivers` (`lib.rs:1942`) asserts the portal
/// resolves to an exact graph in `call_control.function_graphs()`, and the
/// walker's own entry (`cel::objects::<Impl>::resolve_value`) is an inherent
/// method rather than a free function.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_vm_eval() {
    run_pipeline_census("vm::eval", CallPath::from_segments(["vm", "eval"]));
}

/// The design-of-record P0.b "AST walker" column names
/// `objects::Value::resolve_val` (`cel/src/objects.rs`, since renamed to
/// `resolve_value`). Crate-stripped and *not* free-function-widened: only free
/// function spellings get alias paths, so an associated function has to be
/// named by its owning impl.
///
/// Seed this against the scoped `cel-portals.ullbc`, which is the artefact
/// shape P0.b §1 pinned. The whole-crate `cel.ullbc` has a different type
/// universe and its numbers are not comparable to the doc's.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_walker() {
    run_pipeline_census(
        "objects::Value::resolve_value",
        CallPath::from_segments(["objects", "Value", "resolve_value"]),
    );
}
