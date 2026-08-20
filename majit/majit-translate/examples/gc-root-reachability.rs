//! Answer `framework.py`'s question over a charon artefact: which functions can
//! reach application-level Python, and therefore a collection.
//!
//! Usage: `cargo run -p majit-translate --release --example gc-root-reachability -- <file.ullbc>...`

use majit_translate::memory::gctransform::{framework, liveness};

const PYTHON_DISPATCH_SEEDS_REF: &[&str] = framework::PYTHON_DISPATCH_SEEDS;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: gc-root-reachability <file.ullbc>...");
        std::process::exit(2);
    }
    // Artefacts whose call graph is joined in but whose bodies are not scanned.
    // One artefact is not one program: `pyre-jit.ullbc` carries the portal and
    // the blackhole but hardly any of the interpreter they hand control to, so
    // asked alone it answers that nothing reaches a collection.  Naming
    // `pyre-interpreter.ullbc` here is what makes that answer mean anything.
    // Donors are loaded once and their `Llbc` dropped: only the graph is kept,
    // because holding two multi-hundred-megabyte artefacts open at once is what
    // the memory goes to.
    let donors: Vec<String> = std::env::var("GC_JOIN_WITH")
        .ok()
        .map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let donor_graphs: Vec<(String, framework::CallGraph)> = donors
        .iter()
        .map(|path| {
            let llbc = match majit_charon_reader::Llbc::load(path) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("{path}: {e:?}");
                    std::process::exit(1);
                }
            };
            (path.clone(), framework::build(&llbc))
        })
        .collect();

    for path in &args {
        let llbc = match majit_charon_reader::Llbc::load(path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("{path}: {e:?}");
                std::process::exit(1);
            }
        };
        let cg = framework::build(&llbc);
        let total = cg.names.len();
        println!("== {} ({} fun_decls) ==", llbc.crate_name(), total);
        // Which crates this artefact actually carries bodies for.  Charon
        // inlines the dependency graph, so one artefact is usually enough for
        // a cross-crate answer — but that has to be shown, not assumed.
        let mut per_crate: std::collections::BTreeMap<&str, usize> = Default::default();
        for name in cg.names.values() {
            per_crate
                .entry(name.split("::").next().unwrap_or("?"))
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
        let mut roots: Vec<(&&str, &usize)> = per_crate.iter().collect();
        roots.sort_by(|a, b| b.1.cmp(a.1));
        let shown: Vec<String> = roots
            .iter()
            .take(6)
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        println!(
            "   crates with bodies   : {} ({}{})",
            per_crate.len(),
            shown.join(" "),
            if per_crate.len() > 6 { " …" } else { "" }
        );
        // Every question below — the seeds, the closure, the opaque taint — is
        // asked of the joined graph and the answer projected back onto this
        // artefact's ids, because the liveness scan walks *these* bodies.  With
        // no donors the join is a renumbering of `cg` and nothing changes.
        let mut parts: Vec<&framework::CallGraph> = vec![&cg];
        parts.extend(donor_graphs.iter().map(|(_, g)| g));
        let joined = framework::Joined::build(&parts);
        for (donor, g) in &donor_graphs {
            println!(
                "   joined with          : {} ({} fun_decls)",
                donor,
                g.names.len()
            );
        }
        if !donor_graphs.is_empty() {
            println!(
                "   joined graph          : {} nodes; {} spelling(s) merged across artefacts, \
                 {} held apart as ambiguous",
                joined.graph.names.len(),
                joined.joined_names,
                joined.ambiguous_names
            );
        }
        // A seed that matches nothing empties half the closure silently, and
        // the name table is the only place to see why.  Substring, not the
        // anchored form the seeds use, so a near-miss spelling still shows up.
        if let Ok(pat) = std::env::var("GC_NAME_GREP") {
            let mut hits: Vec<&String> = joined
                .graph
                .names
                .values()
                .filter(|n| n.contains(&pat))
                .collect();
            hits.sort();
            println!("   names containing {pat:?}: {}", hits.len());
            for n in hits.iter().take(60) {
                println!("       {n}");
            }
        }
        // "Can this particular helper collect?" is the question every
        // adjudication of an opaque finding turns on, and a reachability bit
        // is worthless without the chain behind it.
        if let Ok(pat) = std::env::var("GC_PATH_FROM") {
            let (py0, _) = joined.graph.seeds_for(PYTHON_DISPATCH_SEEDS_REF);
            let (col0, _) = joined.graph.seeds_for(framework::COLLECTING_SEEDS);
            let mut sd = py0;
            sd.extend(col0);
            let mut hits: Vec<(&u64, &String)> = joined
                .graph
                .names
                .iter()
                .filter(|(_, n)| n.contains(&pat))
                .collect();
            hits.sort_by_key(|(_, n)| n.as_str());
            for (id, n) in hits.iter().take(20) {
                match joined.graph.path_to(**id, &sd) {
                    Some(chain) => println!("   {n}\n       REACHES: {}", chain.join(" -> ")),
                    None => println!("   {n}\n       reaches no seed"),
                }
            }
        }
        println!(
            "   python-dispatch seeds : {}",
            joined.graph.seed_report(framework::PYTHON_DISPATCH_SEEDS)
        );
        println!(
            "   collecting-alloc seeds: {}",
            joined.graph.seed_report(framework::COLLECTING_SEEDS)
        );
        let (jpy, _) = joined.graph.seeds_for(framework::PYTHON_DISPATCH_SEEDS);
        let (jcol, _) = joined.graph.seeds_for(framework::COLLECTING_SEEDS);
        let mut joined_seeds = jpy;
        joined_seeds.extend(jcol);
        let reach = joined.project(0, &joined.graph.reaching(&joined_seeds));
        // The seeds as *this* artefact spells them.  The tiering below compares
        // a finding's `callee_id`, which is an id in this artefact, so it needs
        // the local set; the closure above needs the joined one.
        let (py, _) = cg.seeds_for(framework::PYTHON_DISPATCH_SEEDS);
        let (col, _) = cg.seeds_for(framework::COLLECTING_SEEDS);
        let mut seeds = py;
        seeds.extend(col);
        println!(
            "   can reach a collection: {} / {} ({}%)",
            reach.len(),
            total,
            if total == 0 {
                0
            } else {
                reach.len() * 100 / total
            }
        );
        // What the join bought, so a run without donors is never mistaken for
        // the whole-program answer this one is.
        if !donor_graphs.is_empty() {
            let alone = cg.reaching(&seeds).len();
            println!(
                "       without the join  : {alone} / {total} — the join admits {} more",
                reach.len().saturating_sub(alone)
            );
        }
        println!(
            "   unresolved-callee fns : {} (call through fn-ptr / dyn / unresolved trait)",
            cg.indirect.len()
        );

        // Judge the hand-written brackets: a `push_roots()` scope can only be
        // protecting against a collection if its own function can reach one.
        // This is a *necessary* condition, so a function that fails it holds a
        // bracket that nothing in this crate can justify.
        let pin_ids: Vec<u64> = cg
            .names
            .iter()
            .filter(|(_, n)| n.ends_with("gc_roots::push_roots"))
            .map(|(&id, _)| id)
            .collect();
        if pin_ids.is_empty() {
            // Not a reason to stop: an artefact that brackets nothing is the
            // one most likely to hold defects, and the liveness scan below is
            // exactly what answers for it.
            println!("   (no gc_roots::push_roots in this artefact's name table)");
        }
        let mut bracketed: Vec<u64> = cg
            .callees
            .iter()
            .filter(|(_, cs)| pin_ids.iter().any(|p| cs.contains(p)))
            .map(|(&id, _)| id)
            .collect();
        bracketed.sort_unstable();
        // An unresolved callee is undecidable *transitively*: a helper that
        // dispatches through a `global_hook!` function pointer makes every one
        // of its callers undecidable too, not just itself.  Without this the
        // "cannot reach a collection" bucket silently absorbs them.
        let opaque = joined.project(0, &joined.graph.reaching(&joined.graph.indirect));
        let (mut justified, mut undecidable, mut unjustified) = (0usize, 0usize, Vec::new());
        for id in &bracketed {
            if reach.contains(id) {
                justified += 1;
            } else if opaque.contains(id) {
                undecidable += 1;
            } else {
                unjustified.push(*id);
            }
        }
        let mut traits: Vec<(&String, &usize)> = cg.opaque.dyn_trait.iter().collect();
        traits.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let top: Vec<String> = traits
            .iter()
            .take(8)
            .map(|(n, c)| format!("{}={c}", n.rsplit("::").next().unwrap_or(n)))
            .collect();
        println!(
            "   opaque call census    : dyn-trait={} sites over {} traits, fn-value={}, unknown={}",
            traits.iter().map(|(_, c)| **c).sum::<usize>(),
            traits.len(),
            cg.opaque.fn_value,
            cg.opaque.unknown
        );
        let mut variants: Vec<(&&str, &usize)> = cg.opaque.by_variant.iter().collect();
        variants.sort_by(|a, b| b.1.cmp(a.1));
        println!(
            "       by charon spelling: {}",
            variants
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        println!("       top traits: {}", top.join(" "));
        if std::env::var("GC_OPAQUE_SOURCES").is_ok() {
            let src = framework::dynamic_call_sources(&llbc);
            let mut rows: Vec<(&String, &usize)> = src.iter().collect();
            rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            println!("       fn-value callee sources:");
            for (label, count) in rows.iter().take(20) {
                println!("           {count:5}  {label}");
            }
        }
        println!(
            "   fns tainted by an unresolved callee (transitive): {} / {}",
            opaque.len(),
            total
        );
        println!(
            "   functions holding a push_roots bracket: {}",
            bracketed.len()
        );
        println!("       can reach a collection (justified) : {justified}");
        println!("       hold an unresolved call (undecided): {undecidable}");
        println!(
            "       cannot reach any collection        : {}",
            unjustified.len()
        );
        for id in unjustified.iter().take(40) {
            println!(
                "           {}",
                cg.names.get(id).map_or("?", String::as_str)
            );
        }
        if unjustified.len() > 40 {
            println!("           … and {} more", unjustified.len() - 40);
        }

        // The other direction, and the one that finds defects rather than
        // overhead: a call that can collect, a GC pointer live across it, and
        // no bracket anywhere in the function.
        let gc_tys = liveness::gc_ptr_type_ids(&llbc);
        if gc_tys.is_empty() {
            println!("   (no PyObjectRef type id found — liveness scan skipped)");
            continue;
        }
        let push_root_ids: std::collections::HashSet<u64> = pin_ids.iter().copied().collect();
        let (found, stats) = liveness::scan(
            &llbc,
            &cg,
            &reach,
            &push_root_ids,
            &gc_tys,
            liveness::MOVABLE_GC_MARKERS,
        );
        println!(
            "   liveness scan: {} bodies; {} with a terminator this reader could not parse; \
             {} call(s) withheld as dominated by a push_roots",
            stats.bodies_scanned, stats.unparsed_terminator_bodies, stats.withheld_under_a_bracket
        );
        println!(
            "       and {} body/bodies with a statement this reader could not parse",
            stats.unparsed_statement_bodies
        );
        // The resolved graph is an *under*-approximation of what collects: a
        // call whose dispatch edge is unresolved is excluded, so a clean
        // resolved census is not a clean census.  Re-run with the opaque set
        // folded in and report both, rather than let the difference go unsaid.
        let mut conservative = reach.clone();
        conservative.extend(opaque.iter().copied());
        let (found_conservative, stats_conservative) = liveness::scan(
            &llbc,
            &cg,
            &conservative,
            &push_root_ids,
            &gc_tys,
            liveness::MOVABLE_GC_MARKERS,
        );
        let conservative_fns: std::collections::BTreeSet<&str> = found_conservative
            .iter()
            .map(|f| f.func_name.as_str())
            .collect();
        let mut by_fn: std::collections::BTreeMap<&str, Vec<&liveness::Finding>> =
            Default::default();
        for f in &found {
            by_fn.entry(f.func_name.as_str()).or_default().push(f);
        }
        let non_arg_fns: Vec<&&str> = by_fn
            .iter()
            .filter(|(_, v)| v.iter().any(|f| !f.live_non_arg.is_empty()))
            .map(|(k, _)| k)
            .collect();
        println!(
            "   unbracketed calls that can collect with a live PyObjectRef: {} in {} fn(s)",
            found.len(),
            by_fn.len()
        );
        println!(
            "       counting unresolved dispatch as collecting too: {} in {} fn(s)",
            found_conservative.len(),
            conservative_fns.len()
        );
        // The conservative scan covers a superset of bodies, so its own
        // withheld and unparsed figures are the ones its finding count has to
        // be read against; quoting the resolved scan's line next to it would
        // pair a count with another scan's accounting.
        println!(
            "           over {} bodies; {} with an unparsable terminator, {} with an \
             unparsable statement; {} call(s) withheld under a push_roots",
            stats_conservative.bodies_scanned,
            stats_conservative.unparsed_terminator_bodies,
            stats_conservative.unparsed_statement_bodies,
            stats_conservative.withheld_under_a_bracket
        );
        println!(
            "       of which hold a NON-ARGUMENT live pointer: {} fn(s)",
            non_arg_fns.len()
        );
        // Tier 1: the callee is *itself* a dispatch seed, so "this call runs
        // Python" needs no transitive argument.  Tier 2 reaches Python only
        // through further calls and is far likelier to be a false positive.
        let mut tier1: Vec<&liveness::Finding> = found
            .iter()
            .filter(|f| seeds.contains(&f.callee_id) && !f.live_non_arg.is_empty())
            .collect();
        tier1.sort_by(|a, b| a.func_name.cmp(&b.func_name).then(a.line.cmp(&b.line)));
        let t1_fns: std::collections::BTreeSet<&str> =
            tier1.iter().map(|f| f.func_name.as_str()).collect();
        println!(
            "       tier 1 (callee IS a dispatch seed): {} call(s) in {} fn(s)",
            tier1.len(),
            t1_fns.len()
        );
        if std::env::var("GC_LIVENESS_TIER1").is_ok() {
            for f in &tier1 {
                println!(
                    "           {}:{}  across {}  live: {:?}",
                    f.func_name,
                    f.line,
                    f.callee_name.rsplit("::").next().unwrap_or(""),
                    f.live_non_arg
                );
            }
        }
        // Tier 1.5: not a direct dispatch seed, but a live pointer that this
        // body later hands to a `list`/`dict` accessor — the shape both
        // confirmed defects in `list.index` and `mul` had.
        let mut tier15: Vec<&liveness::Finding> = found
            .iter()
            .filter(|f| !seeds.contains(&f.callee_id) && !f.movable_use.is_empty())
            .collect();
        tier15.sort_by(|a, b| a.func_name.cmp(&b.func_name).then(a.line.cmp(&b.line)));
        let t15_fns: std::collections::BTreeSet<&str> =
            tier15.iter().map(|f| f.func_name.as_str()).collect();
        println!(
            "       tier 1.5 (live ptr later addressed as list/dict): {} call(s) in {} fn(s)",
            tier15.len(),
            t15_fns.len()
        );
        let t15_conservative = found_conservative
            .iter()
            .filter(|f| !seeds.contains(&f.callee_id) && !f.movable_use.is_empty())
            .count();
        println!("       tier 1.5 counting unresolved dispatch too: {t15_conservative} call(s)");
        if std::env::var("GC_LIVENESS_TIER15").is_ok() {
            for f in &tier15 {
                println!(
                    "           {}:{}  across {}  movable-use: {:?}",
                    f.func_name,
                    f.line,
                    f.callee_name.rsplit("::").next().unwrap_or(""),
                    f.movable_use
                );
                match cg.path_to(f.callee_id, &seeds) {
                    Some(chain) => println!("               path: {}", chain.join(" -> ")),
                    None => println!("               path: (none — reached only via a seed alias)"),
                }
            }
        }
        let show: usize = std::env::var("GC_LIVENESS_SHOW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        for name in non_arg_fns.iter().take(show) {
            // Every such call in the body, not the first one found.  A single
            // row per function hides a second unrooted use further down the
            // same body — and that is precisely the one a repair pass, working
            // from this list, then leaves behind.
            for f in by_fn[**name].iter().filter(|f| !f.live_non_arg.is_empty()) {
                println!(
                    "           {}:{}  across {}  live: {:?}",
                    name,
                    f.line,
                    f.callee_name.rsplit("::").next().unwrap_or(""),
                    f.live_non_arg
                );
            }
        }

        // The same question asked of the running frame.  A `PyObjectRef` has a
        // root stack to be pinned on; the frame is carried as a bare
        // `&mut PyFrame` no walker reaches, so the only repair is to re-read it
        // out of a `FrameAnchor` after the call.  That reload kills the stale
        // local at the call, so no bracket set is passed — a body that reloads
        // correctly simply has nothing live across the call to report.
        let frame_tys = liveness::frame_ptr_type_ids(&llbc);
        if frame_tys.is_empty() {
            println!("   (no PyFrame pointer type id found — frame scan skipped)");
            continue;
        }
        let no_bracket: std::collections::HashSet<u64> = Default::default();
        let (frames, frame_stats) =
            liveness::scan(&llbc, &cg, &reach, &no_bracket, &frame_tys, &[]);
        let (frames_conservative, _) =
            liveness::scan(&llbc, &cg, &conservative, &no_bracket, &frame_tys, &[]);
        let frame_fns: std::collections::BTreeSet<&str> =
            frames.iter().map(|f| f.func_name.as_str()).collect();
        println!(
            "   frame carried across a call that can collect: {} in {} fn(s)",
            frames.len(),
            frame_fns.len()
        );
        println!(
            "       counting unresolved dispatch as collecting too: {} call(s)",
            frames_conservative.len()
        );
        println!(
            "           over {} bodies; {} with an unparsable terminator, {} with an \
             unparsable statement",
            frame_stats.bodies_scanned,
            frame_stats.unparsed_terminator_bodies,
            frame_stats.unparsed_statement_bodies
        );
        // The frame reaches the callee as an argument in most of these, and an
        // argument goes just as stale as any other local here (the module
        // header records why upstream can drop them and pyre cannot).  Both
        // columns are therefore hazards; they are split only because a frame
        // the call does not even see is the more obviously wrong shape.
        let mut frame_tier1: Vec<&liveness::Finding> = frames
            .iter()
            .filter(|f| seeds.contains(&f.callee_id))
            .collect();
        frame_tier1.sort_by(|a, b| a.func_name.cmp(&b.func_name).then(a.line.cmp(&b.line)));
        let ft1_fns: std::collections::BTreeSet<&str> =
            frame_tier1.iter().map(|f| f.func_name.as_str()).collect();
        println!(
            "       tier 1 (callee IS a dispatch seed): {} call(s) in {} fn(s)",
            frame_tier1.len(),
            ft1_fns.len()
        );
        if let Ok(filter) = std::env::var("GC_FRAME_SCAN") {
            // A whole-crate list is not a work item; the filter is how a
            // surface (`OpcodeStepExecutor`, say) is cut out of it and read.
            // The resolved rows are marked, so the ones only the conservative
            // scan finds — a body whose dispatch this reader cannot follow —
            // are adjudicated as the weaker claim they are rather than read
            // beside the others.
            let resolved: std::collections::HashSet<(u64, u64, u64)> = frames
                .iter()
                .map(|f| (f.func, f.line, f.callee_id))
                .collect();
            for f in &frames_conservative {
                if !filter.is_empty() && !f.func_name.contains(&filter) {
                    continue;
                }
                let mark = if resolved.contains(&(f.func, f.line, f.callee_id)) {
                    "resolved"
                } else {
                    "opaque  "
                };
                println!(
                    "           [{mark}] {}:{}  across {}  live: {:?} arg: {:?}",
                    f.func_name,
                    f.line,
                    f.callee_name.rsplit("::").next().unwrap_or(""),
                    f.live_non_arg,
                    f.live_arg
                );
            }
        }
    }
}
