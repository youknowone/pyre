//! Answer `framework.py`'s question over a charon artefact: which functions can
//! reach application-level Python, and therefore a collection.
//!
//! Usage: `cargo run -p majit-translate --release --example gc-root-reachability -- <file.ullbc>...`

use majit_translate::memory::gctransform::{framework, liveness};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: gc-root-reachability <file.ullbc>...");
        std::process::exit(2);
    }
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
        println!(
            "   python-dispatch seeds : {}",
            cg.seed_report(framework::PYTHON_DISPATCH_SEEDS)
        );
        println!(
            "   collecting-alloc seeds: {}",
            cg.seed_report(framework::COLLECTING_SEEDS)
        );
        let (py, _) = cg.seeds_for(framework::PYTHON_DISPATCH_SEEDS);
        let (col, _) = cg.seeds_for(framework::COLLECTING_SEEDS);
        let mut seeds = py;
        seeds.extend(col);
        let reach = cg.reaching(&seeds);
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
        let opaque = cg.reaching(&cg.indirect);
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
        let (found, stats) = liveness::scan(&llbc, &cg, &reach, &push_root_ids, &gc_tys);
        println!(
            "   liveness scan: {} bodies; {} with a terminator this reader could not parse; \
             {} call(s) withheld as dominated by a push_roots",
            stats.bodies_scanned, stats.unparsed_terminator_bodies, stats.withheld_under_a_bracket
        );
        // The resolved graph is an *under*-approximation of what collects: a
        // call whose dispatch edge is unresolved is excluded, so a clean
        // resolved census is not a clean census.  Re-run with the opaque set
        // folded in and report both, rather than let the difference go unsaid.
        let mut conservative = reach.clone();
        conservative.extend(opaque.iter().copied());
        let (found_conservative, _) =
            liveness::scan(&llbc, &cg, &conservative, &push_root_ids, &gc_tys);
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
            let v = &by_fn[**name];
            let f = v.iter().find(|f| !f.live_non_arg.is_empty()).unwrap();
            println!(
                "           {}:{}  across {}  live: {:?}",
                name,
                f.line,
                f.callee_name.rsplit("::").next().unwrap_or(""),
                f.live_non_arg
            );
        }
    }
}
