#[path = "src/call_spec.rs"]
mod call_spec;
#[path = "src/llbc_fingerprint.rs"]
mod llbc_fingerprint;
#[path = "src/virtualizable_spec.rs"]
mod virtualizable_spec;

use walkdir::WalkDir;

/// The translation prepass churns the whole graph universe through short-lived
/// allocations; the system allocator keeps the freed spans and the process's
/// peak RSS tracks the churn rather than the live set.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const CODEGEN_CACHE_VERSION: &str = "pyre-jit-trace-codegen-cache-v5";
/// Retained cache entries. Each is ~6 MB, and a handful covers the
/// configurations one checkout switches between (native/wasm × release/dev).
const CODEGEN_CACHE_MAX_ENTRIES: usize = 8;
/// Rewritten on every cache hit; its mtime is the entry's last-use stamp.
const CODEGEN_CACHE_USED_MARKER: &str = ".last-used";
const CODEGEN_OUTPUTS: &[&str] = &[
    "jit_trace_gen.rs",
    "jit_metadata.json",
    "jitcodes.bin",
    "jitcodes_index.bin",
    "indirectcalltargets.bin",
    "jit_drivers.bin",
    "insns.bin",
    "descrs.bin",
    "ei_descr_mints.bin",
    "liveness.bin",
    "fnaddr_bindings.bin",
    "static_pytype_bindings.bin",
    "static_ref_bindings.bin",
];

/// Lowering switches read by `majit-translate` while this build script runs.
/// They affect generated graphs, so Cargo and the content cache must both see
/// their values; otherwise an A/B can silently restore the opposite setting.

/// Outputs whose bytes are this build-script process's own addresses, by
/// construction: `fnaddr_bindings` is `(path, build_fnaddr)` and the two
/// `*_bindings` tables are `(name, build_addr)`, captured so
/// `runtime_fnaddr_patch` can re-pair them with the runtime's addresses (see
/// the comments at their write sites). ASLR moves them on every process, so
/// two processes always disagree and that disagreement is not a defect.
///
/// Excluded from the cross-process verdict only. Within one process the
/// addresses are the same, so `DeterminismCheck::InProcess` still judges
/// them — a difference there would be real.
///
/// Caching them is still sound: a restore serves this table and the
/// `constants_i` baked against it from the *same* generation, so they stay
/// consistent with each other.
const HOST_ADDRESSED_OUTPUTS: &[&str] = &[
    "fnaddr_bindings.bin",
    "static_pytype_bindings.bin",
    "static_ref_bindings.bin",
];

/// Opt-in self-check that the generated outputs are a function of the inputs
/// [`codegen_cache_key`] hashes. Deliberately not part of that key: it changes
/// nothing about what gets generated, it only compares two generations.
const DETERMINISM_CHECK_ENV: &str = "PYRE_CODEGEN_DETERMINISM_CHECK";
/// Subdirectory of `OUT_DIR` holding the second generation's outputs. Left in
/// place after the comparison so a reported difference can be diffed by hand.
const DETERMINISM_PROBE_SUBDIR: &str = "determinism-probe";
/// Changing this value re-runs the build script and nothing else — see
/// [`emit_rerun_directives`] for why the `cache` mode cannot be exercised
/// without it.
const DETERMINISM_NONCE_ENV: &str = "PYRE_CODEGEN_NONCE";

/// Which pair of generations [`codegen_outputs_match`] compares.
///
/// Off by default because it doubles a multi-minute prepass.
///
/// * `1` / `in-process` — generate a second time into
///   [`DETERMINISM_PROBE_SUBDIR`] and compare the two sets byte for byte.
/// * `cache` — compare this run's outputs against the entry already stored
///   under the same cache key, i.e. against a *different* process's bytes.
///   Reports that it compared nothing when no entry exists yet; editing this
///   file rekeys the cache, so that is the expected result of the first run
///   after any change here.
///
/// Neither mode subsumes the other. `in-process` catches whatever is carried
/// in mutable process state — a never-reset global registry, an allocation
/// address used as an order — but it shares this process's hash seeds with
/// the run it compares against, so a seed-derived order is invisible to it.
/// `cache` sees the seed class as well, and only ever compares against a key
/// that already has an entry.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DeterminismCheck {
    Off,
    InProcess,
    AgainstCache,
}

impl DeterminismCheck {
    /// Panics on an unrecognised value rather than reading it as `Off`: a
    /// misspelled gate that silently checks nothing reports exactly like a
    /// clean run.
    fn from_env() -> Self {
        match std::env::var(DETERMINISM_CHECK_ENV)
            .unwrap_or_default()
            .as_str()
        {
            "" | "0" => Self::Off,
            "1" | "in-process" => Self::InProcess,
            "cache" => Self::AgainstCache,
            other => {
                panic!("{DETERMINISM_CHECK_ENV}={other} is not one of: 0, 1, in-process, cache")
            }
        }
    }
}

const LOWERING_GATE_ENV: [&str; 5] = [
    "PYRE_DYN_INDIRECT",
    "PYRE_FNPTR_INDIRECT",
    "PYRE_MIR_FRAMESTATE",
    "PYRE_OPTION_RESIDUAL_NARROW",
    "PYRE_TUPLE_PER_SHAPE_CLASSDEF",
];

/// Build script for pyre-jit: runs majit-translate on the active pyre
/// interpreter to auto-generate tracing code. This is the Rust
/// equivalent of RPython's translation pipeline.
///
/// Analyzes all source files from:
/// - pyre-object (Python object types: W_IntObject, W_FloatObject, etc.)
/// - pyre-interpreter (object space, bytecode dispatch, eval loop)
fn main() {
    println!("cargo::rerun-if-env-changed=MAJIT_LLBC_EXTRACTION");
    if std::env::var_os("MAJIT_LLBC_EXTRACTION").as_deref() == Some(std::ffi::OsStr::new("1")) {
        emit_llbc_extraction_placeholders();
        return;
    }
    // Fail fast with an actionable message when the Charon-extracted LLBC
    // artefacts the codegen consumes are absent.  Without this, the missing
    // set surfaces deep inside `real_main` as a worker-thread `panic!`
    // (`build-script worker panicked: Any { .. }`) printed *below* the
    // 150-line source-path dump, where it is easy to miss.
    preflight_llbc_or_fail();

    // The codegen worker (`real_main`) runs the RPythonTyper
    // specialization (`specialize_legacy_graph_with_registry` → annotator
    // `complete_pending_blocks` / rtyper `specialize_more_blocks`).  Its
    // visit order is keyed off the annotator/rtyper worklist maps
    // (`genpendingblocks`, `annotated`, `all_blocks`, …), which are
    // insertion-ordered `IndexMap`s, so the order in which the callee
    // specialization chain is walked does not depend on the per-process
    // SipHash seed.
    //
    // That covers the walk order and nothing downstream of it.  It does not
    // establish that the emitted bytes are a function of the inputs: two
    // generations of the same sources have been observed to produce different
    // `jitcodes.bin` / `descrs.bin` / `ei_descr_mints.bin` while
    // `jitcodes_index.bin` held still, from ordering and population decided
    // after the walk (heap addresses used as a canonical order, a
    // process-global mint registry).  `DeterminismCheck` is the opt-in check;
    // an earlier reading of the paragraph above concluded "a single in-process
    // run suffices", and that conclusion is why no check existed to catch it.
    //
    // The 1 GiB thread stack is needed for syn's recursive parse of ~150
    // files plus the rtyper chain (on Windows the main thread's 1 MiB default
    // would STATUS_STACK_OVERFLOW).
    run_worker();
}

/// Break the LLBC bootstrap dependency cycle explicitly.
///
/// Charon compiling `pyre-jit` must compile its `pyre-jit-trace` dependency
/// before `pyre-jit.ullbc` exists. None of these artifacts execute during
/// extraction; they only need to satisfy `include!` / `include_bytes!` so
/// rustc can expose `pyre-jit`'s MIR. The next normal Cargo build observes
/// `MAJIT_LLBC_EXTRACTION` changing and replaces every placeholder.
fn emit_llbc_extraction_placeholders() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set");
    std::fs::write(
        format!("{out_dir}/jit_trace_gen.rs"),
        "pub const COMPILED_JIT_DRIVERS: &[(&str, usize)] = &[];\n\
         pub const CANONICAL_JITCODES: &[(&str, usize)] = &[];\n",
    )
    .unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), b"{}\n").unwrap();
    std::fs::write(format!("{out_dir}/jitcodes.bin"), b"").unwrap();
    std::fs::write(
        format!("{out_dir}/jitcodes_index.bin"),
        bincode::serialize(&(Vec::<String>::new(), vec![0_u32])).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/indirectcalltargets.bin"),
        bincode::serialize(&Vec::<usize>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/jit_drivers.bin"),
        bincode::serialize(&Vec::<majit_translate::CompiledJitDriver>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/insns.bin"),
        bincode::serialize(&std::collections::BTreeMap::<String, u8>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/descrs.bin"),
        bincode::serialize(&Vec::<majit_translate::jitcode::BhDescr>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/ei_descr_mints.bin"),
        bincode::serialize(&Vec::<majit_ir::effectinfo::DescrMintEntry>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/liveness.bin"),
        bincode::serialize(&Vec::<u8>::new()).unwrap(),
    )
    .unwrap();
    for name in [
        "fnaddr_bindings.bin",
        "static_pytype_bindings.bin",
        "static_ref_bindings.bin",
    ] {
        std::fs::write(
            format!("{out_dir}/{name}"),
            bincode::serialize(&Vec::<(String, i64)>::new()).unwrap(),
        )
        .unwrap();
    }
}

/// Crates whose Charon artefact this build consumes.  The `.ullbc` and its
/// `.fingerprint` stamp are both named after the crate.  Pyre production
/// configures the exact `eval::eval_loop_jit` portal, so unlike generic
/// two-artefact consumers it requires `pyre-jit` too.
const LLBC_CRATES: &[&str] = &["majit-rlib", "pyre-object", "pyre-interpreter", "pyre-jit"];

/// Pre-flight the LLBC prerequisite, mirroring the resolution order in
/// `majit-translate` (`build_semantic_program_via_active_frontend`):
/// honour the `PYRE_MIR_FRONTEND_LLBC` override, else require the canonical
/// `build/llbc/{pyre-object,pyre-interpreter,pyre-jit}.ullbc` set. The third
/// artifact contains the exact `eval::eval_loop_jit` portal.
///
/// When neither resolves, emit a clean, copy-pasteable bootstrap message
/// and fail the build *before* the worker spawns — so the contributor
/// sees the exact steps to run instead of a worker-thread panic buried
/// under the source-file dump.  Auto-running the bootstrap from here is
/// deliberately avoided: `scripts/extract-llbc.py` shells out to a nested
/// `cargo build`, which would block on the outer build's target-directory
/// lock (deadlock), and a build script that downloads a toolchain breaks
/// hermetic / offline / CI builds.
///
/// That ban does not cover the stamp comparison in `fail_if_llbc_stale`:
/// `scripts/extract-llbc.py --fingerprint` returns before `extract` runs and
/// only performs a `cargo metadata` walk plus `git ls-files`, so it starts no
/// nested build and takes no target-directory lock.
fn preflight_llbc_or_fail() {
    // Explicit override: trust it and let the translator validate the
    // individual paths (its loader panics per-file with the bad path).
    if std::env::var_os("PYRE_MIR_FRONTEND_LLBC")
        .map(|v| std::env::split_paths(&v).any(|p| !p.as_os_str().is_empty()))
        .unwrap_or(false)
    {
        return;
    }

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let llbc_dir = repo_root.join("build").join("llbc");
    let mut missing: Vec<String> = LLBC_CRATES
        .iter()
        .map(|crate_name| format!("{crate_name}.ullbc"))
        .filter(|name| !llbc_dir.join(name).exists())
        .collect();
    // Cross-compiling additionally needs this target's layout sidecars:
    // Charon resolves struct layouts per target, and the artefacts above
    // carry the extraction host's.  Without them every descr field past the
    // first pointer names the wrong bytes on a target whose pointers are a
    // different width, which corrupts reads instead of failing.
    for name in llbc_layout_sidecars() {
        if !llbc_dir.join(&name).exists() {
            missing.push(name);
        }
    }
    if missing.is_empty() {
        // Present is not the same as current: the artefacts are frozen
        // snapshots (AGENTS.md:48) and nothing above compares them to the
        // sources they were extracted from.
        fail_if_llbc_stale(&repo_root);
        return;
    }

    let charon_present = repo_root
        .join("build")
        .join("charon")
        .join("charon")
        .exists();

    // `cargo::error=` lines (no embedded newlines) surface in Cargo's
    // error summary on modern Cargo and fail the build on their own; the
    // framed stderr block below stays readable on every toolchain, and the
    // explicit non-zero exit is the belt-and-suspenders stop for a Cargo
    // too old to recognise `cargo::error`.
    println!(
        "cargo::error=pyre-jit codegen needs the Charon-extracted LLBC artefacts, but build/llbc/ is missing: {}",
        missing.join(", "),
    );
    if !charon_present {
        println!("cargo::error=Install charon (one-time): scripts/install-charon.py");
    }
    // Cross-compiling: the driver's default `LAYOUT_TARGETS` only covers
    // wasm32, so any other target's sidecars are produced only when the
    // extraction is told this target explicitly.  Advertise it (harmless but
    // redundant for wasm32).  The override must also list the sidecar paths,
    // since `PYRE_MIR_FRONTEND_LLBC` is treated as the complete input set and
    // returns before the sidecar check above.
    let sidecars = llbc_layout_sidecars();
    let (extract_cmd, override_extra) = if sidecars.is_empty() {
        ("scripts/extract-llbc.py".to_string(), String::new())
    } else {
        let target = std::env::var("TARGET").unwrap_or_default();
        (
            format!("LLBC_LAYOUT_TARGETS={target} scripts/extract-llbc.py"),
            sidecars
                .iter()
                .map(|name| format!(":/abs/{name}"))
                .collect::<String>(),
        )
    };
    println!("cargo::error=Extract the LLBC: {extract_cmd}");

    // The install step is only needed when the charon binary is absent;
    // with it present the fix is the single extract command.
    let install_line = if charon_present {
        String::new()
    } else {
        "   scripts/install-charon.py                            # one-time\n".to_string()
    };
    eprintln!(
        "\n\
========================================================================\n\
 pyre-jit-trace: JIT codegen prerequisite missing\n\
------------------------------------------------------------------------\n\
 The Charon-extracted LLBC artefacts are required but were not found:\n\
{}\n\
 Bootstrap (run from the repo root):\n\
{}\
   {}\n\
\n\
 …or point the build at existing artefacts:\n\
   export PYRE_MIR_FRONTEND_LLBC={}{}\n\
========================================================================\n",
        missing
            .iter()
            .map(|name| format!("   build/llbc/{name}"))
            .collect::<Vec<_>>()
            .join("\n"),
        install_line,
        extract_cmd,
        // Spelled from `LLBC_CRATES` because setting this override returns
        // before the mandatory-artefact check: an example short one artefact
        // is a working command that silently drops that crate's bodies.
        LLBC_CRATES
            .iter()
            .map(|crate_name| format!("/abs/{crate_name}.ullbc"))
            .collect::<Vec<_>>()
            .join(":"),
        override_extra,
    );

    std::process::exit(1);
}

// The stamp and the `--fingerprint` stdout are the same `key=value` format
// written by the same producer, so both are parsed by `llbc_fingerprint`
// rather than once per reader here.  `tests/llbc_fingerprint_format_test.rs`
// pins that module against the driver's real output.
use llbc_fingerprint::{FreshnessMode, freshness_policy, parse_fingerprint_stdout, stamp_field};

/// Wait for the fingerprint oracle with a deadline.
///
/// A build script that blocks forever is strictly worse than a missing
/// warning, and `std` has no `wait_timeout`, so the wait runs on a helper
/// thread and a timeout abandons the child.  Every non-answer — spawn failure,
/// non-zero exit, unparseable stdout — collapses to `None`, i.e. silence: an
/// unavailable oracle must not break offline, hermetic or vendored builds.
fn llbc_fingerprint_output(child: std::process::Child) -> Option<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });
    let output = rx
        .recv_timeout(std::time::Duration::from_secs(120))
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    parse_fingerprint_stdout(&stdout)
}

/// Ask the extraction driver what the current sources hash to.
///
/// `scripts/llbc_extract.py` prints the same `key=value` lines it writes into
/// the stamp, and its `source=` field is the same `source_fingerprint()` call
/// `stamp_for` records — so there is one implementation of the digest and it
/// is not this one.  Parsed by `llbc_fingerprint::parse_fingerprint_stdout`,
/// which reads the field by name. The driver may add fields, as it did for
/// `external=`; a reader that models the output as one bare value stops
/// answering as soon as that happens.
///
/// `CARGO_FEATURES` and `LLBC_LAYOUT_TARGETS` are replayed out of the stamp:
/// `fingerprint_inputs` (`scripts/llbc_extract.py:255-360`) walks the
/// dependency closure under the feature set and the cross-target layout set in
/// force at extraction time, so recomputing under this build's defaults would
/// report a difference the sources do not have.
fn llbc_source_fingerprint(
    repo_root: &std::path::Path,
    driver: &std::path::Path,
    crate_name: &str,
    features: &str,
    layout_targets: &str,
) -> Option<String> {
    for python in ["python3", "python"] {
        let spawned = std::process::Command::new(python)
            .arg(driver)
            .arg("--fingerprint")
            .arg(crate_name)
            .current_dir(repo_root)
            .env("CARGO_FEATURES", features)
            .env("LLBC_LAYOUT_TARGETS", layout_targets)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn();
        let Ok(child) = spawned else {
            continue;
        };
        return llbc_fingerprint_output(child);
    }
    None
}

/// Compare every `.ullbc` against the sources it was extracted from.
///
/// The prepass reads `build/llbc/*.ullbc`, never the Rust sources
/// (AGENTS.md:48), so a field inserted anywhere but last in a `#[repr(C)]`
/// struct shifts every following descr offset while the build stays green.
/// The existence test in `preflight_llbc_or_fail` cannot see this, and
/// `codegen_cache_key` makes it worse rather than better: it hashes every
/// `<crate>/src`, so a source edit reliably invalidates the codegen cache and
/// re-runs the prepass — over the same unchanged artefact.
///
/// The oracle is the extractor's own stamp.  `scripts/llbc_extract.py:643-652`
/// already skips a crate whose stamp still matches, so this is the comparison
/// the producer trusts, evaluated by the consumer.
///
/// A stale artefact fails the build.  It was warning-only, on the argument
/// that the build still works for everything whose layout did not move; the
/// measured cost of that leniency is the opposite — a binary built over a
/// stale artefact reads the *old* layout while the sources say otherwise, so
/// every measurement taken from it describes code that is not in the tree, and
/// the warning scrolls past inside `target/*/build/*/output` where nobody
/// reads it.  `PYRE_LLBC_STRICT=0` demotes it back to a warning for a
/// deliberately-stale working build, and `PYRE_LLBC_SKIP_FINGERPRINT_CHECK`
/// skips the comparison entirely.
///
/// `PYRE_LLBC_SKIP_FINGERPRINT_CHECK=1` opts out entirely — announcing that it
/// did, so opting out and passing stay distinguishable.  Both take exactly
/// `1`, any other value is reported rather than ignored, and setting both is
/// refused; see `llbc_fingerprint::freshness_policy` for why that is a
/// contradiction and not a precedence question.
///
/// Three outcomes per crate, on two channels: **fresh** is silent, **stale**
/// errors (or warns with strict mode disabled), and **unknown** — unreadable stamp, or an
/// oracle that did not answer — warns without ever being promoted.  Keeping
/// unknown off the silent channel is the point: otherwise "no warning" means
/// both *checked and current* and *nobody checked*, and a reader cannot tell
/// which build they got.  Cf. the same conflation in `llbc_extract.py`'s
/// direct-invocation no-op.
fn fail_if_llbc_stale(repo_root: &std::path::Path) {
    println!("cargo::rerun-if-env-changed=PYRE_LLBC_STRICT");
    println!("cargo::rerun-if-env-changed=PYRE_LLBC_SKIP_FINGERPRINT_CHECK");
    // A non-UTF8 value is not `1`, so lossy conversion lands it in the
    // unrecognised arm and it is reported rather than read as unset.
    let strict_var =
        std::env::var_os("PYRE_LLBC_STRICT").map(|value| value.to_string_lossy().into_owned());
    let skip_var = std::env::var_os("PYRE_LLBC_SKIP_FINGERPRINT_CHECK")
        .map(|value| value.to_string_lossy().into_owned());
    let policy = freshness_policy(strict_var.as_deref(), skip_var.as_deref());
    let policy_directive = match policy.mode {
        FreshnessMode::Refuse => "cargo::error",
        _ => "cargo::warning",
    };
    for line in &policy.diagnostics {
        println!("{policy_directive}={line}");
    }
    match policy.mode {
        // Both switches set. Refuse rather than pick one: see `freshness_policy`.
        FreshnessMode::Refuse => std::process::exit(1),
        FreshnessMode::Skip => return,
        FreshnessMode::Warn | FreshnessMode::Strict => {}
    }
    let driver = repo_root.join("scripts").join("extract-llbc.py");
    if !driver.is_file() {
        return;
    }
    let llbc_dir = repo_root.join("build").join("llbc");
    let mut stale: Vec<(&str, String, String)> = Vec::new();
    // Freshness has THREE outcomes, and the third used to be spelled the same
    // way as "fresh": silence.  A crate whose stamp is unreadable, or whose
    // oracle does not answer, is *unknown* — and "nobody checked" is not a
    // licence to read its field offsets, it is the absence of the check that
    // would grant one.
    let mut unknown: Vec<(&str, &'static str)> = Vec::new();
    for &crate_name in LLBC_CRATES {
        // An absent artefact is the bootstrap case, already reported by the
        // prepass; only an artefact that EXISTS can be silently trusted, so
        // only that case is worth calling unknown.
        let artefact_exists = llbc_dir.join(format!("{crate_name}.ullbc")).is_file();
        let stamp_path = llbc_dir.join(format!("{crate_name}.ullbc.fingerprint"));
        let Ok(stamp) = std::fs::read_to_string(&stamp_path) else {
            if artefact_exists {
                unknown.push((crate_name, "no .fingerprint stamp beside the artefact"));
            }
            continue;
        };
        let Some(recorded) = stamp_field(&stamp, "source=") else {
            unknown.push((crate_name, "stamp carries no source= line"));
            continue;
        };
        let features = stamp_field(&stamp, "features=").unwrap_or_default();
        let layout_targets = stamp_field(&stamp, "layout_targets=").unwrap_or_default();
        let current =
            llbc_source_fingerprint(repo_root, &driver, crate_name, &features, &layout_targets);
        let Some(current) = current else {
            unknown.push((crate_name, "the fingerprint oracle did not answer"));
            continue;
        };
        if current != recorded {
            stale.push((crate_name, recorded, current));
        }
    }
    // Reported as a warning and never promoted by `PYRE_LLBC_STRICT`: the
    // documented contract is that an unavailable oracle must not break offline,
    // hermetic or vendored builds, and failing here would break exactly the
    // builds the fallback exists for.  Saying so costs one line and removes the
    // ambiguity.  `PYRE_LLBC_SKIP_FINGERPRINT_CHECK` acknowledges and silences.
    for (crate_name, reason) in &unknown {
        println!(
            "cargo::warning=LLBC FRESHNESS UNKNOWN: {crate_name}.ullbc was not checked \
             ({reason}); it may or may not match the current sources"
        );
    }
    if stale.is_empty() {
        return;
    }
    // The directive string is the only difference between the two modes, so it
    // is chosen once and the same lines go through it.  `cargo::warning=` and
    // `cargo::error=` each carry a single line with no embedded newline.
    let strict = policy.mode == FreshnessMode::Strict;
    let directive = if strict {
        "cargo::error"
    } else {
        "cargo::warning"
    };
    for (crate_name, recorded, current) in &stale {
        println!(
            "{directive}=LLBC STALE: {crate_name}.ullbc was extracted at source={recorded}, \
             sources now hash to {current}"
        );
    }
    let crates = stale
        .iter()
        .map(|(crate_name, _, _)| *crate_name)
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "{directive}=Field offsets read out of these artefacts may name the wrong bytes; \
         re-extract with: python3 scripts/extract-llbc.py {crates}"
    );
    if strict {
        std::process::exit(1);
    }
}

/// Run the codegen worker on a large-stack thread, propagating any panic so
/// the build fails loudly instead of emitting partial output.
fn run_worker() {
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(real_main)
        .expect("spawn build-script worker")
        .join()
        .expect("build-script worker panicked");
}

fn real_main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let pyre_base = format!("{manifest_dir}/..");
    let repo_root = format!("{manifest_dir}/../..");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Collect ALL source file paths from the active interpreter crates.
    // Only the paths are consumed — module-path derivation below plus
    // `cargo::rerun-if-changed` tracking; the graph bodies come from the
    // Charon-extracted LLBC set, so the file contents are never read.
    let source_dirs = [
        format!("{pyre_base}/pyre-object/src"),
        format!("{pyre_base}/pyre-interpreter/src"),
    ];

    let mut source_paths = Vec::new();

    for dir in &source_dirs {
        collect_rs_files(dir, &mut source_paths);
    }

    // Include the canonical portal source so its module-qualified identity is
    // present in the analyzer manifest. Production configures the exact
    // `eval::eval_loop_jit` CallPath; there is no handler-level fallback.
    // Using the real portal preserves the dispatch loop that blackhole resume
    // needs to continue from a user bytecode PC.
    //
    // Single-file inclusion (not the whole `pyre-jit/src` tree)
    // because `eval_loop_jit` is the only function in pyre-jit that
    // belongs in the portal closure; the rest of pyre-jit
    // (codewriter, assembler, regalloc, etc.) is JIT infrastructure
    // that must NOT be analyzed as user code, and the orchestration
    // would inflate analysis time and risk pulling unrelated
    // helpers into `find_all_graphs(portal)` BFS.
    let eval_path = format!("{pyre_base}/pyre-jit/src/eval.rs");
    collect_single_file(&eval_path, &mut source_paths);

    // `materialize_virtual_from_rd` (`pyre-jit/src/eval.rs`) destructures
    // `majit_ir::RdVirtualInfo` enum variants whose named fields carry
    // primitive concretetypes (`kind: u8`, `descr_index: u32`, ...).
    // Without the enum declaration in the analyser source set the variant
    // arms surface as untyped, and the rtyper's exitswitch emission for
    // `match descr.kind { 0 => ..., 1 => ..., 2 => ..., }` falls back to
    // 'r' (Ref), which `flatten.rs:385` rejects as
    // `switch exitswitch must be int`.  Single-file inclusion mirrors the
    // `pyre-jit/src/eval.rs` carve-out above: only the resoperation enum
    // declarations belong in the analyser closure, not the rest of
    // majit-ir which would inflate analysis time and pull JIT
    // infrastructure into the user-code BFS.
    let resoperation_path = format!("{repo_root}/majit/majit-ir/src/resoperation.rs");
    collect_single_file(&resoperation_path, &mut source_paths);

    eprintln!(
        "[pyre-jit-trace build.rs] collected {} source paths from {} dirs (+ pyre-jit/src/eval.rs): {:?}",
        source_paths.len(),
        source_dirs.len(),
        source_paths,
    );

    // Run analysis on ALL files with PyFrame virtualizable metadata.
    //
    // This keeps the generic analyzer closer to the proc-macro/runtime path:
    // graph rewrite can recognize `next_instr`, `valuestackdepth`, and
    // `locals_cells_stack_w[*]` as virtualizable accesses before legacy
    // TracePattern classification runs.
    let analyze_config = majit_translate::AnalyzeConfig {
        pipeline: majit_translate::PipelineConfig {
            transform: majit_translate::GraphTransformConfig {
                vable_fields: virtualizable_spec::PYFRAME_VABLE_FIELDS
                    .iter()
                    .map(|(name, idx)| {
                        majit_translate::VirtualizableFieldDescriptor::new(
                            *name,
                            Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                            *idx,
                        )
                    })
                    .collect(),
                vable_arrays: virtualizable_spec::PYFRAME_VABLE_ARRAYS
                    .iter()
                    .map(|(name, idx)| {
                        // virtualizable.py:58 — VirtualizableInfo.array_descrs[i] =
                        // cpu.arraydescrof(getattr(VTYPE, name).TO). Python frame
                        // locals are PyObjectRef pointers: itemsize is one
                        // target word (the build host's word would mis-stride
                        // `FixedObjectArray` on a narrower target), is_signed=false.
                        majit_translate::VirtualizableFieldDescriptor::new_with_arraydescr(
                            *name,
                            Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                            *idx,
                            majit_translate::layout::target_word_size(),
                            false, // is_signed: pointers are unsigned
                        )
                    })
                    .collect(),
                call_effects: build_call_effect_overrides(),
                ..Default::default()
            },
            jit_drivers: vec![
                majit_translate::JitDriverSpec {
                    portal: majit_translate::CallPath::from_segments(["eval", "eval_loop_jit"]),
                    greens: vec![
                        "next_instr".to_string(),
                        "is_being_profiled".to_string(),
                        "pycode".to_string(),
                    ],
                    reds: vec!["frame".to_string(), "ec".to_string()],
                    autoreds: false,
                    virtualizables: vec!["frame".to_string()],
                    red_types: vec!["PyFrame".to_string(), "ExecutionContext".to_string()],
                },
                majit_translate::JitDriverSpec {
                    // pypy/interpreter/baseobjspace.py:1003 `_unpackiterable_unknown_length`;
                    // greens=['greenkey'], reds='auto' (baseobjspace.py:29-32).
                    portal: majit_translate::CallPath::from_segments([
                        "baseobjspace",
                        "_unpackiterable_unknown_length",
                    ]),
                    greens: vec!["greenkey".to_string()],
                    reds: vec![],
                    autoreds: true,
                    virtualizables: vec![],
                    red_types: vec![],
                },
            ],
            // pyre production registers no trait-dispatch families (#346).
            register_trait_families: Vec::new(),
        },
    };
    // warmspot.py:516 `vinfos[VTYPEPTR] = VirtualizableInfo(self, VTYPEPTR)` —
    // pyre's runtime `VirtualizableInfo` constructor lives in the
    // `majit-metainterp` crate (`__build_virtualizable_info`) and runs
    // at `JitDriver::new` (jitdriver.rs:285) where the field offsets
    // resolved by `mem::offset_of!` are available.  build.rs cannot
    // import that crate (no metainterp build-dep, and the offsets are
    // a runtime fact), so the codewriter-side factory returns `None`
    // here; the codewriter slot stays empty until the runtime metainterp
    // setter overrides it.  TODO: documented at
    // `CallControl::make_virtualizable_infos`.
    let vinfo_factory: &majit_translate::VirtualizableInfoFactory<'_> = &|_jd_idx, _vtype| None;
    let fnaddr_bindings = pyre_interpreter::jit_trace_fnaddrs();
    // Prebuilt object-space singleton addresses (static `PyType` pointers
    // and dict-strategy refs).  `majit-translate` is the translation
    // layer and must not import `pyre-object`; the driver supplies these
    // across the translation boundary.  Resolved here in the same
    // build-script process the translator runs in, so the captured
    // addresses match a direct `&pyre_object::X` read at the codewriter
    // call site.
    let mut static_pytype_addrs = pyre_interpreter::jit_static_pytype_addrs();
    // This script is compiled for the host even when the crate it feeds is
    // built for wasm, so the `#[pyre_class]` registry is populated here and
    // empty there.  Binding a name the wasm runtime cannot re-pair would
    // leave this process's address baked in the constant pool
    // (`runtime_fnaddr_patch::patch_static_addr_constants` rewrites only
    // names it finds in both pools), so drop the registry-derived rows when
    // the target is wasm and let those statics stay unbound instead.
    let registry_rows = pyre_interpreter::pyre_class_pytype_addrs();
    let registry_keys: std::collections::HashSet<&str> =
        registry_rows.iter().map(|&(key, _)| key).collect();
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        static_pytype_addrs.retain(|(key, _)| !registry_keys.contains(key));
    }
    // Counts the rows that survived, not the registry's size: the table
    // drops every registry row whose static a hand-written row already
    // names, and all of them when the target is wasm.
    let kept_from_registry = static_pytype_addrs
        .iter()
        .filter(|(key, _)| registry_keys.contains(key))
        .count();
    eprintln!(
        "[PREPASS statics] pytype rows = {} ({kept_from_registry} of {} \
         #[pyre_class] registry rows kept)",
        static_pytype_addrs.len(),
        registry_rows.len(),
    );
    let static_ref_addrs = pyre_interpreter::jit_static_ref_addrs();
    let static_int_values = pyre_interpreter::jit_static_int_values();
    let static_addrs = majit_translate::HostStaticAddrs {
        pytypes: &static_pytype_addrs,
        refs: &static_ref_addrs,
        int_values: &static_int_values,
    };
    // Per-source crate-stripped module paths — the analyzer-side
    // metadata (`front::mir`) records
    // `struct_origins[bare_name] = module_path`.  Aligns with
    // the runtime's `build_object_descr_group_with_def_path` qualified
    // def-path slot in `gc_cache._cache_size` so a future
    // `path_hash(canonical_struct_name)` analyzer hash lands on the
    // same Arc the runtime publishes (PyPy `cache[STRUCT]` lltype-
    // object identity, descr.py:108-118).
    let module_paths: Vec<String> = source_paths
        .iter()
        .map(|p| module_path_from_source_file(p))
        .collect();
    let module_path_refs: Vec<&str> = module_paths.iter().map(|s| s.as_str()).collect();

    emit_rerun_directives(&repo_root, &source_paths);

    let cache_key = codegen_cache_key(manifest_dir, &repo_root, &source_paths);
    let cache_dir = codegen_cache_dir(&repo_root, &cache_key);
    // The verbose prepass is a census over every attempted graph. Restoring
    // generated outputs would skip the analysis entirely and leave no
    // `PREPASS phaseA/phaseB fail` lines, contradicting the documented
    // `PYRE_RTYPER_VERBOSE=1` workflow. The generated artifacts themselves do
    // not depend on verbosity, so an ordinary build may still reuse the cache.
    let verbose_prepass = std::env::var_os("PYRE_RTYPER_VERBOSE").is_some_and(|value| value == "1");
    // The callee census is emitted from the analysis itself. A cache restore
    // would print no rows, which is indistinguishable from an empty census.
    let callee_census = std::env::var_os("PYRE_CALLEE_CENSUS").is_some_and(|value| value == "1");
    // Restoring the outputs leaves the self-check nothing to compare, so it
    // bypasses the cache for the same reason the verbose prepass does.
    let determinism_check = DeterminismCheck::from_env();
    if !verbose_prepass
        && !callee_census
        && determinism_check == DeterminismCheck::Off
        && restore_codegen_cache(&cache_dir, &out_dir)
    {
        eprintln!(
            "[pyre-jit-trace build.rs] restored generated JIT trace artifacts from cache {}",
            cache_key
        );
        touch_codegen_cache_entry(&cache_dir);
        prune_codegen_cache(&repo_root, &cache_dir);
        return;
    }

    // One whole generation, writing every [`CODEGEN_OUTPUTS`] entry into the
    // directory it is handed. That directory is `OUT_DIR` for the build's own
    // outputs and the probe subdirectory for the second generation
    // `DeterminismCheck::InProcess` compares against; the parameter shadows
    // the outer `out_dir` so both calls read as "write into out_dir".
    let generate_into = |out_dir: &str| {
        let pipeline = majit_translate::analyze_multiple_pipeline_with_modules(
            &module_path_refs,
            &analyze_config,
            None,
            vinfo_factory,
            &fnaddr_bindings,
            static_addrs,
        );

        // Generate tracing code from the canonical graph-first analysis result.
        let code = majit_translate::generate_trace_code_from_pipeline(&pipeline);

        std::fs::write(format!("{out_dir}/jit_trace_gen.rs"), &code).unwrap();

        // JSON metadata for debugging
        let json = serde_json::to_string_pretty(&pipeline).unwrap();
        std::fs::write(format!("{out_dir}/jit_metadata.json"), &json).unwrap();

        // Persist `pipeline.jitcodes` (RPython `all_jitcodes` from
        // codewriter.py:89) as individually encoded entries plus a name/offset
        // index. Runtime materializes entries lazily into the shared
        // MetaInterpStaticData jitcodes store — same single-store model as
        // RPython `warmspot.py:281-282` `self.metainterp_sd.jitcodes =
        // codewriter.make_jitcodes()`.
        let mut jitcodes_bin = Vec::new();
        let mut jitcode_names = Vec::with_capacity(pipeline.jitcodes.len());
        let mut jitcode_offsets = Vec::with_capacity(pipeline.jitcodes.len() + 1);
        jitcode_offsets.push(0_u32);
        for jitcode in &pipeline.jitcodes {
            jitcode_names.push(jitcode.name.clone());
            jitcodes_bin.extend(bincode::serialize(jitcode).unwrap());
            jitcode_offsets.push(
                u32::try_from(jitcodes_bin.len())
                    .expect("serialized jitcodes.bin exceeds the u32 offset range"),
            );
        }
        let jitcodes_index_bin = bincode::serialize(&(jitcode_names, jitcode_offsets)).unwrap();
        std::fs::write(format!("{out_dir}/jitcodes.bin"), &jitcodes_bin).unwrap();
        std::fs::write(format!("{out_dir}/jitcodes_index.bin"), &jitcodes_index_bin).unwrap();
        let indirectcalltargets_bin =
            bincode::serialize(&pipeline.indirectcalltarget_indices).unwrap();
        std::fs::write(
            format!("{out_dir}/indirectcalltargets.bin"),
            &indirectcalltargets_bin,
        )
        .unwrap();

        // Persist the explicit portal → main-JitCode mapping. Runtime consumes
        // this directly instead of rediscovering the portal through name or flag
        // scans.
        let jit_drivers_bin = bincode::serialize(&pipeline.jit_drivers).unwrap();
        std::fs::write(format!("{out_dir}/jit_drivers.bin"), &jit_drivers_bin).unwrap();

        // Persist the runtime opname → u8 table so
        // `JitCode.code` (assembler-local mapping) decodes back to the
        // canonical `(opname, argcodes)` shape at runtime (shadow dispatch,
        // IR diffing).  RPython equivalent: the table handed to
        // `BlackholeInterpBuilder::setup_insns` at metainterp startup
        // (`pyjitpl.py:2227-2243`).
        //
        // RPython parity (`assembler.py:220 self.insns.setdefault(key,
        // len(self.insns))`): the table is the assembler's emission-driven
        // dict, populated by `write_insn` calls during graph flattening.
        // Pyre's analog is `pipeline.insns`, snapshotted from
        // `codewriter.assembler.insns()` after `make_jitcodes` finishes
        // (`majit-translate/src/lib.rs:910`).  Each distinct key gets a
        // fresh byte; the forward map is injective.  `blackhole.py:913`
        // aliases the bhimpl handler under two Python attribute names
        // (`bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not`) but
        // does NOT register a second opname in `Assembler.insns`; the
        // alias is at the dispatch-function-name level only.  Pyre
        // therefore registers exactly one opname per byte; the runtime
        // inverse (`byte → opname`) is 1:1 and panics on duplicate-byte
        // collisions (`jitcode_runtime.rs:INSNS_BYTE_TO_OPNAME`).
        //
        // Serialize through a `BTreeMap` view so the byte output is stable
        // across processes (Rust's `HashMap` SipHash makes raw iteration
        // non-deterministic; RPython's Python dict is insertion-ordered).
        let insns_sorted: std::collections::BTreeMap<&String, &u8> =
            pipeline.insns.iter().collect();
        let insns_bin = bincode::serialize(&insns_sorted).unwrap();
        std::fs::write(format!("{out_dir}/insns.bin"), &insns_bin).unwrap();

        // RPython `blackhole.py:59 self.setup_descrs(asm.descrs)` + `:102-103
        // def setup_descrs(self, descrs): self.descrs = descrs`. Persists the
        // build-time assembler's shared descr pool so that 'd'/'j' argcodes
        // in `JitCode.code` resolve at runtime via
        // `BlackholeInterpBuilder::setup_descrs(...)` — the single-store
        // model (same list consumed by every `BlackholeInterpreter` produced
        // by `acquire_interp`).
        let descrs_bin = bincode::serialize(&pipeline.descrs).unwrap();
        std::fs::write(format!("{out_dir}/descrs.bin"), &descrs_bin).unwrap();

        // `MAJIT_MINT_INDEX_CENSUS=1`: how many `fielddescrof` mints resolved a
        // slot for `index_in_parent`, and how many carried out the `0` they were
        // initialised with. The split the emitted descr cannot record — an
        // unwritten initialiser and a real slot-0 claim are the same bytes once
        // this function returns.
        //
        // STOP: It has to be read HERE and nowhere else, for the reason the comment
        // below already gives: pyre "mints in this process and resolves in
        // another". The counters are process-global atomics in the mint process,
        // so the runtime's `[jit-stats]` line — where every other `field_pos_*`
        // counter is read — starts them at zero again and would report `0 0` on
        // every run however many placeholders this build produced. That reading
        // is worse than no counter: `placeholder=0` is exactly the value that
        // says the hazard is absent.
        //
        // STOP: Gated, and `cargo::warning` only reaches a build script that
        // actually RUNS. A cached build prints nothing, which reads the same as
        // a clean census. Touch a source file in the analyzed set before
        // trusting a silent run.
        if std::env::var_os("MAJIT_MINT_INDEX_CENSUS").is_some() {
            let [claimed, placeholder] = majit_ir::descr::GcCache::mint_index_census();
            println!(
                "cargo::warning=mint_index_census: claimed={claimed} placeholder={placeholder} \
                 (fielddescrof mints; placeholder=0 means no descr in this build carries an \
                 unresolved index_in_parent)"
            );
        }

        // The table above is RPython's `opcode_descrs` (`pyjitpl.py:2261
        // setup_descrs(asm.descrs)`), not its `all_descrs` (`pyjitpl.py:2289
        // self.cpu.setup_descrs()` = the full gccache walk at `descr.py:25-47`).
        // Upstream never has to distinguish them here because one gccache serves
        // one process, so `compute_bitstrings` unions descrs that are already
        // present. Pyre mints in this process and resolves in another, so the
        // raw-set members no opcode names would have no slot on the far side;
        // persist their mint arguments and let the runtime cache take the same
        // `descr.py:224-238` miss branch this one did.
        let ei_descr_mints_bin = bincode::serialize(&pipeline.ei_descr_mints).unwrap();
        std::fs::write(format!("{out_dir}/ei_descr_mints.bin"), &ei_descr_mints_bin).unwrap();

        // RPython `pyjitpl.py:2264 self.liveness_info = "".join(asm.all_liveness)`.
        // Persist the build-time assembler's shared `all_liveness` byte stream so a
        // runtime consumer re-tracing a build-time jitcode (whose `BC_LIVE` ops
        // carry offsets baked against this table) can install it into
        // `metainterp_sd.liveness_info` and resolve those offsets.
        let liveness_bin = bincode::serialize(&pipeline.all_liveness).unwrap();
        std::fs::write(format!("{out_dir}/liveness.bin"), &liveness_bin).unwrap();

        // RPython's translator AOT-compiles every helper into a single binary, so
        // `JitCode.fnaddr` / `constants_i` funcptrs are linker-resolved and stable
        // at runtime.  Pyre's `majit-translate` runs in `build.rs` — a separate
        // process from `pyre-dynasm` — so every fnaddr captured here is the
        // build-script process's address, which ASLR (and the divergent executable
        // layouts) invalidates at runtime.  Persist the `(path, build_fnaddr)`
        // table the codewriter consumed so the runtime patcher
        // (`runtime_fnaddr_patch::patch_constants_i_fnaddrs`) can pair each build
        // address with the matching runtime address from
        // `pyre_interpreter::jit_trace_fnaddrs()` and overwrite stale constants
        // before the walker invokes them.
        let fnaddr_bindings_owned: Vec<(String, i64)> = fnaddr_bindings
            .iter()
            .map(|(p, a)| ((*p).to_string(), *a))
            .collect();
        let fnaddr_bindings_bin = bincode::serialize(&fnaddr_bindings_owned).unwrap();
        std::fs::write(
            format!("{out_dir}/fnaddr_bindings.bin"),
            &fnaddr_bindings_bin,
        )
        .unwrap();

        // Same ASLR hazard for the static-data addresses the codewriter baked
        // into `constants_i` (host `PyType` singletons and prebuilt refs supplied
        // via `HostStaticAddrs`): the build-script process's `&pyre_object::X`
        // address does not survive into the runtime executable.  Persist the
        // `(name, build_addr)` tables so `runtime_fnaddr_patch::
        // patch_constants_i_static_addrs` can re-pair them with the runtime
        // addresses from `jit_static_pytype_addrs` / `jit_static_ref_addrs`.
        let pytype_bindings_owned: Vec<(String, i64)> = static_pytype_addrs
            .iter()
            .map(|(n, a)| ((*n).to_string(), *a))
            .collect();
        std::fs::write(
            format!("{out_dir}/static_pytype_bindings.bin"),
            bincode::serialize(&pytype_bindings_owned).unwrap(),
        )
        .unwrap();
        let ref_bindings_owned: Vec<(String, i64)> = static_ref_addrs
            .iter()
            .map(|(n, a)| ((*n).to_string(), *a))
            .collect();
        std::fs::write(
            format!("{out_dir}/static_ref_bindings.bin"),
            bincode::serialize(&ref_bindings_owned).unwrap(),
        )
        .unwrap();

        // Report
        eprintln!(
            "[pyre-jit-trace build.rs] canonical analysis: {} JIT drivers, {} functions, {} blocks, {} flat ops, {} all_jitcodes ({} bytes bodies + {} bytes index), generated {} bytes",
            pipeline.jit_drivers.len(),
            pipeline.functions.len(),
            pipeline.total_blocks,
            pipeline.total_ops,
            pipeline.jitcodes.len(),
            jitcodes_bin.len(),
            jitcodes_index_bin.len(),
            code.len(),
        );
    };

    generate_into(&out_dir);

    let reproducible = match determinism_check {
        DeterminismCheck::Off => true,
        DeterminismCheck::InProcess => {
            let probe_dir = std::path::Path::new(&out_dir).join(DETERMINISM_PROBE_SUBDIR);
            let _ = std::fs::remove_dir_all(&probe_dir);
            std::fs::create_dir_all(&probe_dir).expect("create the determinism probe directory");
            eprintln!(
                "[pyre-jit-trace build.rs] codegen determinism: generating a second time into {}",
                probe_dir.display()
            );
            // The second generation runs against whatever process-global
            // state the first one left behind. If that state makes it panic,
            // the outputs are not a function of the inputs either, so report
            // it rather than take down a build whose own outputs — the first
            // generation's, already written — are intact.
            let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                generate_into(&probe_dir.to_string_lossy())
            }))
            .is_ok();
            if completed {
                codegen_outputs_match(
                    "a second in-process generation",
                    std::path::Path::new(&out_dir),
                    &probe_dir,
                    false,
                )
            } else {
                println!(
                    "cargo::warning=codegen determinism: the second in-process generation \
                     panicked (see the panic above), so nothing was compared; the codegen does \
                     not survive running twice in one process. Compare across processes with \
                     {DETERMINISM_CHECK_ENV}=cache instead"
                );
                false
            }
        }
        DeterminismCheck::AgainstCache => {
            if CODEGEN_OUTPUTS
                .iter()
                .all(|name| cache_dir.join(name).is_file())
            {
                codegen_outputs_match(
                    &format!("the stored cache entry {cache_key}"),
                    &cache_dir,
                    std::path::Path::new(&out_dir),
                    true,
                )
            } else {
                // Compared nothing, which is neither a pass nor a failure —
                // say so rather than let an empty comparison read as clean.
                // Storing below is what gives the next run something to
                // compare against.
                println!(
                    "cargo::warning=codegen determinism: cache key {cache_key} has no stored \
                     entry, so nothing was compared; re-run this build to compare against the \
                     entry it is about to store"
                );
                true
            }
        }
    };
    if !reproducible {
        println!(
            "cargo::warning=codegen determinism: the outputs at cache key {cache_key} are not \
             reproducible; refusing to store them"
        );
        return;
    }

    if let Err(e) = store_codegen_cache(&cache_dir, &out_dir) {
        eprintln!(
            "[pyre-jit-trace build.rs] warning: could not store generated JIT trace cache {}: {e}",
            cache_key
        );
        return;
    }
    touch_codegen_cache_entry(&cache_dir);
    prune_codegen_cache(&repo_root, &cache_dir);
}

fn build_call_effect_overrides() -> Vec<majit_translate::CallEffectOverride> {
    call_spec::PYFRAME_CALL_EFFECTS
        .iter()
        .map(|spec| {
            let target = match spec.target {
                call_spec::CallTargetSpec::Method {
                    name,
                    receiver_root,
                } => majit_translate::CallTarget::method(name, Some(receiver_root.to_string())),
                call_spec::CallTargetSpec::FunctionPath(segments) => {
                    majit_translate::CallTarget::function_path(segments.iter().copied())
                }
            };
            let effect = match spec.effect {
                call_spec::CallEffectKind::Elidable => majit_translate::CallEffectKind::Elidable,
                call_spec::CallEffectKind::Residual => majit_translate::CallEffectKind::Residual,
                call_spec::CallEffectKind::Declared(declared) => {
                    majit_translate::CallEffectKind::Declared(
                        majit_translate::DeclaredCallEffects {
                            extra: declared_extra_effect(declared.extra),
                            can_collect: declared.can_collect,
                            can_invalidate: declared.can_invalidate,
                        },
                    )
                }
            };
            majit_translate::CallEffectOverride::new(target, effect)
        })
        .collect()
}

/// Carry one declared `EF_*` across the crate boundary.
///
/// Two enums rather than one shared type because `call_spec.rs` is the
/// data-only contract pyre owns and `majit-translate` is the consumer; the
/// arms are enumerated so adding a member on either side is a compile error
/// here rather than a silent re-mapping.
fn declared_extra_effect(
    extra: call_spec::DeclaredExtraEffect,
) -> majit_translate::DeclaredExtraEffect {
    use call_spec::DeclaredExtraEffect as Spec;
    use majit_translate::DeclaredExtraEffect as Translate;
    match extra {
        Spec::ElidableCannotRaise => Translate::ElidableCannotRaise,
        Spec::LoopInvariant => Translate::LoopInvariant,
        Spec::CannotRaise => Translate::CannotRaise,
        Spec::ElidableOrMemoryError => Translate::ElidableOrMemoryError,
        Spec::ElidableCanRaise => Translate::ElidableCanRaise,
        Spec::CanRaise => Translate::CanRaise,
        Spec::ForcesVirtualOrVirtualizable => Translate::ForcesVirtualOrVirtualizable,
    }
}

fn emit_rerun_directives(repo_root: &str, source_paths: &[String]) {
    for path in source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
    emit_rerun_if_changed_recursive(&format!("{repo_root}/majit/majit-translate/src"));
    println!("cargo::rerun-if-changed=src/virtualizable_spec.rs");
    println!("cargo::rerun-if-changed=src/call_spec.rs");
    println!("cargo::rerun-if-env-changed=PYRE_RTYPER_VERBOSE");
    println!("cargo::rerun-if-env-changed=PYRE_CALLEE_CENSUS");
    println!("cargo::rerun-if-env-changed=PYRE_CALLEE_CENSUS_ROWS");
    println!("cargo::rerun-if-env-changed={DETERMINISM_CHECK_ENV}");
    // Re-runs this script without changing anything it hashes. That is the
    // only way to exercise `DeterminismCheck::AgainstCache`, which needs two
    // runs at the *same* cache key: with the gate value held constant Cargo
    // considers the script fresh and skips the second run, and any input edit
    // that would force one also rekeys the cache and leaves nothing to
    // compare against. Deliberately absent from `codegen_cache_key`.
    println!("cargo::rerun-if-env-changed={DETERMINISM_NONCE_ENV}");
    for key in LOWERING_GATE_ENV {
        println!("cargo::rerun-if-env-changed={key}");
    }
    // The mir-frontend analysis derives `jit_trace_gen.rs` from
    // the workspace LLBC artefacts or the `PYRE_MIR_FRONTEND_LLBC`
    // override. Track both so re-extracting LLBC or repointing the override
    // invalidates Cargo's build-script cache and our content cache key.
    println!("cargo::rerun-if-env-changed=PYRE_MIR_FRONTEND_LLBC");
    if let Some(paths) = std::env::var_os("PYRE_MIR_FRONTEND_LLBC") {
        for path in std::env::split_paths(&paths) {
            if !path.as_os_str().is_empty() {
                println!("cargo::rerun-if-changed={}", path.display());
            }
        }
    }
    // Derived from `LLBC_CRATES` rather than repeated: a crate added there is a
    // production input, and a copy of the list that missed it would let cargo
    // skip this script after that artefact alone changed.
    for crate_name in LLBC_CRATES {
        println!("cargo::rerun-if-changed={repo_root}/build/llbc/{crate_name}.ullbc");
    }
    for sidecar in llbc_layout_sidecars() {
        println!("cargo::rerun-if-changed={repo_root}/build/llbc/{sidecar}");
    }
}

/// Layout-sidecar artefact names this build consumes, empty for a native
/// build.
///
/// `scripts/extract-llbc.py` re-extracts each crate for every cross target
/// and reduces the result to its `type_decls`; `majit-translate`'s
/// `auto_discover_workspace_llbc_paths` merges them ahead of the host
/// artefacts so their Charon-resolved field offsets win.
fn llbc_layout_sidecars() -> Vec<String> {
    let target = std::env::var("TARGET").unwrap_or_default();
    let host = std::env::var("HOST").unwrap_or_default();
    if !majit_translate::layout::is_cross_target(&target, &host) {
        return Vec::new();
    }
    // `pyre-jit` is absent: it has no sidecar (see the extraction driver's
    // spec for why), so only the object model gets cross-target layouts.
    // The naming convention lives in `majit_translate::layout` so it stays in
    // lockstep with `auto_discover_workspace_llbc_paths`.
    ["majit-rlib", "pyre-object", "pyre-interpreter"]
        .iter()
        .map(|crate_name| majit_translate::layout::layout_sidecar_filename(crate_name, &target))
        .collect()
}

/// The cache sits under `build/` rather than the Cargo target directory so
/// `cargo clean`, a fresh `CARGO_TARGET_DIR`, or a profile switch does not
/// re-pay the translation prepass — the same reason the prepass's own inputs
/// (`build/llbc/*.ullbc`) live there. `codegen_cache_key` already folds
/// HOST/TARGET/PROFILE/OPT_LEVEL, the feature set, the build-script binary's
/// own bytes and the LLBC content, so an entry is only ever served back to the
/// configuration that produced it.
fn codegen_cache_base(repo_root: &str) -> std::path::PathBuf {
    std::path::Path::new(repo_root).join("build/pyre-jit-trace-cache")
}

fn codegen_cache_dir(repo_root: &str, cache_key: &str) -> std::path::PathBuf {
    codegen_cache_base(repo_root)
        .join(CODEGEN_CACHE_VERSION)
        .join(cache_key)
}

/// Record that `cache_dir` was used, so [`prune_codegen_cache`] evicts by last
/// use rather than by creation: a configuration built daily but keyed long ago
/// would otherwise be dropped ahead of one-off keys minted by a source edit.
fn touch_codegen_cache_entry(cache_dir: &std::path::Path) {
    let _ = std::fs::write(cache_dir.join(CODEGEN_CACHE_USED_MARKER), b"");
}

/// Drop entries beyond [`CODEGEN_CACHE_MAX_ENTRIES`], least recently used
/// first, along with any directory left by an earlier `CODEGEN_CACHE_VERSION`.
/// Every distinct (target, profile, feature set, build-script binary, LLBC
/// content) combination mints a key and nothing removed them before, so the
/// directory grew without bound — 110 entries / 682 MB in one worktree.
///
/// Best effort throughout: a concurrent build whose entry is removed
/// mid-restore re-runs the prepass, because `restore_codegen_cache` copies
/// every output or reports failure — it never serves a partial set.
fn prune_codegen_cache(repo_root: &str, keep: &std::path::Path) {
    let base = codegen_cache_base(repo_root);
    if let Ok(versions) = std::fs::read_dir(&base) {
        for version in versions.flatten() {
            if version.file_name() != std::ffi::OsStr::new(CODEGEN_CACHE_VERSION) {
                let _ = std::fs::remove_dir_all(version.path());
            }
        }
    }
    let Ok(entries) = std::fs::read_dir(base.join(CODEGEN_CACHE_VERSION)) else {
        return;
    };
    let mut by_last_use: Vec<(std::time::SystemTime, std::path::PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        // `store_codegen_cache` stages into a dot-prefixed sibling before its
        // rename; leaving those alone keeps a concurrent store intact.
        if path == keep || !path.is_dir() || entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let last_use = std::fs::metadata(path.join(CODEGEN_CACHE_USED_MARKER))
            .or_else(|_| std::fs::metadata(&path))
            .and_then(|meta| meta.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        by_last_use.push((last_use, path));
    }
    by_last_use.sort_by_key(|item| std::cmp::Reverse(item.0));
    // `keep` is excluded above and occupies one of the retained slots.
    let retained = CODEGEN_CACHE_MAX_ENTRIES.saturating_sub(1);
    for (_, path) in by_last_use.into_iter().skip(retained) {
        let _ = std::fs::remove_dir_all(path);
    }
}

/// Compare two generated output sets file by file. True only when every
/// [`CODEGEN_OUTPUTS`] entry was read on both sides and is byte-identical: an
/// output that cannot be read on one side is an inconclusive comparison, and
/// an inconclusive comparison must not report as a clean one.
///
/// Reports through `cargo::warning`, which cargo surfaces for a workspace
/// crate's build script. A bare `println!` / `eprintln!` needs `-vv`, which is
/// how a build-script line documenting the cache restore went unread long
/// enough to be mistaken for the restore not happening.
///
/// Names the identical outputs as well as the differing ones. Which files hold
/// still localises the cause faster than which files move: `jitcodes.bin`
/// moving while `jitcodes_index.bin` holds says the entries changed while
/// their names and boundaries did not, which rules out the walk order and
/// points at what is written into each entry. A size change and an
/// equal-size change are different findings too — one moved a count, the
/// other only an order — so both lengths are reported, not just that they
/// differ.
///
/// `cross_process` excuses [`HOST_ADDRESSED_OUTPUTS`] from the verdict. They
/// are still reported, on their own line: a check that files a by-design
/// difference as a defect is one people learn to ignore, and then it reports
/// nothing at all.
fn codegen_outputs_match(
    label: &str,
    baseline: &std::path::Path,
    probe: &std::path::Path,
    cross_process: bool,
) -> bool {
    let mut identical: Vec<&str> = Vec::new();
    let mut differing: Vec<String> = Vec::new();
    let mut host_addressed: Vec<&str> = Vec::new();
    let mut unreadable: Vec<String> = Vec::new();
    for name in CODEGEN_OUTPUTS {
        match (
            std::fs::read(baseline.join(name)),
            std::fs::read(probe.join(name)),
        ) {
            (Ok(a), Ok(b)) if a == b => identical.push(name),
            (Ok(_), Ok(_)) if cross_process && HOST_ADDRESSED_OUTPUTS.contains(name) => {
                host_addressed.push(name);
            }
            (Ok(a), Ok(b)) => {
                let offset = a
                    .iter()
                    .zip(b.iter())
                    .position(|(x, y)| x != y)
                    .unwrap_or_else(|| a.len().min(b.len()));
                differing.push(format!(
                    "{name} ({} vs {} bytes, first difference at offset {offset})",
                    a.len(),
                    b.len()
                ));
            }
            (a, b) => unreadable.push(format!(
                "{name} (baseline {}, probe {})",
                if a.is_ok() { "read" } else { "unreadable" },
                if b.is_ok() { "read" } else { "unreadable" }
            )),
        }
    }
    if !host_addressed.is_empty() {
        eprintln!(
            "[pyre-jit-trace build.rs] codegen determinism: {} host-address table(s) differ from \
             {label} as expected, excluded from the verdict: {}",
            host_addressed.len(),
            host_addressed.join(" ")
        );
    }
    if differing.is_empty() && unreadable.is_empty() {
        eprintln!(
            "[pyre-jit-trace build.rs] codegen determinism: all {} reproducible outputs are \
             identical to {label}",
            identical.len()
        );
        return true;
    }
    for entry in &differing {
        println!("cargo::warning=codegen determinism: DIFFERS from {label}: {entry}");
    }
    for entry in &unreadable {
        println!("cargo::warning=codegen determinism: NOT COMPARED against {label}: {entry}");
    }
    println!(
        "cargo::warning=codegen determinism: {} of {} outputs unchanged from {label}: {}",
        identical.len(),
        CODEGEN_OUTPUTS.len(),
        identical.join(" ")
    );
    if !host_addressed.is_empty() {
        println!(
            "cargo::warning=codegen determinism: excluded as host addresses (expected to differ \
             across processes): {}",
            host_addressed.join(" ")
        );
    }
    false
}

fn restore_codegen_cache(cache_dir: &std::path::Path, out_dir: &str) -> bool {
    if !CODEGEN_OUTPUTS
        .iter()
        .all(|name| cache_dir.join(name).is_file())
    {
        return false;
    }
    for name in CODEGEN_OUTPUTS {
        let src = cache_dir.join(name);
        let dst = std::path::Path::new(out_dir).join(name);
        if let Err(e) = std::fs::copy(&src, &dst) {
            eprintln!(
                "[pyre-jit-trace build.rs] warning: cache restore failed for {}: {e}",
                src.display()
            );
            return false;
        }
    }
    true
}

fn store_codegen_cache(cache_dir: &std::path::Path, out_dir: &str) -> std::io::Result<()> {
    if CODEGEN_OUTPUTS
        .iter()
        .all(|name| cache_dir.join(name).is_file())
    {
        return Ok(());
    }
    let Some(parent) = cache_dir.parent() else {
        return Ok(());
    };
    std::fs::create_dir_all(parent)?;
    let tmp_dir = parent.join(format!(
        ".{}.tmp-{}",
        cache_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("cache"),
        std::process::id()
    ));
    if tmp_dir.exists() {
        std::fs::remove_dir_all(&tmp_dir)?;
    }
    std::fs::create_dir_all(&tmp_dir)?;
    for name in CODEGEN_OUTPUTS {
        let src = std::path::Path::new(out_dir).join(name);
        let dst = tmp_dir.join(name);
        std::fs::copy(src, dst)?;
    }
    match std::fs::rename(&tmp_dir, cache_dir) {
        Ok(()) => Ok(()),
        Err(e) if cache_dir.exists() => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            eprintln!("[pyre-jit-trace build.rs] cache already stored by another process: {e}");
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            Err(e)
        }
    }
}

fn codegen_cache_key(manifest_dir: &str, repo_root: &str, source_paths: &[String]) -> String {
    let mut h = CacheHasher::new();
    h.write_str(CODEGEN_CACHE_VERSION);
    for key in ["HOST", "TARGET", "PROFILE", "OPT_LEVEL"] {
        h.write_str(key);
        h.write_os(std::env::var_os(key));
    }
    let mut cargo_env: Vec<(String, String)> = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("CARGO_FEATURE_") || key.starts_with("CARGO_CFG_TARGET_")
        })
        .collect();
    cargo_env.sort();
    for (key, value) in cargo_env {
        h.write_str(&key);
        h.write_str(&value);
    }
    h.write_os(std::env::var_os("PYRE_MIR_FRONTEND_LLBC"));
    for key in LOWERING_GATE_ENV {
        h.write_str(key);
        h.write_os(std::env::var_os(key));
    }

    // The codegen output also depends on every crate linked into this
    // build-script binary — `majit-translate`'s own dependencies
    // (`majit-ir`, `majit-charon-reader`, `rustpython-compiler-core`, …)
    // and their serde wire formats — whose sources `majit-translate/src`
    // below does not cover. Without them the key would stay identical across
    // such a change and restore a stale snapshot (e.g. `*.bin` written under
    // an older `majit-ir` bincode layout).
    //
    // Hash their *sources* rather than the build-script executable's bytes.
    // The binary is not reproducible: recompiling it from unchanged sources —
    // which `cargo clean`, a fresh `CARGO_TARGET_DIR`, or a touched `build.rs`
    // all force — yields different bytes, so keying on them rekeyed the cache
    // on almost every build and made it serve only reruns that skipped the
    // recompile. Workspace sources plus `Cargo.lock` (external crate
    // versions) and the compiler's version string cover the same change
    // surface and are stable across recompiles.
    hash_file_content(&mut h, &std::path::Path::new(repo_root).join("Cargo.lock"));
    h.write_str(&rustc_version_string());
    for workspace_dir in ["majit", "pyre"] {
        let root = std::path::Path::new(repo_root).join(workspace_dir);
        let Ok(crates) = std::fs::read_dir(&root) else {
            continue;
        };
        let mut src_dirs: Vec<std::path::PathBuf> = crates
            .flatten()
            .map(|entry| entry.path().join("src"))
            .filter(|src| src.is_dir())
            .collect();
        src_dirs.sort();
        for src in src_dirs {
            hash_rs_dir_content(&mut h, &src);
        }
    }

    hash_file_content(&mut h, &std::path::Path::new(manifest_dir).join("build.rs"));
    hash_file_content(
        &mut h,
        &std::path::Path::new(manifest_dir).join("src/virtualizable_spec.rs"),
    );
    hash_file_content(
        &mut h,
        &std::path::Path::new(manifest_dir).join("src/call_spec.rs"),
    );

    for path in source_paths {
        hash_file_content(&mut h, std::path::Path::new(path));
    }
    hash_rs_dir_content(
        &mut h,
        &std::path::Path::new(repo_root).join("majit/majit-translate/src"),
    );
    hash_llbc_inputs(&mut h, repo_root);

    format!("{:016x}", h.finish())
}

/// `rustc -vV`, so a toolchain upgrade that changes codegen or a type layout
/// rekeys the cache. Falls back to a marker when the compiler cannot be run —
/// a missing string only makes the key coarser, never staler, because every
/// other input is still hashed.
fn rustc_version_string() -> String {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    std::process::Command::new(rustc)
        .arg("-vV")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_else(|| "rustc-version-unavailable".to_string())
}

fn hash_llbc_inputs(h: &mut CacheHasher, repo_root: &str) {
    // Hash the LLBC by content, not by (len, mtime) signature. The
    // analysis (`analyze_multiple_pipeline_with_modules`) derives every
    // generated artefact from these graph bodies, so a content change that
    // happens to preserve size and mtime — `git checkout`, a cache restore
    // that keeps timestamps, an in-place rewrite of equal length — must
    // still rekey the cache. A signature would let `restore_codegen_cache`
    // serve stale output and skip re-analysis. The `.ullbc` set is ~660 MB
    // (89 % of it `fun_decls`), so the read costs under a second — still
    // negligible next to the tens of seconds of analysis it gates.
    if let Some(paths) = std::env::var_os("PYRE_MIR_FRONTEND_LLBC") {
        for path in std::env::split_paths(&paths) {
            if !path.as_os_str().is_empty() {
                hash_file_content(h, &path);
            }
        }
        return;
    }
    // Same single source of truth as `emit_rerun_directives`: a crate missing
    // from this hash keys the codegen cache on a stale snapshot of it.
    for crate_name in LLBC_CRATES {
        hash_file_content(
            h,
            &std::path::Path::new(repo_root)
                .join("build")
                .join("llbc")
                .join(format!("{crate_name}.ullbc")),
        );
    }
    // Cross-target sidecars are build inputs too: they supply the target field
    // offsets `auto_discover_workspace_llbc_paths` merges in.  Without them in
    // the key, regenerating a sidecar (same Rust sources, same target — e.g.
    // after fixing the extraction flags) leaves the key unchanged, so
    // `restore_codegen_cache` would serve the descrs built from the old
    // offsets.  Empty on a native build.
    for sidecar in llbc_layout_sidecars() {
        hash_file_content(
            h,
            &std::path::Path::new(repo_root)
                .join("build")
                .join("llbc")
                .join(&sidecar),
        );
    }
}

fn hash_rs_dir_content(h: &mut CacheHasher, dir: &std::path::Path) {
    let mut paths = Vec::new();
    for entry in WalkDir::new(dir) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        paths.push(entry.path().to_path_buf());
    }
    paths.sort();
    for path in paths {
        hash_file_content(h, &path);
    }
}

fn hash_file_content(h: &mut CacheHasher, path: &std::path::Path) {
    h.write_path(path);
    let Ok(mut file) = std::fs::File::open(path) else {
        h.write_str("missing");
        return;
    };
    h.write_str("content");
    let mut buf = [0_u8; 64 * 1024];
    loop {
        match std::io::Read::read(&mut file, &mut buf) {
            Ok(0) => break,
            Ok(n) => h.write_bytes(&buf[..n]),
            Err(e) => {
                h.write_str("read-error");
                h.write_str(&e.to_string());
                break;
            }
        }
    }
}

/// Length-prefixed hashing wrapper over the std hasher.
///
/// `DefaultHasher` is fixed-key SipHash: deterministic within a given Rust
/// toolchain (no per-process seed), which is all the cache needs — a key
/// produced by one build matches the same build's stored entry. std does
/// not promise the algorithm is stable across Rust releases, so a toolchain
/// upgrade changes every key; that is fine here (a miss just regenerates,
/// and the build-script executable is already in the key, so a toolchain
/// bump rekeys regardless). Inputs are length-prefixed so adjacent fields
/// cannot run together: `("ab", "c")` and `("a", "bc")` hash differently.
struct CacheHasher(std::collections::hash_map::DefaultHasher);

impl CacheHasher {
    fn new() -> Self {
        Self(std::collections::hash_map::DefaultHasher::new())
    }

    fn finish(&self) -> u64 {
        std::hash::Hasher::finish(&self.0)
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        // Fixed-width length prefix, then the payload — the prefix frames
        // the byte run so concatenation stays unambiguous.
        std::hash::Hasher::write_u64(&mut self.0, bytes.len() as u64);
        std::hash::Hasher::write(&mut self.0, bytes);
    }

    fn write_str(&mut self, value: &str) {
        self.write_bytes(value.as_bytes());
    }

    fn write_os(&mut self, value: Option<std::ffi::OsString>) {
        match value {
            Some(value) => self.write_str(&value.to_string_lossy()),
            None => self.write_str("<unset>"),
        }
    }

    fn write_path(&mut self, path: &std::path::Path) {
        self.write_str(&path.to_string_lossy());
    }
}

/// Collect a single `.rs` file by absolute path, mirroring
/// `collect_rs_files`'s read-into-vecs convention.  Used to thread
/// `pyre-jit/src/eval.rs` (the portal canonical)
/// into the analysis without including the rest of pyre-jit's JIT
/// infrastructure (codewriter, assembler, regalloc, ...).
/// Crate-stripped module path for a source file at `path`.
///
/// Strips the crate root (`/.../<crate>/src/` prefix) and the `.rs`
/// suffix, then converts `/` to `::` for nested files.  Matches the
/// runtime `module_path!()` macro output after the leading crate
/// segment is dropped — both sides hash the same string so
/// `gc_cache._cache_size[LLType::Struct(path_hash(path))]` slots
/// align (PyPy descr.py:108-118 `cache[STRUCT]` identity).
///
/// Examples (input → output):
/// - `"pyre/pyre-object/src/intobject.rs"` → `"intobject"`
/// - `"pyre/pyre-interpreter/src/pyframe.rs"` → `"pyframe"`
/// - `"pyre/pyre-interpreter/src/foo/bar.rs"` → `"foo::bar"`
/// - `"pyre/pyre-interpreter/src/lib.rs"` → `""` (crate root, no qualifier)
///
/// Returns `""` when the path does not contain `/src/` — callers
/// outside the canonical layout (synthesized files, fixtures) keep
/// the simple-name registration.
fn module_path_from_source_file(path: &str) -> String {
    // Windows `WalkDir` yields native paths with `\` separators; the marker
    // search + `/lib` / `/mod` suffix strips + final `/` → `::` rewrite
    // below all assume forward slashes, so an unnormalised Windows path
    // falls into the `rfind` `None` branch and every source file ends up
    // with an empty `module_path`.  Empty module paths skip
    // `register_struct_origins` (`lib.rs:374-382`), which breaks
    // classdef-keyed method resolution downstream and silently drops
    // graphs from the analyzer — surfacing later as missing opcodes in
    // `pipeline.insns` (e.g. `setfield_vable_i/rid`).
    let normalized_path = path.replace('\\', "/");
    let path = normalized_path.as_str();
    let marker = "/src/";
    let Some(idx) = path.rfind(marker) else {
        return String::new();
    };
    let rest = &path[idx + marker.len()..];
    let stem = rest.strip_suffix(".rs").unwrap_or(rest);
    let normalized = stem
        .strip_suffix("/lib")
        .or_else(|| stem.strip_suffix("/mod"))
        .unwrap_or(stem);
    if normalized == "lib" || normalized == "mod" {
        return String::new();
    }
    normalized.replace('/', "::")
}

fn collect_single_file(path: &str, paths: &mut Vec<String>) {
    match std::fs::metadata(path) {
        Ok(_) => paths.push(path.to_string()),
        Err(e) => {
            eprintln!("[pyre-jit-trace build.rs] warning: cannot read {path}: {e}");
        }
    }
}

/// Collect all `.rs` files from a directory tree.
///
/// Sorts entries by path so the collected source order is stable
/// across platforms.  Without this, `WalkDir` yields entries in the
/// filesystem's native `readdir` order — APFS (macOS) and ext4
/// (Linux) and NTFS (Windows) return different sequences, which
/// causes the analyzer to encounter type/method definitions in a
/// different order and exposes platform-divergent classdef-less
/// SomeInstance failures (PR 91 CI: Ubuntu/Windows fail with
/// `SomeBuiltin.call(): no analyser registered for std.ptr.null_mut`
/// and `SomeInstance.getattr on classdef-less instance` while macOS
/// passes).  Stable lexicographic order makes the build reproducible
/// and lets one fix cover every platform.
fn collect_rs_files(dir: &str, paths: &mut Vec<String>) {
    for entry in WalkDir::new(dir).sort_by_file_name() {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        paths.push(entry.path().to_string_lossy().to_string());
    }
}

fn emit_rerun_if_changed_recursive(dir: &str) {
    for entry in WalkDir::new(dir) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        println!("cargo::rerun-if-changed={}", entry.path().display());
    }
}
