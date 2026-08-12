//! A triage document and the environment gates it owns must name the same set —
//! every gate read from a workspace member's Rust or from this project's own
//! Python has a live entry, and every live entry has a reader.
//!
//! **Namespaces.** A gate's prefix selects which document owns it, and both
//! brakes run once per namespace against that document. See `NAMESPACES` for
//! the current set; adding a namespace is one row there plus the document it
//! names. A namespace joins only when that document has been written and
//! audited, not merely to make this test pass.
//!
//! The charter (§3.6) says a gate is a staging area, not a home, and
//! `gate-triage.md` is the standing list of what to retire and when. That list
//! only works if a gate joins it when it is born: audited by hand, it drifted to
//! 63% empty — 66 of 105 live gates were absent — because nothing failed when a
//! new gate skipped it. This test is that failure.
//!
//! Adding a gate therefore costs one row. The row is cheap; the alternative is
//! another hand audit that goes stale the week after it lands.
//!
//! Both directions, because they rot differently and only one of them is loud. A
//! gate with no row is a gate nobody will retire. A row with no reader is the
//! quieter half: it survives the deletion of the code it describes, and the next
//! sweep spends its time re-deriving that the name is already gone —
//! `PYRE_FBW_REC_UNROLL` sat in the config-switch list from 2026-07-06, when
//! PR#374 deleted `fbw_unroll_bound()`, until this test went looking.
//!
//! **Scope.** Rust sources in workspace member crates, and this project's own
//! Python under `pyre/` and `scripts/`. The harness reads six gates that no Rust
//! file reads — `check.py`, `check_synthetic.py`, the `extra_tests` runners and
//! `scripts/llbc_extract.py` — and two of them, `PYRE_SYNTH_PYRE` and
//! `PYRE_SYNTH_PYTHON`, were undocumented until this scan reached them.
//!
//! Shell and YAML are not scanned. Every `PYRE_*` in them is *set* for a child
//! process whose reader is Rust or Python, so the read side is already covered
//! here; the one remaining `PYRE_*` name outside both languages,
//! `PYRE_CLASS_DESCRIPTORS`, is a linkme distributed slice and not an env var at
//! all (`gate-triage.md` §2).
//!
//! **What this file cannot see, stated so it is not mistaken for coverage.** A
//! gate named only in a comment is invisible here by construction — `READ_FORMS`
//! matches reads, and the fixture below pins `PYRE_MENTIONED_IN_A_COMMENT` as
//! *not* a gate. So a doc-comment citing a gate that no longer exists, or never
//! did, passes both brakes. That failure is real and separately tracked; it is
//! not what this test is for, and widening the prefix does not reach it.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // <root>/pyre/pyrex/ -> <root>
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("pyrex manifest sits two levels below the repo root")
        .to_path_buf()
}

/// Workspace member directories, read from the root `Cargo.toml`.
///
/// Deriving the search roots from the member list rather than walking the whole
/// repository is what keeps untracked scratch copies of source files out: a
/// `.rs` file only reaches a compiler if it belongs to a member crate, so only
/// those can hold a live gate. It also needs no maintenance — a new crate has to
/// join `members` to build at all.
fn workspace_member_dirs(root: &Path) -> Vec<PathBuf> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("read root Cargo.toml");
    // Anchored at line start: `default-members = [` sits above `members = [`
    // and an unanchored search finds that one instead.
    let after = manifest
        .split_once("\nmembers = [")
        .expect("root Cargo.toml has a members list")
        .1;
    let list = after.split_once(']').expect("members list is closed").0;
    let dirs: Vec<PathBuf> = list
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let line = line.strip_prefix('"')?;
            Some(root.join(line.split_once('"')?.0))
        })
        .collect();
    assert!(
        dirs.len() > 10,
        "parsed only {} workspace members — the Cargo.toml parse is wrong",
        dirs.len()
    );
    dirs
}

/// Where this project's own Python lives.
///
/// An allow-list rather than a tree walk, for two reasons the Rust side does not
/// have. The repository vendors the CPython and PyPy sources — `lib-python/`,
/// `pypy/`, `rpython/`, `lib_pypy/`, some 4600 `.py` files that are upstream's
/// and not ours — and untracked scratch directories sit at the repository root.
/// Neither is under these two.
const PYTHON_ROOTS: [&str; 2] = ["pyre", "scripts"];

fn collect_sources(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // A member crate's own `target/` from a standalone build.
            if entry.file_name() != "target" {
                collect_sources(&path, ext, out);
            }
        } else if path.extension().is_some_and(|e| e == ext) {
            out.push(path);
        }
    }
}

/// The function names that read an environment variable in this tree.
///
/// `env::var` (with its `_os` suffix) is the common one, but a search for it
/// alone misses the host seam: `host_os::var` on the sandbox path and
/// `host_seam::ops::getenv`, which takes a *byte* string. `PYRE_STDLIB` is read
/// through both seam forms in `pyre-interpreter/src/importing.rs` and stays
/// visible to a `std::env` search only because unrelated `env::var` readers
/// exist in `pyre-wasm-runner`. A wasm- or sandbox-only gate would have no such
/// cover and would bypass the brake entirely.
///
/// `environ.get` and `getenv` are the Python side. Only these two, because they
/// read unambiguously: the harness also writes gates into a child's environment
/// (`env["PYRE_PROBE"] = …`, `env.pop(…)`), and a subscript form cannot be told
/// from a read without parsing. Writing a gate for a child is not owning it —
/// the child's read is what this file asks about, and that read is in Rust.
///
/// A form belonging to the other language costs nothing: `env::var(` cannot
/// occur in Python, nor `environ.get(` in Rust.
const READ_FORMS: [&str; 4] = ["env::var", "host_os::var", "getenv", "environ.get"];

/// Cargo's build-script declaration that the build depends on a gate:
/// `println!("cargo::rerun-if-env-changed=NAME")`.
///
/// Counted as a read, and it has to be, because a build script can reach the
/// environment through an indirection no textual scan can follow.
/// `pyre-jit-trace/build.rs:490` binds `std::env::var_os` to a **closure** and
/// passes the gate name to that:
///
/// ```ignore
/// let env_switch = |name: &str| std::env::var_os(name).map(…);
/// let strict_var = env_switch("PYRE_LLBC_STRICT");
/// ```
///
/// so `PYRE_LLBC_STRICT` and `PYRE_LLBC_SKIP_FINGERPRINT_CHECK` have no
/// `env::var("NAME")` anywhere in the tree. Without this form both read as
/// retired, and the second brake tells you to move two **live** gates into a
/// retirement section — destroying true documentation to fix a false reading.
///
/// This does not relax the "a mention is not a read" rule the read forms exist
/// to enforce. A stale `rerun-if-env-changed` is not inert the way a stale
/// comment is: it is live build configuration deciding when Cargo reruns the
/// script, so it cannot rot unobserved the way `PYRE_FBW_REC_UNROLL` rotted
/// inside a document for a month.
const CARGO_ENV_DEP: &str = "rerun-if-env-changed=";

/// A gate namespace: the prefix its names carry, and the document that owns them.
///
/// A prefix assigns ownership, and both brakes below run once per namespace
/// against that namespace's own document. `pyre-wasm-runner/src/main.rs:140`
/// already filters guest environment names on `PYRE_` **or** `MAJIT_`, so two
/// prefixes is an existing convention in this tree rather than a new one.
///
/// **Why a namespace owns a document rather than sharing one.** `majit/**` is
/// consumed by `cel-jit` and `wasmi-majit` by rev, not only by pyre. Recording
/// majit's gates in a pyre document would put majit's retirement charter behind
/// a dependency those consumers do not have, and would stale the pyre document
/// every time majit adds a gate — visible only from pyre's test suite.
struct GateNamespace {
    prefix: &'static str,
    doc: &'static str,
}

/// WARNING: **Adding a row here is a data change: it arms both brakes over a whole
/// namespace at once, and they fail with one entry per undocumented name.**
///
/// `MAJIT_` was added in "gate-triage: catalog the MAJIT_* gates and arm the
/// brake's second namespace" (`1d670fa7c9c` as of 2026-08-11 — branch-local,
/// so re-derive from the subject), together with the `majit/gate-triage.md`
/// it names — the two cannot land separately in either order without a red.
/// Every function below is parameterized over the namespace so that adoption is
/// exactly this one row plus the document; see
/// `both_brakes_are_parameterized_over_the_namespace` for the proof that a
/// second namespace works, run against a synthetic document.
///
/// WARNING: Neither brake reads the documents' contents — they compare NAMES. A row
/// whose retirement condition is `UNRECORDED` (or empty) satisfies them exactly
/// as well as a filled one. `UNRECORDED` rows remain common in
/// `majit/gate-triage.md`; green here is evidence that every gate is *listed*,
/// and no evidence that anything is *described*. That hole is deliberate: a
/// catalog of plausible guesses would satisfy the brakes permanently while
/// describing nothing, which is worse than the absence, because the absence is
/// legible and a plausible sentence is not.
const NAMESPACES: [GateNamespace; 2] = [
    GateNamespace {
        prefix: "PYRE_",
        doc: "pyre/gate-triage.md",
    },
    GateNamespace {
        prefix: "MAJIT_",
        doc: "majit/gate-triage.md",
    },
];

/// The namespace owning this gate name, if any.
fn namespace_of(name: &str) -> Option<&'static GateNamespace> {
    NAMESPACES.iter().find(|ns| name.starts_with(ns.prefix))
}

/// Gate names this text reads from the environment.
///
/// Matches the read forms above rather than every mention of the name, so a gate
/// discussed in a comment or held in a Rust const does not count as live. That
/// is the same distinction `gate-triage.md` §2 draws by hand.
fn gates_read_by(text: &str) -> Vec<&str> {
    let mut found = Vec::new();
    for form in READ_FORMS {
        for (at, _) in text.match_indices(form) {
            let rest = &text[at + form.len()..];
            // `env::var_os` is the same read wearing a suffix.
            let rest = rest.strip_prefix("_os").unwrap_or(rest);
            let Some(rest) = rest.strip_prefix('(') else {
                continue;
            };
            let rest = rest.trim_start();
            // `host_seam::ops::getenv` names its gate with a byte string.
            let rest = rest.strip_prefix('b').unwrap_or(rest);
            let Some(rest) = rest.strip_prefix('"') else {
                continue;
            };
            let Some(end) = rest.find('"') else { continue };
            let name = &rest[..end];
            if namespace_of(name).is_some() {
                found.push(name);
            }
        }
    }
    // The build-script dependency declaration, which carries its name bare
    // rather than as a quoted call argument.
    for (at, _) in text.match_indices(CARGO_ENV_DEP) {
        let rest = &text[at + CARGO_ENV_DEP.len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .unwrap_or(rest.len());
        let name = &rest[..end];
        if namespace_of(name).is_some() {
            found.push(name);
        }
    }
    found
}

#[test]
fn gates_read_by_matches_the_read_forms_and_nothing_else() {
    let sample = r#"
        std::env::var("PYRE_A").is_ok();
        env::var_os("PYRE_B").is_none();
        std::env::var(
            "PYRE_C",
        );
        host_os::var("PYRE_D").ok();
        crate::host_seam::ops::getenv(b"PYRE_E");
        os.environ.get("PYRE_F")
        os.getenv("PYRE_G")
        println!("cargo::rerun-if-env-changed=PYRE_H");
        # a gate written into a child's environment is not a read of ours
        env["PYRE_NOT_A_READ_EITHER"] = "1"
        // PYRE_MENTIONED_IN_A_COMMENT
        const PYRE_CONST: &str = "PYRE_NOT_A_READ";
        other::var("PYRE_NOT_ENV");
        env::var("HOME");
    "#;
    let mut got = gates_read_by(sample);
    // Sorted: the scan groups by read form, so the order carries no meaning.
    got.sort_unstable();
    assert_eq!(
        got,
        vec![
            "PYRE_A", "PYRE_B", "PYRE_C", "PYRE_D", "PYRE_E", "PYRE_F", "PYRE_G", "PYRE_H"
        ]
    );
}

/// Does this `##` heading introduce a section that records history?
///
/// §1/§1b/§1c list gates whose readers were deleted, §2 lists names that were
/// never env vars, and §3 lists names with no read site. A name there is the
/// record of a gate that is *gone*, so it must not satisfy the brake — otherwise
/// re-introducing a reader for `PYRE_SINGLE_PASS` lands green on the strength of
/// its own retirement row, with neither its new polarity nor a retirement plan
/// written down.
fn is_history_heading(heading: &str) -> bool {
    let lower = heading.to_ascii_lowercase();
    // "retired", not "retire": §4 is a *live* section whose heading says when
    // its gates will go ("retire when the epic closes").
    lower.contains("retired") || lower.contains("dead") || lower.contains("not gates")
}

/// The names carrying `prefix` that this line spells out, in the order it
/// spells them.
///
/// Tokenized rather than substring-searched: `contains("PYRE_A")` is satisfied
/// by a documented `PYRE_ANCHOR_STRICT`, so a new gate whose name is a prefix of
/// a listed one would slip through the brake unnoticed.
///
/// The same tokenization keeps the two namespaces apart where they overlap in
/// text: `PYRE_MAJIT_STATS_ANCESTOR` contains `MAJIT_`, and the preceding-
/// character guard is what stops it being read as a majit gate.
fn gate_names_in<'a>(line: &'a str, prefix: &str) -> Vec<&'a str> {
    let mut names = Vec::new();
    for (at, _) in line.match_indices(prefix) {
        // A name preceded by a name character is the tail of a longer token.
        if line[..at]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            continue;
        }
        let end = line[at..]
            .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .map_or(line.len(), |off| at + off);
        names.push(&line[at..end]);
    }
    names
}

/// Every `PYRE_*` name the triage document lists as a live gate.
///
/// Scoped to the live sections for the reason in `is_history_heading` — `###`
/// subsections inherit their `##` parent, which is what keeps §6a–§6c live under
/// §6 — and then filtered by what the document records about each name.
///
/// **Retirement is a property of the gate, not of the line.** A retirement line
/// retires its *subject*, and a name stays retired wherever else it is written:
/// §5's table records `PYRE_CARRIER_EXC_RESUME` retired with its reader deleted,
/// while §1e's prose — a live section — discusses the seam that gate used to
/// guard and names it three more times. Counting those mentions as documentation
/// made a gate with no reader look live.
///
/// **Subject, not mention**: only the *first* name on a retirement line is
/// retired by it. §1c's row for `PYRE_AUTHORITATIVE` ends "`PYRE_PROBE_AUTHORITATIVE`
/// is separate and remains live" — the whole point of that clause is that the
/// second name is not the one being retired.
///
/// "retired", not "retire": §4 is a live section whose gates are the ones that
/// "retire when the epic closes".
///
/// A line that *negates* the word is refused rather than read either way — see
/// `negates_retirement`.
fn gates_documented_in<'a>(triage: &'a str, prefix: &str) -> BTreeSet<&'a str> {
    let mut live_names: BTreeSet<&str> = BTreeSet::new();
    let mut retired_subjects: BTreeSet<&str> = BTreeSet::new();
    let mut live = true;
    for (at, line) in triage.lines().enumerate() {
        if let Some(heading) = line.strip_prefix("## ") {
            live = !is_history_heading(heading);
        }
        let names = gate_names_in(line, prefix);
        let lower = line.to_ascii_lowercase();
        if lower.contains("retired") {
            if let Some(&subject) = names.first() {
                assert!(
                    !negates_retirement(&lower),
                    "line {} of the {prefix}* triage document names {subject} beside a \
                     negated `retired`, so the scan cannot tell whether the line records \
                     a retirement:\n  {line}\n\n\
                     The scan reads `retired` as a verdict on the line's first name and \
                     would drop {subject} from the live set wherever else the document \
                     writes it — which a line saying it is *not* retired does not ask \
                     for. Either state the retirement plainly, or write the note without \
                     the word.",
                    at + 1
                );
                retired_subjects.insert(subject);
            }
        } else if live {
            live_names.extend(names);
        }
    }
    live_names.retain(|name| !retired_subjects.contains(name));
    live_names
}

/// Does this line say a gate is *not* retired?
///
/// The scan cannot recover intent from prose: it reads `retired` anywhere on a
/// line as a verdict on that line's first gate name, so a negation makes it
/// conclude the opposite of what the line says, and the gate is dropped from the
/// live set everywhere. That drop is silent in the direction that matters —
/// `every_live_triage_entry_still_has_a_reader` simply stops checking a name it
/// no longer sees, so a documented gate whose reader is gone passes.
///
/// Refusing is not the same as guessing the other way. A negation is where the
/// two readings are furthest apart and the parse has no way to choose, so it
/// says so and names the line.
///
/// Scoped to lines that would actually drop a name: the phrasing already appears
/// in `pyre/gate-triage.md`'s §1b, as the "Deferred, not retired" heading over a
/// list, and that line spells no gate name — nothing is dropped, so nothing is
/// refused. When this landed, neither document had a line satisfying both halves.
fn negates_retirement(lowercased_line: &str) -> bool {
    ["not retired", "never retired", "not yet retired"]
        .iter()
        .any(|negation| lowercased_line.contains(negation))
}

/// A documented token that is a wildcard's stem rather than a gate.
///
/// This file writes `PYRE_*` and `PYRE_FBW_*` when it means a family, and the
/// token scan stops at the `*`, leaving a name ending in `_`. No gate is named
/// that way, so the trailing underscore is the whole test.
fn is_wildcard_stem(name: &str) -> bool {
    name.ends_with('_')
}

/// Every source file a gate can be read from.
fn source_files(root: &Path) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    for member in workspace_member_dirs(root) {
        collect_sources(&member, "rs", &mut sources);
    }
    assert!(
        sources.len() > 100,
        "found only {} .rs files across the workspace members — the walk is not \
         reaching the tree",
        sources.len()
    );

    let rust_files = sources.len();
    for py_root in PYTHON_ROOTS {
        let dir = root.join(py_root);
        assert!(
            dir.is_dir(),
            "PYTHON_ROOTS names {py_root}, which is not a directory under {}",
            root.display()
        );
        collect_sources(&dir, "py", &mut sources);
    }
    assert!(
        sources.len() - rust_files > 100,
        "found only {} .py files under {PYTHON_ROOTS:?} — the walk is not \
         reaching the harness",
        sources.len() - rust_files
    );
    sources
}

/// `(gate name, repo-relative file)` for every environment read in the tree.
fn read_sites(root: &Path) -> BTreeSet<(String, String)> {
    // This file's own fixture spells out `env::var("PYRE_A")` and friends, which
    // are test data rather than gates.
    let self_path = root.join(file!());
    let mut sites = BTreeSet::new();
    for path in source_files(root) {
        if path == self_path {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for name in gates_read_by(&text) {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            sites.insert((name.to_string(), rel.display().to_string()));
        }
    }
    sites
}

/// The triage document owning `ns`, read from disk.
fn triage_text(root: &Path, ns: &GateNamespace) -> String {
    let path = root.join(ns.doc);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn namespace(prefix: &str) -> &'static GateNamespace {
    NAMESPACES
        .iter()
        .find(|ns| ns.prefix == prefix)
        .expect("NAMESPACES names this prefix")
}

/// The live/history split in `gate-triage.md` is load-bearing for both brakes,
/// and a heading reword would otherwise change what they accept without failing.
/// Anchored on each side: a retired name must not count, a listed live name must.
#[test]
fn pyre_triage_live_history_split_holds_at_four_anchors() {
    let root = repo_root();
    let ns = namespace("PYRE_");
    // Bound, not inlined: the returned set borrows the document text.
    let triage = triage_text(&root, ns);
    let documented = gates_documented_in(&triage, ns.prefix);

    assert!(
        !documented.contains("PYRE_SINGLE_PASS"),
        "PYRE_SINGLE_PASS is named only in a retirement section and has no read \
         site, so it must not count as documented — is_history_heading no longer \
         matches gate-triage.md's headings"
    );
    assert!(
        !documented.contains("PYRE_FBW_VABLE_SCALAR_CA"),
        "PYRE_FBW_VABLE_SCALAR_CA is marked RETIRED inside §1d, a section whose \
         heading reads live — the per-row retirement check is no longer catching it"
    );
    assert!(
        !documented.contains("PYRE_CARRIER_EXC_RESUME"),
        "PYRE_CARRIER_EXC_RESUME is recorded retired in §5 and its reader is gone, \
         but §1e's live prose names it three times — a retired subject is retired \
         wherever else it is written, and this check is what enforces that"
    );
    assert!(
        documented.contains("PYRE_JD1"),
        "PYRE_JD1 is listed live in §6a but did not count as documented — \
         is_history_heading is excluding a live section"
    );
}

#[test]
fn every_live_gate_has_a_triage_entry() {
    let root = repo_root();
    let sites = read_sites(&root);

    for ns in &NAMESPACES {
        let triage = triage_text(&root, ns);
        let documented = gates_documented_in(&triage, ns.prefix);
        let missing: Vec<&(String, String)> = sites
            .iter()
            .filter(|(name, _)| name.starts_with(ns.prefix) && !documented.contains(name.as_str()))
            .collect();

        assert!(
            missing.is_empty(),
            "{} {}* gate(s) are read from the environment but have no entry in \
             {}:\n{}\n\n\
             Add a row for each: what it gates, its default polarity, and — for a \
             default-ON experiment — the epic whose close retires it.",
            missing.len(),
            ns.prefix,
            ns.doc,
            missing
                .iter()
                .map(|(name, file)| format!("  {name}  ({file})"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

#[test]
fn every_live_triage_entry_still_has_a_reader() {
    let root = repo_root();
    let read: BTreeSet<String> = read_sites(&root)
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    for ns in &NAMESPACES {
        let triage = triage_text(&root, ns);
        let stale: Vec<&str> = gates_documented_in(&triage, ns.prefix)
            .into_iter()
            .filter(|name| !is_wildcard_stem(name) && !read.contains(*name))
            .collect();

        assert!(
            stale.is_empty(),
            "{} name(s) are listed live in {} but nothing in the tree reads \
             them:\n{}\n\n\
             Deleting the row loses why the gate existed. Move each to a retirement \
             section (§1c) naming the change that removed its reader — that is what \
             stops the next reader of this name from passing on an obituary.",
            stale.len(),
            ns.doc,
            stale
                .iter()
                .map(|name| format!("  {name}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

/// Both brakes must actually run per namespace, against that namespace's own
/// document — not against the first one, and not against a union of all of them.
///
/// This is the whole content of the generalization, and the live brakes cannot
/// demonstrate it. `every_live_gate_has_a_triage_entry` and
/// `every_live_triage_entry_still_has_a_reader` do already call per namespace,
/// but a union of documents that are each individually correct is still correct
/// for every name in either — so a union bug is invisible against the committed
/// documents whatever `NAMESPACES` holds. Making it visible needs each side
/// wrong in the specific way a union would hide, which this test cannot do to a
/// real triage document without editing it. So the proof is run here on
/// synthetic text: a two-namespace world where each side is wrong in the way the
/// union would hide.
///
/// `EX_ELSEWHERE` is an `EX_` gate written **only** in the `SYN_` document. It
/// must not count as documented for `EX_`, and the positive control on the line
/// above is what makes that absence mean something: the same call against the
/// document that does spell it finds it. Without the control, a scan that had
/// stopped working entirely would pass this test.
#[test]
fn both_brakes_are_parameterized_over_the_namespace() {
    let ex_doc = "## §1 live\n`EX_KNOWN` gates a thing.\n";
    let syn_doc = "## §1 live\n`SYN_KNOWN` gates a thing.\n`EX_ELSEWHERE` is named here.\n";

    assert!(
        gates_documented_in(ex_doc, "EX_").contains("EX_KNOWN"),
        "prefix scan missed its own name"
    );
    assert!(
        gates_documented_in(syn_doc, "SYN_").contains("SYN_KNOWN"),
        "prefix scan missed its own name"
    );

    // Positive control: the name is findable by this scan, in the document that
    // spells it.
    assert!(
        gates_documented_in(syn_doc, "EX_").contains("EX_ELSEWHERE"),
        "the control failed — the scan cannot find EX_ELSEWHERE anywhere, so the \
         assertion below would pass for the wrong reason"
    );
    // The check: it is not documented for `EX_`, because `EX_`'s document does
    // not spell it. A scan over the union of all documents would pass the
    // control and fail here — which is the failure that makes one shared
    // document unsafe, and the reason `MAJIT_` cannot simply be pointed at
    // `pyre/gate-triage.md`.
    assert!(
        !gates_documented_in(ex_doc, "EX_").contains("EX_ELSEWHERE"),
        "a name absent from its own namespace's document counted as documented — \
         the lookup is not per document"
    );

    // Retirement is per document too: the same name may be live in one and
    // retired in the other, and each document governs only its own namespace.
    let retired = gates_documented_in("## §1 live\n`EX_KNOWN` retired 2026-01-01.\n", "EX_");
    assert!(
        !retired.contains("EX_KNOWN"),
        "a retirement line in the namespace's own document did not retire its subject"
    );
}

/// `PYRE_MAJIT_STATS_ANCESTOR` and friends spell `MAJIT_` inside a `PYRE_` name.
///
/// The preceding-character guard in `gate_names_in` is the only thing keeping
/// them out of a future `MAJIT_` namespace, and it is one `continue` in a loop —
/// cheap to lose in an edit, and its loss would silently move four documented
/// pyre gates into majit's ledger.
#[test]
fn a_prefix_appearing_inside_a_longer_name_is_not_a_gate_of_that_namespace() {
    let line = "`PYRE_MAJIT_STATS_ANCESTOR`, `PYRE_MAJIT_STATS_ROOT_ONLY`, `MAJIT_LOG`";
    assert_eq!(
        gate_names_in(line, "MAJIT_"),
        vec!["MAJIT_LOG"],
        "a `MAJIT_` embedded in a `PYRE_MAJIT_*` name was read as a majit gate"
    );
    assert_eq!(
        gate_names_in(line, "PYRE_"),
        vec!["PYRE_MAJIT_STATS_ANCESTOR", "PYRE_MAJIT_STATS_ROOT_ONLY"]
    );
}
