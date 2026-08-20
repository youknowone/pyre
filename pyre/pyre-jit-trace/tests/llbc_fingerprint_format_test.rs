//! The LLBC `--fingerprint` format contract, tested against the real producer.
//!
//! `build.rs` asks `scripts/extract-llbc.py --fingerprint <crate>` what the
//! sources currently hash to and compares the answer with the stamp beside the
//! artefact.  Producer and consumer live in different languages and are joined
//! only by this format, so the interesting failure is not a wrong answer — it
//! is the consumer quietly failing to parse a format the producer legitimately
//! extended, after which every crate reports "not checked" and the build looks
//! exactly like success.
//!
//! Adding fingerprint fields turned one bare digest into multiple `key=value`
//! lines, and the consumer's bare-sha256 test
//! stopped matching.  The unit cases below would all have passed on that broken
//! code had they been written beforehand — they encode whatever shape their
//! author believed in.  **Only `real_driver_output_parses` could have caught it**,
//! because it asks the producer instead of assuming.

use pyre_jit_trace::llbc_fingerprint::{
    parse_fingerprint_fields, parse_fingerprint_stdout, platform_key, stamp_field,
};

const HASH: &str = "d9b2992606c82b29c531f4f4d6a42e43808564620ab63d781eba135f9307c584";
// Distinct from HASH so a parse that filled `closure` from the source line, or
// swapped the two, cannot pass. `fail_if_llbc_stale` splits stale from warning
// on exactly that pair, so the binding is what these tests have to pin.
const CLOSURE_HASH: &str = "1de0e0c0e50a19e0ea1eb03c8ed0da4b8f3c78dfd3a86d5f6f65f2d0d9f7c4a1";

/// Walk up from this crate until the extraction driver is found.
fn repo_root() -> Option<std::path::PathBuf> {
    let mut dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("scripts").join("extract-llbc.py").is_file() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

/// Walk up from this crate until the neutral extraction engine is found.
fn engine_root() -> Option<std::path::PathBuf> {
    let mut dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("scripts").join("llbc_extract.py").is_file() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

/// The pair test: the real driver's real output, parsed by the real consumer.
///
/// Ignored by default because it shells out to `python3` and the driver is not
/// present in every checkout; CI runs the ignored set, where an absent driver
/// is a hard failure rather than a silent skip. A test that quietly passes when
/// it could not run is the same defect this file exists to prevent.
#[test]
#[ignore = "shells out to python3 + scripts/extract-llbc.py; run in CI via --ignored"]
fn real_driver_output_parses() {
    let root = repo_root().expect(
        "scripts/extract-llbc.py not found above CARGO_MANIFEST_DIR — \
         this test was asked to run, so an absent driver is a failure, not a skip",
    );
    let out = std::process::Command::new("python3")
        .arg(root.join("scripts").join("extract-llbc.py"))
        .arg("--fingerprint")
        .arg("pyre-object")
        .current_dir(&root)
        .output()
        .expect("failed to spawn python3");

    assert!(
        out.status.success(),
        "driver exited {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );

    let stdout = String::from_utf8(out.stdout).expect("driver stdout is not UTF-8");
    let parsed = parse_fingerprint_fields(&stdout);

    assert!(
        parsed.is_some(),
        "the consumer could not parse the producer's actual output.\n\
         This is the producer/consumer drift this test exists to catch: the \
         driver's format moved and `parse_fingerprint_stdout` was not taught.\n\
         raw stdout ({} bytes): {:?}",
        stdout.len(),
        stdout,
    );
    let digest = parsed.unwrap().source;
    assert_eq!(digest.len(), 64, "digest is not a sha256: {digest:?}");
    assert!(
        digest.bytes().all(|b| b.is_ascii_hexdigit()),
        "digest is not hex: {digest:?}",
    );
}

#[test]
fn parses_the_current_three_field_output() {
    let stdout = format!("source={HASH}\nclosure={CLOSURE_HASH}\nexternal=\n");
    assert_eq!(parse_fingerprint_stdout(&stdout).as_deref(), Some(HASH));
    let fields = parse_fingerprint_fields(&stdout).expect("three-field output must parse");
    assert_eq!(fields.source, HASH);
    assert_eq!(fields.closure, CLOSURE_HASH);
    assert_eq!(fields.external, "");
}

#[test]
fn field_order_does_not_matter() {
    let stdout = format!("external=\nclosure={CLOSURE_HASH}\nsource={HASH}\n");
    assert_eq!(parse_fingerprint_stdout(&stdout).as_deref(), Some(HASH));
    let fields = parse_fingerprint_fields(&stdout).expect("reordered output must parse");
    assert_eq!(fields.source, HASH);
    assert_eq!(fields.closure, CLOSURE_HASH);
}

#[test]
fn an_unknown_trailing_field_is_ignored() {
    let stdout = format!("source={HASH}\nclosure={HASH}\nexternal=\nsomething_new=42\n");
    assert_eq!(parse_fingerprint_stdout(&stdout).as_deref(), Some(HASH));
}

/// No bare-hash fallback. Accepting the old single-value shape would let the
/// consumer keep silently honouring a format the producer no longer emits —
/// the exact mechanism that produced the defect. Driver and consumer version
/// together in one repository, so strict is the safe direction.
#[test]
fn a_bare_hash_is_rejected() {
    assert_eq!(parse_fingerprint_stdout(HASH), None);
}

#[test]
fn a_short_digest_is_rejected() {
    let stdout = format!("source={}\nclosure={HASH}\nexternal=\n", &HASH[..63]);
    assert_eq!(parse_fingerprint_stdout(&stdout), None);
}

#[test]
fn a_non_hex_digest_is_rejected() {
    let stdout = format!("source={}zz\nclosure={HASH}\nexternal=\n", &HASH[..62]);
    assert_eq!(parse_fingerprint_stdout(&stdout), None);
}

#[test]
fn empty_output_is_rejected() {
    assert_eq!(parse_fingerprint_stdout(""), None);
}

#[test]
fn output_without_closure_is_rejected() {
    let stdout = format!("source={HASH}\nexternal=\n");
    assert_eq!(parse_fingerprint_fields(&stdout), None);
}

#[test]
fn a_non_hash_closure_is_rejected() {
    let stdout = format!("source={HASH}\nclosure=unknown\nexternal=\n");
    assert_eq!(parse_fingerprint_fields(&stdout), None);
}

#[test]
fn stamp_field_reads_one_key() {
    let stamp = format!("crate=pyre-object\nsource={HASH}\nclosure={HASH}\nexternal=\n");
    assert_eq!(stamp_field(&stamp, "source=").as_deref(), Some(HASH));
    assert_eq!(stamp_field(&stamp, "closure=").as_deref(), Some(HASH));
    assert_eq!(stamp_field(&stamp, "external=").as_deref(), Some(""));
    assert_eq!(stamp_field(&stamp, "absent="), None);
}

/// The other pair test: `platform_key` against the producer's own table.
///
/// `platform=` is written by `platform_info` in `scripts/llbc_extract.py` and
/// read by `fail_if_llbc_stale` to refuse artefacts extracted for another
/// host.  The two spellings are hand-written in different languages, so the
/// failure that matters is not a wrong answer for some exotic host -- it is
/// the two tables disagreeing about the host doing the build, which turns the
/// check into either a permanent false alarm or a permanent no-op.
#[test]
#[ignore = "shells out to python3 + scripts/llbc_extract.py; run in CI via --ignored"]
fn platform_key_agrees_with_the_producer() {
    // Not `repo_root`: that stops at the nearest ancestor holding a
    // `scripts/extract-llbc.py`, and pyre's own driver is one -- the neutral
    // engine it delegates to lives a level further up.
    let root = engine_root().expect(
        "scripts/llbc_extract.py not found above CARGO_MANIFEST_DIR — \
         this test was asked to run, so an absent engine is a failure, not a skip",
    );
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            "import sys; sys.path.insert(0, 'scripts'); \
             from llbc_extract import platform_info; print(platform_info()[0])",
        )
        .current_dir(&root)
        .output()
        .expect("failed to spawn python3");

    assert!(
        out.status.success(),
        "producer exited {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );

    let theirs = String::from_utf8(out.stdout).expect("stdout is not UTF-8");
    let theirs = theirs.trim();
    // `std::env::consts` names the machine this test binary runs on, which is
    // the machine the producer just described -- the same pairing
    // `fail_if_llbc_stale` makes from its build script.
    let ours = platform_key(std::env::consts::OS, std::env::consts::ARCH);
    assert_eq!(
        ours,
        Some(theirs),
        "`platform_key` and `platform_info` disagree about this host. \
         The stamp records the producer's spelling, so a check built on \
         the consumer's would compare two names for the same machine."
    );
}

#[test]
fn every_host_the_producer_extracts_on_has_a_spelling() {
    // `platform_info`'s table, which is what a stamp can carry.
    assert_eq!(platform_key("macos", "aarch64"), Some("darwin-arm64"));
    assert_eq!(platform_key("macos", "x86_64"), Some("darwin-x86_64"));
    assert_eq!(platform_key("linux", "aarch64"), Some("linux-aarch64"));
    assert_eq!(platform_key("linux", "x86_64"), Some("linux-x86_64"));
    // Collapsed on the producing side, so it collapses here: spelling it
    // per-architecture would read every Windows stamp as a mismatch.
    assert_eq!(platform_key("windows", "x86_64"), Some("windows"));
    assert_eq!(platform_key("windows", "aarch64"), Some("windows"));
    // A host the producer refuses to extract on cannot have written a stamp,
    // and answering for it would compare against a name nothing writes.
    assert_eq!(platform_key("freebsd", "x86_64"), None);
}
