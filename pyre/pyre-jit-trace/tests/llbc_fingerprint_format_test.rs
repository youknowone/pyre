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
//! Adding the `external=` field turned one bare digest into two `key=value`
//! lines, and the consumer's bare-sha256 test
//! stopped matching.  The unit cases below would all have passed on that broken
//! code had they been written beforehand — they encode whatever shape their
//! author believed in.  **Only `real_driver_output_parses` could have caught it**,
//! because it asks the producer instead of assuming.

use pyre_jit_trace::llbc_fingerprint::{parse_fingerprint_stdout, stamp_field};

const HASH: &str = "d9b2992606c82b29c531f4f4d6a42e43808564620ab63d781eba135f9307c584";

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
    let parsed = parse_fingerprint_stdout(&stdout);

    assert!(
        parsed.is_some(),
        "the consumer could not parse the producer's actual output.\n\
         This is the producer/consumer drift this test exists to catch: the \
         driver's format moved and `parse_fingerprint_stdout` was not taught.\n\
         raw stdout ({} bytes): {:?}",
        stdout.len(),
        stdout,
    );
    let digest = parsed.unwrap();
    assert_eq!(digest.len(), 64, "digest is not a sha256: {digest:?}");
    assert!(
        digest.bytes().all(|b| b.is_ascii_hexdigit()),
        "digest is not hex: {digest:?}",
    );
}

#[test]
fn parses_the_current_two_field_output() {
    let stdout = format!("source={HASH}\nexternal=\n");
    assert_eq!(parse_fingerprint_stdout(&stdout).as_deref(), Some(HASH));
}

#[test]
fn field_order_does_not_matter() {
    let stdout = format!("external=\nsource={HASH}\n");
    assert_eq!(parse_fingerprint_stdout(&stdout).as_deref(), Some(HASH));
}

#[test]
fn an_unknown_trailing_field_is_ignored() {
    let stdout = format!("source={HASH}\nexternal=\nsomething_new=42\n");
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
    let stdout = format!("source={}\nexternal=\n", &HASH[..63]);
    assert_eq!(parse_fingerprint_stdout(&stdout), None);
}

#[test]
fn a_non_hex_digest_is_rejected() {
    let stdout = format!("source={}zz\nexternal=\n", &HASH[..62]);
    assert_eq!(parse_fingerprint_stdout(&stdout), None);
}

#[test]
fn empty_output_is_rejected() {
    assert_eq!(parse_fingerprint_stdout(""), None);
}

#[test]
fn stamp_field_reads_one_key() {
    let stamp = format!("crate=pyre-object\nsource={HASH}\nexternal=\n");
    assert_eq!(stamp_field(&stamp, "source=").as_deref(), Some(HASH));
    assert_eq!(stamp_field(&stamp, "external=").as_deref(), Some(""));
    assert_eq!(stamp_field(&stamp, "absent="), None);
}
