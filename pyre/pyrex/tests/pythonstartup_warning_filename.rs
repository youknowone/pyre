//! A `PYTHONSTARTUP` path with a non-UTF-8 byte names both the SyntaxWarning
//! and `co_filename` with the same fsdecoded spelling.

#![cfg(all(feature = "dynasm", unix))]

use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

#[test]
fn pythonstartup_syntaxwarning_filename_matches_file() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("startup-warn-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let name = std::ffi::OsStr::from_bytes(b"start\xff.py");
    let path = dir.join(name);
    let mut file = match std::fs::File::create(&path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("skip: cannot create a non-UTF-8 path ({error})");
            return;
        }
    };
    file.write_all(b"'\\q'\nprint('FILE', ascii(__file__))\nprint('CODE', ascii((lambda: 0).__code__.co_filename))\n")
        .unwrap();
    drop(file);

    let output = Command::new(PYRE)
        .args(["-i", "-q"])
        .env("PYTHONSTARTUP", &path)
        .env("PYTHONSAFEPATH", "1")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn pyre");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let file_line = stdout
        .lines()
        .find(|line| line.starts_with("FILE "))
        .unwrap_or_else(|| {
            panic!(
                "no FILE line\nstdout={stdout}\nstderr={}",
                String::from_utf8_lossy(&output.stderr)
            )
        });
    let code_line = stdout
        .lines()
        .find(|line| line.starts_with("CODE "))
        .expect("CODE line");
    assert_eq!(file_line["FILE ".len()..], code_line["CODE ".len()..]);
    assert!(
        !output
            .stderr
            .windows(3)
            .any(|bytes| bytes == [0xef, 0xbf, 0xbd]),
        "SyntaxWarning filename was lossy: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // The fsdecoded name carries U+DCFF; `sys.stderr` writes it through
    // `errors='backslashreplace'` as the ASCII text `\udcff`.
    let escaped = br"start\udcff.py";
    assert!(
        output
            .stderr
            .windows(escaped.len())
            .any(|bytes| bytes == escaped),
        "SyntaxWarning filename is not the fsdecoded path: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}
