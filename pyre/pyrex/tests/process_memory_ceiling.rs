//! `PYRE_MAX_MEMORY` stops a process that grows without bound.
//!
//! Two programs allocate past 512 MiB: one through Python objects
//! (`bytearray`, the GC heap / `incminimark.py` `max_heap_size`) and one
//! through `bytes` repetition. Each must exit on its own with a stderr line
//! naming the limit and `PYRE_MAX_MEMORY`, and on macOS and Linux the peak
//! RSS must stay under the limit plus a margin for the binary, stacks and
//! allocator metadata that sit outside the capped heap.
//!
//! The loops are finite (1.5 GiB) so a ceiling that fails to engage stops
//! the test instead of running into the machine watchdog. Do not invoke Cargo
//! from here: the parent `cargo test` holds the target directory lock.

#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

const LIMIT: usize = 512 * 1024 * 1024;

/// Binary, thread stacks and mappings outside the reserved arena.
const RSS_MARGIN: usize = 256 * 1024 * 1024;

const OBJECTS: &str = "\
data = []
for _ in range(1536):
    data.append(bytearray(1 << 20))
print('completed')
";

const BYTES: &str = "\
data = []
for _ in range(1536):
    data.append(b'x' * (1 << 20))
print('completed')
";

/// Peak RSS in bytes, measured by the platform's `time(1)`: BSD `-l` prints
/// `<bytes>  maximum resident set size`, GNU `-v` prints
/// `Maximum resident set size (kbytes): <kib>`. `None` where neither exists
/// or `/usr/bin/time` is not installed.
fn run(program: &str, label: &str) -> (std::process::Output, Option<usize>) {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("memcap_{label}.py"));
    std::fs::write(&path, program).expect("write program");
    let mut rss_path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    rss_path.push(format!("memcap_{label}.time"));

    let has_time = std::path::Path::new("/usr/bin/time").exists();
    let time_flag = if !has_time {
        None
    } else if cfg!(target_os = "macos") {
        Some("-l")
    } else if cfg!(target_os = "linux") {
        Some("-v")
    } else {
        None
    };
    let mut cmd = match time_flag {
        Some(flag) => {
            let mut cmd = Command::new("/usr/bin/time");
            cmd.args([flag, "-o"]).arg(&rss_path).arg(PYRE);
            cmd
        }
        None => Command::new(PYRE),
    };
    let out = cmd
        .arg(&path)
        .env("PYRE_MAX_MEMORY", "512M")
        .env("PYTHONIOENCODING", "utf-8")
        .env_remove("PYPY_GC_MAX")
        .output()
        .expect("spawn pyre-dynasm");
    let peak = time_flag.map(|_| {
        let rss = std::fs::read_to_string(&rss_path).unwrap_or_default();
        rss.lines()
            .find_map(|line| {
                let line = line.trim();
                if let Some(kib) = line.strip_prefix("Maximum resident set size (kbytes):") {
                    return kib.trim().parse::<usize>().ok().map(|k| k * 1024);
                }
                let (n, rest) = line.split_once(' ')?;
                rest.contains("maximum resident set size")
                    .then(|| n.parse::<usize>().ok())?
            })
            .unwrap_or(0)
    });
    (out, peak)
}

fn assert_capped(label: &str, program: &str) {
    let (out, peak) = run(program, label);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("completed"),
        "{label}: grew past the ceiling\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("PYRE_MAX_MEMORY") && stderr.contains(&LIMIT.to_string()),
        "{label}: stderr did not name the limit\n{stderr}"
    );
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = out.status.signal() {
            assert_ne!(sig, 9, "{label}: killed by SIGKILL\n{stderr}");
        }
    }
    if let Some(peak) = peak {
        assert!(
            peak > 0 && peak < LIMIT + RSS_MARGIN,
            "{label}: peak RSS {peak} is not under {} (limit {LIMIT})",
            LIMIT + RSS_MARGIN
        );
        eprintln!("{label}: peak RSS {peak}");
    }
}

#[test]
fn bytearray_growth_stops_at_the_ceiling() {
    assert_capped("objects", OBJECTS);
}

#[test]
fn bytes_repeat_stops_at_the_ceiling() {
    assert_capped("bytes", BYTES);
}
