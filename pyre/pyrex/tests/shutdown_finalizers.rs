//! Finalization ordering can keep a referent alive across several collections.
//! Module teardown must not omit its pre-clear collection just because the
//! detached import cache contained only ordinary modules.

// `test.support` imports `unicodedata`, which only `pyre-module` provides.
#![cfg(all(feature = "dynasm", feature = "pyre-module"))]

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn pyre_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pyre-dynasm"));
    for flag in [
        "MAJIT_STATS",
        "MAJIT_LOG",
        "PYRE_MC_DIAG",
        "PYRE_GC_DIAG",
        "PYRE_FBW_SPEC_CENSUS",
        "PYRE_CELL_CENSUS",
        "PYRE_FBW_DEPTH_CENSUS",
    ] {
        command.env_remove(flag);
    }
    command
}

#[test]
fn chained_finalizers_run_before_module_globals_are_cleared() {
    // The two Link finalizers defer Last to the collection between detaching
    // sys.modules and clearing module dictionaries. No non-module import-cache
    // entry is removed, so the former dropped_unlisted condition skipped it.
    let program = r#"
import sys
import types

module = types.ModuleType('shutdown_finalizer_chain')
sys.modules[module.__name__] = module
exec('''
class State:
    value = "globals alive"

class Last:
    def __del__(self):
        print(State.value)

class Link:
    def __init__(self, next):
        self.next = next

    def __del__(self):
        pass

Link(Link(Last()))
''', module.__dict__)
"#;
    let output = pyre_command()
        .args(["-c", program])
        .output()
        .expect("run the shutdown finalizer chain");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert_eq!(stdout.trim(), "globals alive", "{stderr}");
    assert!(stderr.is_empty(), "{stderr}");
}

#[test]
fn mutually_referencing_modules_keep_globals_during_finalization() {
    // The vendored test_module finalization fixture exercises module cycles,
    // private globals, peers, imported functions and builtins in __del__.
    let output = pyre_command()
        .args(["-c", "from test.test_module import final_a"])
        .output()
        .expect("run the module finalization fixture");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(stderr.is_empty(), "{stdout}\n{stderr}");
    assert_eq!(stdout.lines().count(), 16, "{stdout}");
    let mut lines: Vec<_> = stdout.lines().collect();
    lines.sort_unstable();
    lines.dedup();
    assert_eq!(
        lines,
        [
            "final_a.x = a",
            "final_b.x = b",
            "len = len",
            "shutil.rmtree = rmtree",
            "x = a",
            "x = b"
        ]
    );
}

#[test]
fn new_finalizable_cycles_do_not_extend_shutdown_indefinitely() {
    let mut child = pyre_command()
        .args([
            "-c",
            r#"
class Reproducer:
    def __init__(self):
        self.cycle = self
    def __del__(self):
        type(self)()
Reproducer()
print("body done", flush=True)
"#,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start self-reproducing finalizer");
    let deadline = Instant::now() + Duration::from_secs(30);
    while child.try_wait().expect("query child status").is_none() {
        if Instant::now() >= deadline {
            child.kill().expect("stop hung finalization");
            let output = child.wait_with_output().expect("reap hung child");
            panic!(
                "shutdown did not finish: stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let output = child.wait_with_output().expect("read completed child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert_eq!(stdout.trim(), "body done");
    assert!(stderr.is_empty(), "{stderr}");
}

#[test]
fn command_startup_sees_sys_path_and_reuses_import_name() {
    let program = r#"
import sys
assert sys.path
by_name = {f.__name__: f for f in sys.meta_path}
assert 'BuiltinImporter' in by_name and 'FrozenImporter' in by_name and 'PathFinder' in by_name, list(by_name)
assert by_name['BuiltinImporter'].find_spec('sys') is not None
assert by_name['FrozenImporter'].find_spec('token') is None
assert by_name['PathFinder'].find_spec('token') is not None
assert by_name['PathFinder'].find_spec('not_a_real_module_zz') is None
assert list.append.__qualname__ == 'list.append'
boot = sys.modules['_frozen_importlib']
seen = []
real = boot.__import__
def wrap(name, globals=None, locals=None, fromlist=(), level=0):
    seen.append(name)
    return real(name, globals, locals, fromlist, level)
boot.__import__ = wrap
def f():
    import token
name = f.__code__.co_names[f.__code__.co_names.index('token')]
f()
assert seen and seen[0] is name, (seen, name)
"#;
    let output = pyre_command()
        .args(["-S", "-c", program])
        .output()
        .expect("run command startup sys.path / import-name script");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
}

#[test]
fn user_del_on_garbage_runs_at_shutdown() {
    let path = std::env::temp_dir().join(format!(
        "pyre-shutdown-del-{}-{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let _ = std::fs::remove_file(&path);
    let source = format!(
        "class A:\n    def __del__(self):\n        open({:?}, 'wb').write(b'ran')\nA()\n",
        path
    );
    let output = pyre_command()
        .args(["-c", &source])
        .output()
        .expect("run user __del__ shutdown script");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let _ = std::fs::remove_file(&path);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert_eq!(text, "ran");
}
