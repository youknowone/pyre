//! Helpers shared by this crate's `#[test]` bodies.
//!
//! Nothing here is compiled into the library; the module exists so that two
//! tests in different files can share one synchronisation point, which a
//! helper private to one `mod tests` cannot provide.

use std::collections::HashMap;

static STRUCT_REGISTRY_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Publish `table` into the process-global name → `StructId` map and hold
/// every other registering test out until the returned guard drops.
///
/// `register_struct_ids` REPLACES the whole table (`majit-ir/src/descr.rs`,
/// `*guard = table`), which is right for production — the front end populates
/// it once per program, and merging would leak one program's names into the
/// next — and hostile to `cargo test`'s default parallelism, where a second
/// registering test wipes the first one's only entry while that test is still
/// running. The victim does not fail where it registered: its lookup silently
/// takes a different arm and returns a wrong answer.
///
/// Measured, not inferred: two registering tests, run as a pair with default
/// threads, failed 9 of 12 runs; in the full lib-test binary the same race
/// surfaced once in 6, because the scheduler rarely puts them adjacent. Both
/// pass in isolation and under `--test-threads=1`, so neither the isolated run
/// nor the serial run can see this.
///
/// Scope is one test BINARY, which is the whole racing population: other
/// crates' tests run in their own processes and cannot reach this table. That
/// is also why the lock lives here rather than in one file's `mod tests` — a
/// second file's writer would otherwise take a different lock and race anyway.
///
/// The lock is poison-tolerant. A test that panics mid-body would otherwise
/// convert one real failure into a cascade of unrelated ones.
///
/// Bind the result to a NAMED local (`let _registry = …`). A bare `let _ = …`
/// drops the guard at once and restores exactly the race this exists to
/// remove — while still compiling, and still passing in isolation.
#[must_use = "the returned guard holds the registry lock for the rest of \
              the test; dropping it immediately re-opens the race"]
pub(crate) fn register_struct_ids_serialized(
    table: HashMap<String, Option<majit_ir::descr::StructId>>,
) -> parking_lot::MutexGuard<'static, ()> {
    let guard = STRUCT_REGISTRY_TEST_LOCK.lock();
    majit_ir::descr::register_struct_ids(table);
    guard
}

/// Publish an origin table while excluding tests that replace either global
/// struct-name registry. Production publishes both together for one program;
/// tests must likewise prevent one table from changing under another test's
/// descriptor construction.
#[must_use = "the returned guard holds the registry lock for the rest of the test"]
pub(crate) fn register_struct_origins_serialized(
    table: HashMap<String, String>,
) -> parking_lot::MutexGuard<'static, ()> {
    let guard = STRUCT_REGISTRY_TEST_LOCK.lock();
    majit_ir::descr::register_struct_origins(table);
    guard
}
