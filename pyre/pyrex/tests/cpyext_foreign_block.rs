//! A block the object allocator never handed out belongs to its extension.
//!
//! cffi allocates every `__CDataOwn` with plain `malloc` and frees it with
//! `free`, so a block arrives holding whatever was there before and stops
//! existing when its deallocator returns.  Both ends were read: `PyObject_Init`
//! released the type it found in an uninitialised header, and the resurrection
//! check read the count out of a block that had been freed.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import gc

import cpyext_foreign_block as m

# 0 is what `calloc` leaves behind, 0x7f what `malloc` may.
m.make(0)
m.make(0x7f)
for _ in range(3):
    gc.collect()

assert m.released_yet(), 'the deallocator has not run'
first = m.released_intact()
assert first == -1, 'byte %d of the released block was written' % first

print('cpyext-foreign-block-ok')
"#;

#[test]
fn a_released_foreign_block_is_not_read_or_written() {
    let fixtures = Fixtures::new("cpyext-foreign-block");
    fixtures.compile("cpyext_foreign_block");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-foreign-block-ok");
}
