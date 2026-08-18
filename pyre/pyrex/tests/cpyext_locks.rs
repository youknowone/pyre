//! What an extension holds while it works: its own mutex, a lock it
//! allocated, a buffer it resizes, and the error it chains onto one already on
//! its way out.
//!
//! Every expectation was taken from CPython 3.14.6 running this same script
//! against this same fixture.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import threading

import cpyext_locks as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


# ── the buffer an extension fills then cuts down ───────────────────────

# The pattern a caller with an upper bound uses: take a buffer that big, write
# what it turns out to need, then say how much that was.
eq('shrink', m.build_then_shrink(b'hello', 16, 5), (5, b'hello'))
eq('exact', m.build_then_shrink(b'hello', 5, 5), (5, b'hello'))
eq('shrink to zero', m.build_then_shrink(b'hello', 16, 0), (0, b''))
# The buffer is still the caller's to write after the resize, so what it holds
# is what was written last rather than what was there before.
eq('shrink then fill', m.shrink_then_fill(64, 8), (8, b'abcdefgh'))
eq('grow then fill', m.shrink_then_fill(4, 12), (12, b'abcdefghijkl'))

# A `bytes` that already exists is immutable, so the answer is a different one
# and the prefix is what carries over.
eq('existing shrink', m.resize_existing(b'abcdefgh', 3), (3, b'abc', None))
eq('existing same', m.resize_existing(b'abcdefgh', 8), (8, b'abcdefgh', None))
eq('existing grow', m.resize_existing(b'abcdefgh', 12), (12, b'abcdefgh', None))
eq('existing empty', m.resize_existing(b'', 4), (4, b'', None))
eq('existing to empty', m.resize_existing(b'abc', 0), (0, b'', None))

# The reference is given up whatever went wrong, which is what lets a caller
# write `if (_PyBytes_Resize(&v, n) < 0) return NULL;`.  Only the class: the
# report names the source of whoever made the call, and the runtime's own
# files are not the extension's.
for name, argument, size in [('negative', b'abc', -1), ('nonbytes', [], 4)]:
    answer = m.resize_refused(argument, size)
    eq('refused %s' % name, (answer[0], answer[1], answer[2][0]),
       (-1, True, 'SystemError'))

# ── the mistake an extension reports about itself ──────────────────────

# The macro names the caller's own file and line, so both sides say the same
# thing about the same call.
answer = m.bad_internal_call()
eq('bad internal call class', answer[0], 'SystemError')
eq('bad internal call text', answer[1].split('/')[-1].startswith('cpyext_locks.c:'), True)
eq('bad internal call tail', answer[1].endswith(': bad argument to internal function'), True)

# ── the mutex an extension embeds ──────────────────────────────────────

# Taken, then released, then taken again: the byte really goes back to where
# it started.
eq('mutex states', m.mutex_states(), (0, 1, 0, 1))
eq('mutex local', m.mutex_local(), (0, 1, 0))

# ── the lock an extension allocates ────────────────────────────────────

# Taken; refused while held; taken again after the release; a wait with a
# deadline gives up rather than blocking for good; a wait on a free lock takes
# it at once.
eq('thread lock', m.thread_lock(), (1, 0, 1, 0, 1))

eq('ident is this thread', m.thread_ident(), threading.get_ident())
seen = []
thread = threading.Thread(target=lambda: seen.append(m.thread_ident()))
thread.start()
thread.join()
eq('ident is per thread', seen[0] != m.thread_ident(), True)

# ── chaining onto what was already on its way out ──────────────────────

# The error being reported does not hide the one that was already on its way
# out: that one becomes its context.
first, second = ValueError('first'), TypeError('second')
raised, context = m.chain(first, second)
eq('chain both', (raised is first, context is second), (True, True))

# With nothing pending there is nothing to chain onto, so it is simply raised.
third = KeyError('third')
raised, context = m.chain(None, third)
eq('chain nothing pending', (raised is third, context), (True, None))

# A NULL argument is nothing to chain, so whatever was pending stands.
fourth = IndexError('fourth')
raised, context = m.chain(fourth, None)
eq('chain nothing to add', (raised is fourth, context), (True, None))

print('cpyext-locks-ok')
"#;

#[test]
fn the_locks_an_extension_holds() {
    let fixtures = Fixtures::new("cpyext-locks");
    fixtures.compile("cpyext_locks");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-locks-ok");
}
