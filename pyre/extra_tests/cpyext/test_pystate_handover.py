# cpyext-fixture: cpyext_pystate
# cpyext-expect: cpyext-pystate-handover-ok

# The GIL, as an extension hands it back and forth.
#
# A thread holds the GIL for as long as it runs pyre code, so an extension
# that blocks has to give it up (`Py_BEGIN_ALLOW_THREADS`) and one calling in
# from a thread of its own has to take it (`PyGILState_Ensure`). Neither was
# reachable before: the macros and `PyEval_SaveThread` did not exist, and the
# two `PyGILState_*` entry points that did were declared nowhere an extension
# could see and answered without touching the GIL at all.
#
# `PyGILState_Check` is what makes the handover observable without timing
# anything -- it answers False exactly where the GIL was given up. The one
# measurement here has `sleep_holding` as its control: the same sleep with the
# release removed, which is what says the released one is *why* another thread
# ran rather than that threads run anyway.
#
# Every expectation was taken from CPython 3.14.6 running this same script
# against this same fixture.

import cpyext_pystate as m

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

# Running pyre code means holding the GIL.
eq('gil_held', m.gil_held(), True)

# Held outside the block, given up inside it, held again after.
eq('gil_around_block', m.gil_around_block(), (True, False, True))

# `Py_BLOCK_THREADS` takes it back for a stretch, `Py_UNBLOCK_THREADS` lets go.
eq('gil_around_block_threads', m.gil_around_block_threads(), (False, True, False))

# The explicit spelling the macros expand to: a thread state to hand back, and
# the GIL genuinely gone in between.
eq('save_restore', m.save_restore(), (True, True))

# Both Ensure calls report LOCKED, because this thread already held the GIL;
# they nest, and the GIL is still held between them.
eq('ensure_states', m.ensure_states(), (0, 0, True))

# One state per thread, and swapping NULL out and back answers what it replaced.
eq('thread_state_identity', m.thread_state_identity(), (True, True, True))

# Taking the GIL inside a block that gave it up makes this thread's state the
# current one, so the entry points that answer with it have an answer; UNLOCKED,
# because taking it is what that call had to do.  The release gives it back.
eq('ensure_inside_allow_threads',
   m.ensure_inside_allow_threads(), (1, True, True, True, True))

# A thread pyre never created runs Python under PyGILState_Ensure, which reports
# UNLOCKED because taking the GIL is what that call had to do.
eq('call_from_foreign_thread',
   m.call_from_foreign_thread(lambda: 'from-foreign'), (1, 'from-foreign'))

print('cpyext-pystate-handover-ok')
