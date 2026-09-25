# cpyext-fixture: cpyext_pystate
# cpyext-expect: cpyext-pystate-concurrency-ok

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

import threading
import time

import cpyext_pystate as m

stop = False
count = 0

def spin():
    global count
    while not stop:
        count += 1

def advanced_during(call, ms):
    before = count
    call(ms)
    return count - before

worker = threading.Thread(target=spin, daemon=True)
worker.start()
time.sleep(0.05)

released = advanced_during(m.sleep_released, 300)
holding = advanced_during(m.sleep_holding, 300)
stop = True
worker.join(timeout=5)

# A thread that never runs bytecode never reaches the periodic hand-off, so a
# held sleep gives the counter nothing; CPython's counterpart gives it about one
# switch interval's worth, spent before the call rather than during it. The
# margins are wide because what is being asserted is the difference between the
# two, not the rate of either.
assert released > 200, 'nothing ran while the GIL was released: %d' % released
assert released > 5 * holding, (
    'releasing the GIL did not let another thread run: released %d, held %d'
    % (released, holding))

print('cpyext-pystate-concurrency-ok')
