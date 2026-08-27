"""RPython translation target: the MASKING variant, as a native binary.

Same as `target.py` but importing `marked_masking` -- part 2's `&`/`|`
spelling. The two targets are the A/B, translated.

This is the row the post actually reports. `runner.py` next door runs the
metainterpreter in process and answers what the *trace* looks like; it cannot
answer how fast the trace runs, because `meta_interp` executes traces in an
interpreter. Only a translated binary can, which is what the post built:

    pypy ../../../../rpython/bin/rpython --opt=jit  target.py   # RPython + JIT
    pypy ../../../../rpython/bin/rpython --opt=2    target.py   # translated to C

Those are the post's two rows -- 16,500,000 and 720,000 chars/s on its 2010
hardware -- and dividing one by the other on ONE machine is the only quantity
that travels off it.

The binary takes `n length repeats` and prints one `chars/s` line per timed
run, after one untimed warm-up so the timed runs measure compiled code rather
than the tracing that produced it.
"""

import os
import sys
from time import time

# Import-time only, so it costs the translated binary nothing: put this
# directory and the `rpython/` checkout above it on `sys.path`, exactly as
# `runner.py` does, so the target runs untranslated for a smoke test and
# translates without PYTHONPATH being set.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_d = _HERE
while not os.path.isdir(os.path.join(_d, 'rpython', 'jit')):
    _parent = os.path.dirname(_d)
    if _parent == _d:
        _d = None
        break
    _d = _parent
if _d is not None and _d not in sys.path:
    sys.path.insert(0, _d)

import fixture_masking as fixture
import marked_masking as marked


def entry_point(argv):
    n = 20
    length = 1 << 20
    repeats = 5
    if len(argv) > 1:
        n = int(argv[1])
    if len(argv) > 2:
        length = int(argv[2])
    if len(argv) > 3:
        repeats = int(argv[3])

    re = fixture.bench_regex(n)
    s = fixture.nonmatching(length, n, 42)
    print "nodes=%d chars=%d repeats=%d" % (fixture.count(re), len(s), repeats)

    # The sink exists so nothing here is dead code to the C compiler: a
    # matcher whose answer is never read is a matcher that need not run.
    sink = 0

    # Untimed. The first pass pays for recording and compiling the trace, and
    # timing it would report that transition rather than the steady state.
    if marked.match(re, s):
        sink += 1

    times = []
    for _ in range(repeats):
        t0 = time()
        hit = marked.match(re, s)
        dt = time() - t0
        if hit:
            sink += 1
        times.append(dt)
        print "run chars_per_sec=%d secs=%f" % (int(len(s) / dt), dt)

    # Insertion sort, because RPython lists have no `sort` -- `SomeList` in
    # `annotator/unaryop.py` implements append/extend/reverse/insert/remove/
    # pop/index and nothing else. `repeats` is single digits, so the shape of
    # the sort does not matter; having a median does.
    i = 1
    while i < len(times):
        v = times[i]
        j = i - 1
        while j >= 0 and times[j] > v:
            times[j + 1] = times[j]
            j -= 1
        times[j + 1] = v
        i += 1
    best = times[0]
    mid = times[len(times) // 2]
    worst = times[len(times) - 1]
    # Median, not mean: benchmark noise is one-sided -- a preemption or a
    # thermal step can only make a run slower -- so a mean reports a run that
    # never happened.
    print "median chars_per_sec=%d" % int(len(s) / mid)
    print "min chars_per_sec=%d  max chars_per_sec=%d" % (
        int(len(s) / worst), int(len(s) / best))

    # 0 is the answer the benchmark input is built to produce; anything else
    # means the generator drifted and the row is not this benchmark's.
    print "sink=%d (0 == never matched, which is the benchmark)" % sink
    if sink != 0:
        return 1
    return 0


def target(driver, args):
    return entry_point, None


def jitpolicy(driver):
    from rpython.jit.codewriter.policy import JitPolicy
    return JitPolicy()


if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))
