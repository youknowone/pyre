"""Run the RPython matcher through the real metainterpreter and print the
optimized loop.

`LLJitMixin.meta_interp` is the harness RPython's own JIT tests use: it runs
the tracer, the optimizer and the LLGraph backend in process, so the loop it
produces is the trace RPython would compile -- without a multi-hour
translation. That is the object to compare against majit's peeled body.

  PYTHONPATH=<repo root> pypy runner.py [n] [length]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp.warmspot import get_stats

import fixture
import marked


class RegexJit(LLJitMixin):
    pass


def build_and_match(n, length, seed):
    """The entry point `meta_interp` traces. Everything is built inside it,
    because the harness annotates from the argument types and ints are the
    only shape that is unambiguous."""
    re = fixture.bench_regex(n)
    s = fixture.nonmatching(length, n, seed)
    if marked.match(re, s):
        return 1
    return 0


def census(ops):
    """Ops by name, the same census `shortcircuit.rs` prints for majit."""
    counts = {}
    for op in ops:
        name = op.getopname()
        counts[name] = counts.get(name, 0) + 1
    return counts


def peeled(ops):
    """Everything from the last LABEL on -- what runs per input character.

    A compiled loop is preamble plus peeled body, exactly as in majit, and
    grading the whole thing charges the body for reads the preamble hoisted.
    """
    last = -1
    for i, op in enumerate(ops):
        if op.getopname() == 'label':
            last = i
    return ops[last:] if last >= 0 else ops


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

    harness = RegexJit()
    res = harness.meta_interp(build_and_match, [n, length, 42],
                              listops=True, backendopt=True)
    print
    print '=== result: %s (0 == did not match, which is the benchmark) ===' % res

    stats = get_stats()
    loops = stats.get_all_loops()
    print '=== %d loop(s)/bridge(s) compiled ===' % len(loops)
    for idx, loop in enumerate(loops):
        body = peeled(loop.operations)
        print
        print '--- trace %d: %d ops total, %d in the peeled body ---' % (
            idx, len(loop.operations), len(body))
        c = census(body)
        for name in sorted(c):
            print '  %-28s %d' % (name, c[name])
        if '--listing' in sys.argv:
            print '  --- ops in order ---'
            for j, op in enumerate(body):
                print '  %4d  %s' % (j, op.getopname())


if __name__ == '__main__':
    main()
