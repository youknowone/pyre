"""Run the RPython matcher through the real metainterpreter and print the
optimized loop.

`LLJitMixin.meta_interp` is the harness RPython's own JIT tests use: it runs
the tracer, the optimizer and the LLGraph backend in process, so the loop it
produces is the trace RPython would compile -- without a multi-hour
translation. That is the object to compare against majit's peeled body.

Needs Python 2, because RPython is Python 2:

  pypy runner.py [n] [length] [--listing]

The repository root is located from this file, so PYTHONPATH need not be set.
"""

import sys
import os

# The version check comes before every other import. RPython, and this file's
# own `print` calls, are Python 2; under Python 3 the failure would otherwise
# be a SyntaxError naming a `print` line, which says nothing about the cause.
# Everything below this guard is written to *parse* under both, so the guard is
# what the reader sees.
if sys.version_info[0] != 2:
    sys.stderr.write(
        "runner.py needs Python 2: RPython's toolchain is Python 2, and this\n"
        "script imports rpython.jit.metainterp directly.\n"
        "\n"
        "    pypy %s\n"
        "\n"
        "(ran under Python %d.%d)\n" % (
            " ".join([sys.argv[0]] + (sys.argv[1:] or ["20", "4096"])),
            sys.version_info[0], sys.version_info[1],
        )
    )
    sys.exit(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def _repo_root():
    """The directory holding `rpython/`, found by walking up from this file.

    The RPython checkout is at the root of this repository, so nothing has to
    be installed and PYTHONPATH does not have to be set. Walking up rather
    than counting `..` keeps this correct if the example moves.
    """
    d = _HERE
    while True:
        if os.path.isdir(os.path.join(d, 'rpython', 'jit')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


_ROOT = _repo_root()
if _ROOT is None:
    sys.stderr.write(
        "could not find an `rpython/` checkout above %s.\n"
        "This script reads RPython's own JIT out of the repository root.\n" % _HERE
    )
    sys.exit(2)
sys.path.insert(0, _ROOT)

from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp.warmspot import get_stats

from rpython.rlib import jit

import fixture
import marked


# `warmspot.py:112` runs `jd.warmstate.set_param_trace_eagerness(2)  # for
# tests`, hardcoded, where a translated PyPy uses the `rlib/jit.py` PARAMETERS
# default of 200 and so does majit. Left alone the two sides answer different
# questions: at 2 this matcher compiles 1407 bridges over 4096 characters, at
# 200 it compiles 9, so a bridge count taken here is not comparable to majit's.
# `build_and_match` sets it back with `jit.set_param`, which runs after
# warmspot's setup and is the supported way to move a parameter.
TRACE_EAGERNESS = int(os.environ.get('RP_TRACE_EAGERNESS', '200'))


class RegexJit(LLJitMixin):
    pass


def build_and_match(n, length, seed):
    """The entry point `meta_interp` traces. Everything is built inside it,
    because the harness annotates from the argument types and ints are the
    only shape that is unambiguous."""
    jit.set_param(marked.jitdriver, 'trace_eagerness', TRACE_EAGERNESS)
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
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n = int(args[0]) if len(args) > 0 else 20
    length = int(args[1]) if len(args) > 1 else 4096

    # The cross-port input pin, on this side. `regex.rs`
    # `test_nonmatching_digest_pins_the_cross_port_input` asserts the same
    # digest; a trace census that agrees with the Rust one only means anything
    # if both ran over the same bytes.
    if not fixture.check_digest():
        return 1

    harness = RegexJit()
    res = harness.meta_interp(build_and_match, [n, length, 42],
                              listops=True, backendopt=True)
    print('')
    print('=== result: %s (0 == did not match, which is the benchmark) ===' % res)

    stats = get_stats()
    loops = stats.get_all_loops()
    # `send_loop_to_backend` calls `stats.add_new_loop` (compile.py:550);
    # `send_bridge_to_backend` does not. So `get_all_loops()` is the LOOPS, and
    # the bridges are only in `compiled_count`, which counts both
    # (compile.py:552, compile.py:604). Printing `len(loops)` and calling it
    # "loops/bridges" reports 1 for a run that compiled nine bridges.
    bridges = stats.compiled_count - len(loops)
    print('=== %d loop(s), %d bridge(s), %d aborted '
          '(trace_eagerness=%d) ===' % (
              len(loops), bridges, stats.aborted_count, TRACE_EAGERNESS))
    for idx, loop in enumerate(loops):
        body = peeled(loop.operations)
        print('')
        print('--- trace %d: %d ops total, %d in the peeled body ---' % (
            idx, len(loop.operations), len(body)))
        c = census(body)
        for name in sorted(c):
            print('  %-28s %d' % (name, c[name]))
        if '--listing' in sys.argv:
            print('  --- ops in order ---')
            for j, op in enumerate(body):
                print('  %4d  %s' % (j, op.getopname()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
