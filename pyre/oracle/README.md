# `pyre/oracle`

Python scripts that are run for **what they print**, not for how fast they run.

`pyre/wasm_check.py` executes every script here and every script in
`pyre/bench`, then requires the stdout to agree across host engines. Both
directories serve that comparison; only `pyre/bench` is also scored for speed,
and only `pyre/bench` carries the `*.jitstats` baselines `pyre/check.py` and the
wasm codegen tests read.

Keeping the two apart is what lets either set be described in one sentence. A
script belongs here when it is deterministic and interesting to compare but is
not a workload anyone would rank engines by — an arbitrary-precision stress
loop, a shape that reproduces one codegen decision, a recursion small enough to
finish under an interpreter with no JIT.

Scripts here must print a stable final line and exit 0 under CPython and under
every pyre backend. Nothing filters this directory by name: a script that does
not meet that bar belongs somewhere else rather than on a skip list.
