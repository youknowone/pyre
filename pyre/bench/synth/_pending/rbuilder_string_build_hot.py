# pyre-check: max-pypy-ratio=60
# rbuilder epic (#50) end-to-end fixture — the first Python-level program whose
# hot loops drive the StringBuilder value the epic models. Each loop builds
# strings through an interpreter path that accumulates into a `Wtf8Buf`/`String`
# (unicodeobject.rs concat / repeat / join / replace), which the front lowers to
# builder-mode (`graph_has_builder_accumulator`): `ll_new` -> `ll_append*` ->
# `ll_build`, and the shrink/fold `build` tree. Output is a deterministic integer
# accumulator per loop, so check.py can compare it byte-for-byte across the
# dynasm / cranelift / wasm backends and against CPython / PyPy.
#
# PENDING (do not move to pyre/bench/synth/ until baselined): this file lives in
# `_pending/` because `run_synthetic_suite` globs `*.py` non-recursively, so the
# gate skips it. The jitstats snapshots (check.snap/{dynasm,cranelift}/synth/)
# and the `max-pypy-ratio` above must be RECORDED and TUNED on a charon-capable
# host — `python pyre/check.py --snapshot --synthetic-pattern rbuilder_string_build_hot`
# — before promotion; trip counts may then need the same GC-threshold headroom
# `str_fstring.py` documents. Authored + verified deterministic on CPython here;
# the JIT gate is CI-only in this environment (charon absent).

# ── join_hot: str.join over a small fixed list (join buf accumulator) ──
join_hot__N = 60000

def join_hot__main():
    parts = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    acc = 0
    i = 0
    while i < join_hot__N:
        s = '-'.join(parts)
        acc = acc + len(s)
        i = i + 1
    print(acc)
join_hot__main()

# ── repeat_hot: str * n (repeat accumulator sized len*count) ──
repeat_hot__N = 60000

def repeat_hot__main():
    unit = 'ab'
    acc = 0
    i = 0
    while i < repeat_hot__N:
        s = unit * (i & 7)
        acc = acc + len(s)
        i = i + 1
    print(acc)
repeat_hot__main()

# ── concat_chain: incremental += building a growing buffer (append + grow) ──
concat_chain__N = 4000

def concat_chain__main():
    acc = 0
    i = 0
    while i < concat_chain__N:
        s = ''
        j = 0
        while j < 32:
            s = s + 'xy'
            j = j + 1
        acc = acc + len(s)
        i = i + 1
    print(acc)
concat_chain__main()

# ── replace_hot: str.replace over a repeated pattern (replace buf loop) ──
replace_hot__N = 60000

def replace_hot__main():
    base = 'a.b.c.d.e.f'
    acc = 0
    i = 0
    while i < replace_hot__N:
        s = base.replace('.', '::')
        acc = acc + len(s)
        i = i + 1
    print(acc)
replace_hot__main()
