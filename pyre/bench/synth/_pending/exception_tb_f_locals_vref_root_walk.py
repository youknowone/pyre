# pyre-check: skip-backends=wasm
#
# PARKED: its guard_failures is not the same on every host. macos-latest and
# ubuntu-24.04 both report 5980 on cranelift; windows-latest reported 5979 in
# the run that promoted this file. That is the band #1043 removed the
# closure_per_call overlay over — two counters there disagreed with themselves
# across jobs, one toward its shared value and one away from it — so a
# `.cranelift.win32.jitstats` overlay cannot hold this either, and a missing
# baseline is a hard fail rather than an opt-out. The walker guard had no suite
# gate when that was written -- #1060 gave it one the next day, see ANSWERED
# below -- and this file is the reproduction, correct on all three native
# backends and PYRE_NO_JIT=1 at every size of a 512K-32M PYPY_GC_NURSERY sweep.
#
# ⛔ MEASURED 2026-08-24 at be1d37c1f94: IT NO LONGER DISCRIMINATES ITS GUARD.
# Removing `forward_virtual_ref_forced`'s value-stack arm from
# `walk_frame_value_slot` (pyre-interpreter/src/eval.rs) -- the whole of the fix
# this file reproduces -- and rebuilding cranelift, this file still exits 0 and
# prints the expected line: 3/3 at the default N, and once each at N=30000 and
# N=60000.  The pre-conversion source was used for that run, so the harness is
# not what silenced it.  Both binaries read the same freshly extracted LLBC.
# Something landed between 2026-08-05 and now that also prevents the fault, so
# promoting this would spend a run on three backends to gate nothing.  Find what
# closed it first; the reproduction below is only worth what it still catches.
#
# WHY, measured the same day with a counter compiled into the walker: the
# guard's arm NEVER FIRES.  `walk_frame_value_slot` runs -- the call counter
# reaches 3 within the first collection and the arm is reached on 26 of 40
# fixtures that read `f_locals`, `tb_frame` or `__traceback__` -- and across all
# 26 `forward_virtual_ref_forced` returned true exactly 0 times.  No
# `locals_cells_stack_w` slot holds a `JitVirtualRef` any more, on this fixture
# or on any of them, so there is nothing left for the arm to catch and removing
# it cannot fault.  `walk_raw_exception_roots` is still unguarded and
# `is_exception` still dereferences `ob_type`, so the arm is not redundant --
# its input has disappeared.
#
# ⛔ ANSWERED 2026-08-26 at 4953fb0edf8: NEITHER READING HOLDS. The vrefs are
# still built, and the guard already has a gate that is not this file.
#
# Measured with birth/force counters in `virtualref.rs` (`alloc_virtual_ref`,
# `force_virtual`) and hit counters on both `forward_virtual_ref_forced` call
# sites in `walk_pyframe_roots_area`:
#
#   this file, cranelift AND dynasm, at `15000 head` and at the default
#     alloc=8, force=4, value-stack arm hits=0 over 16-63 slot walks
#   the f_backref arm -- the SAME helper one block earlier in the same walk
#     0 here, but 1 on exception_residual_raise_caught_in_frame.py
#   the 70 synth fixtures naming f_locals/tb_frame/__traceback__/f_back/
#   _getframe, cranelift: value-stack arm hits 0 on all 70
#
# Eight vrefs are built on this shape at every size tried, so the virtualizable
# optimisation is applying and the "quietly not applying" reading is out. The
# f_backref line is the positive control the note above lacked: the helper is
# live and has a producer -- `inline_call.rs walker_ec_enter` stores the
# concrete vref into `ec.topframeref` and threads the caller's into
# `frame.f_backref`. Both are frame-shaped fields. Neither is a
# `locals_cells_stack_w` slot, and it is the value-stack arm specifically that
# has nothing feeding it.
#
# ⛔ The birth counter sees TRACING-time births only. A vref materialized by
# compiled code is allocated through NEW_WITH_VTABLE (`optimizeopt/
# virtualize.rs` stamps JIT_VIRTUAL_REF_VTABLE there), which this counter never
# observes -- `alloc=8` bounds the tracing population, not the runtime one. The
# runtime population is why the arm exists, so its emptiness is measured here
# only through the arm's own 0 hits.
#
# The gate: #1060 (`fa2eda0bd8e`) extracted the slot body as
# `walk_frame_value_slot` and covered the arm with
# `test_frame_value_slot_holding_a_virtual_ref_skips_the_pyobject_walks`
# (`pyre-interpreter/src/eval.rs`), which hand-builds a vref and asserts the
# visitor is handed exactly the slot and then the vref's own `forced` field.
# That PR recorded that removing the early return aborts the test with SIGABRT,
# and CI runs it on every push (`pyre-ci.yml`, `cargo test --all
# --no-default-features --features dynasm,cpyext`). Promoting this file would
# spend three backend runs re-gating what one unit test already holds.
#
# When it is promoted, `# pyre-check: selfcheck` is the shape, and the baseline
# blocker above dissolves rather than needing an overlay: `run_selfcheck` takes
# no jit-stats snapshot at all (0 of the 25 selfcheck fixtures carry a
# `.jitstats` baseline), so nothing records or compares `guard_failures`.  That
# shape is also the only one that reads the `skip-backends` line above --
# check.py spells it `synth_skip_backends(path) if selfcheck else ()` -- so the
# wasm exemption documented here is inert in any other shape.
#
# wasm is exempted above. It reads the catching frame mid-`except` as if the
# implicit `del e` had already run, so `f_locals` loses `e` on part of the loop
# and a second tuple reaches `seen`:
#   2 [(('drive', ('e', 'k', 'seen')), 'mid'), (('drive', ('k', 'seen')), 'mid')]
# Compiled-code only — clean at N=4000, wrong from N=8000 — and measured
# identical with the whole source tree reset to the merge base, so it predates
# the walker guard this file gates. Same stale-`f_locals`-from-compiled-code
# class as getframe_caller_locals_nested_compiled_callee, exempted the same way:
# not because wasm is right here.
#
# Reading `f_locals` off the OUTERMOST traceback node — the catching frame —
# leaves a JIT virtual ref in that frame's `locals_cells_stack_w`. A minor
# collection triggered by a nursery allocation from compiled code then walks the
# slot as a GC root. A vref's leading word is the `JIT_VIRTUAL_REF_VTABLE` magic
# rather than a PyObject `ob_type`, so a root walker that hands the slot to the
# raw exception walk dereferences the magic as a type pointer and takes a
# SIGSEGV. Reading the innermost node instead is clean.
#
# The collection has to land while the vref is on the stack, which is why the
# loop is long: on cranelift the crash is deterministic from roughly the 6000th
# iteration (2000 and 4000 stay clean). dynasm and the plain interpreter never
# reach that GC point on this shape, so this costs all three backends a run to
# gate one of them.
#
# The argument form is for narrowing by hand — `<pyre> thisfile.py 4000 tail`
# and the rest. The suite runs it with none, which takes the defaults below.
#
# Expected output: 1 [(('drive', ('e', 'k', 'seen')), 'mid')]

import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
WHICH = sys.argv[2] if len(sys.argv) > 2 else "head"


def mid(i):
    raise ValueError("boom")


def locs(tb):
    out = []
    idx = 0
    while tb is not None:
        f = tb.tb_frame
        want = (
            WHICH == "all"
            or (WHICH == "head" and idx == 0)
            or (WHICH == "tail" and tb.tb_next is None)
        )
        if want:
            out.append((f.f_code.co_name, tuple(sorted(f.f_locals))))
        else:
            out.append(f.f_code.co_name)
        tb = tb.tb_next
        idx += 1
    return tuple(out)


def drive():
    seen = set()
    k = 0
    while k < N:
        try:
            mid(k)
        except ValueError as e:
            seen.add(locs(e.__traceback__))
            e.__traceback__ = None
        k += 1
    return sorted(seen)


r = drive()
print(len(r), r)
