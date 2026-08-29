# pyre-check: max-pypy-ratio=96
# Deep self-recursion whose caller frame is relocated by a minor collection
# while the recursive callee runs.  The bytecode CALL fast path drops the
# arguments, runs the callee, then pushes its result onto the caller's value
# stack.  The callee allocates enough to trigger a minor collection that moves
# the caller frame, so the raw pointer captured before the call goes stale; the
# result push and its valuestackdepth bump must land on the forwarded live
# frame.  When they hit the abandoned copy the live frame keeps a stack depth
# one slot short, the following BINARY_OP reads the range iterator instead of
# the recursion result ("unsupported operand type(s) for +: 'range_iterator'
# and 'int'"), and the dropped exception segfaults.  cat(5) sums to 5! per
# call; 1000 outer iterations warm the JIT and sustain the allocation pressure
# that forces the relocation.  The ratio ceiling above is deliberately loose --
# branchy recursion is the architectural JIT gap -- and what this fixture is
# really for is the correctness/crash guard.
#
# The wasm baseline diverges from the two native ones on purpose, and the
# divergence is the news rather than noise, so re-snapshotting it without
# reading this is the wrong move.  Once `for k in range(n)` folds on a live
# bound, `cat`'s inner loop gets a procedure token, the self-recursive
# `cat(n - 1)` reaches an already-compiled loop, and the metainterp records a
# CALL_ASSEMBLER where this fixture previously had none.  dynasm and cranelift
# compile that (`guard_failures` 636 -> 500, identical on both).  wasm declines
# it twice: `general_int_call_assembler_target` finds the target still carrying
# `register_pending_call_assembler_target`'s all-zero geometry
# (`ca_none_frame_bytes`, with the input types matching), so `allow_ca` is
# false, `wasm_unsupported_trace_reason` refuses the whole trace, and on wasm
# every `compile_loop` `Err` IS a `loops_aborted` bump.  The two declined loops
# fall back to the interpreter: `guard_failures` 638 -> 939, and one extra
# bridge.  They also spend the key's abort budget, so it latches
# `JC_DONT_TRACE_HERE` (`abort_ceiling_banned` 0 -> 1), which bans inlining
# from elsewhere too -- the cost is wider than the two aborts.
#
# That admission gap is wasm's alone and predates the fold: wasm answers
# `supports_tmp_callback_call_assembler` false, so instead of a redirectable
# tmp callback the metainterp bakes the pending token, and a pending token that
# is not the one its trace is finally compiled under never has its geometry
# filled in.  Net wall clock on wasm still improves 3x with the aborts in place
# (min-of-five interleaved, 0.42s -> 0.14s), which is why the fold stands and
# the divergence is recorded rather than optimised away.
def cat(n):
    if n <= 1:
        return 1
    r = 0
    for k in range(n):
        r += cat(n - 1)
    return r


total = 0
for i in range(1000):
    total += cat(5)
print(total)
