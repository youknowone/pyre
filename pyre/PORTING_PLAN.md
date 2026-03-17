# pyre Porting Plan

This document tracks the interpreter-side restructuring needed to port PyPy's
interpreter architecture into `pyre` while keeping `pyre-bytecode` thin and
keeping meta-tracing as the primary JIT model.

## Constraints

- `pyre-bytecode` stays intentionally thin for now.
- `pyre` remains the focus; `majit` feature requests belong in
  `majit/COMPATIBILITY_MATRIX.md`.
- Structural work should move `pyre` closer to `pypy/interpreter` and
  `pypy/objspace/std`, not invent a parallel architecture unless there is a
  clear simplification payoff.

## Current Layering

- `pyre-object`
  - Raw object layout and standard object implementations.
  - Target analogue: `pypy/objspace/std/*object.py`.
- `pyre-runtime`
  - Interpreter-owned runtime objects and services.
  - Current contents: builtin function objects, code/function wrappers,
    display/error types, builtin installation, execution context.
  - Target analogue: the lower-level part of `pypy/interpreter/*`.
- `pyre-objspace`
  - Object operation dispatch and type-specialized fast paths.
- `pyre-interp`
  - Frame state, bytecode execution loop, and JIT portal integration.
- `pyre-jit`
  - `pyre`-specific tracing/JIT adapter layer on top of `majit`.
- `pyre-main`
  - CLI entry point.

## Completed Restructuring

- Moved interpreter-owned runtime objects out of `pyre-object` into
  `pyre-runtime`.
- Moved builtin installation and execution-context creation out of `PyFrame`.
- Added `PyExecutionContext` as the shared per-run environment used by frames.

## Next Work Items

1. Split module globals, locals, and builtins cleanly.
   - Introduce explicit module/global state instead of a single merged namespace.
   - Keep `PyFrame` focused on execution state.

2. Port the missing interpreter substrate from PyPy.
   - Add `pycode`, function-call argument binding, module objects,
     descriptor/call protocol, and error propagation that more closely matches
     `pypy/interpreter`.

3. Port frame-adjacent semantics before broad language coverage.
   - Closures/cells/freevars.
   - Exceptions and traceback state.
   - Generators/coroutines and frame suspension.

4. Align the JIT portal with the PyPy frame model.
   - Replace symbolic-only locals/stack handling with real virtualizable
     accesses in the trace recorder.
   - Add promotion points and guard behavior where PyPy's interpreter relies on
     them.

5. Grow validation in the same order as the structure.
   - Keep `pyre-interp` unit tests green.
   - Add focused tests for execution-context/module/frame responsibilities.
   - Add PyPy-inspired trace and deopt regression cases once frame semantics are
     stable.

## Virtualizable Frame Rollout

`PyFrame` already carries the heap fields that should become the canonical
virtualizable state:

- `next_instr`
- `stack_depth`
- `locals_w[*]`
- `value_stack_w[*]`

What is still missing is moving the JIT state and trace recorder over to that
frame model instead of mirroring it in parallel vectors.

Current status:

- Done: `PyFrame` arrays now use a stable pointer-backed layout that matches the
  virtualizable helper model.
- Done: the portal imports frame locals/stack/instruction state through
  virtualizable sync hooks instead of cloning those arrays in `eval.rs`.
- Done: fast-local trace paths now emit explicit virtualizable array reads and
  writes against the frame-backed locals array.
- Done: JIT residual calls for Python functions now re-enter the real
  interpreter call path with the caller frame attached, instead of using a
  placeholder helper.
- Done: `BuildList`, `BuildTuple`, `BuildMap`, and `MakeFunction` no longer use
  semantics-changing shortcuts in the tracer; they now dispatch to residual
  helpers that construct the same runtime object kinds as the interpreter.
- Done: `Call`, `BuildList`, `BuildTuple`, `BuildMap`, `StoreSubscr`,
  `ListAppend`, `UnpackSequence`, and `MakeFunction` now share an
  interpreter-owned stack protocol helper layer instead of open-coding stack
  choreography separately in `eval.rs` and `trace.rs`.
- Done: supported bytecode dispatch now goes through a single
  interpreter-owned opcode-step dispatcher. `eval.rs` and `trace.rs` no longer
  keep separate top-level `match Instruction` loops for the same opcode set.
- Done: fast-local, namespace, stack-manipulation, and range-iterator
  boilerplate now also routes through shared opcode-step helpers. `trace.rs`
  no longer keeps a second copy of `LoadFast*`, `StoreFast*`, `Load/StoreName`,
  `LoadGlobal`, `PopTop`, `PushNull`, `Copy`, `Swap`, `GetIter`, `ForIter`,
  and `PopIter` stack choreography.
- Done: trace-side stack pushes, pops, and swaps now update the
  virtualizable `PyFrame.value_stack_w` / `stack_depth` state as they record,
  instead of keeping the operand stack only in a parallel symbolic vector.
- Done: trace-side stack reads (`peek`, `pop`, `swap`) now also source their
  values from the virtualizable frame stack array, not just from the shadow
  symbolic stack vector. The shadow stack no longer stores live operand values;
  only shape bookkeeping remains, while opcode execution reads from the same
  frame-backed stack model that the interpreter mutates.
- Done: trace-side fast-local stores and stack swaps no longer keep the shadow
  locals/stack vectors eagerly synchronized for opcode semantics. Before
  closing a loop, the tracer now captures jump args directly from the
  virtualizable frame arrays, so loop closure no longer depends on refreshing
  a second locals/stack OpRef copy.
- Done: trace-side guards now capture explicit typed fail_args from the
  frame-backed live state instead of relying only on a `num_live` count. Guard
  deoptimization snapshots are now assembled directly from the virtualizable
  frame-backed locals and operand stack.
- Done: `ReturnValue` no longer aborts tracing. The tracer now records a
  terminal `Finish` action so return paths can be compiled as traces instead of
  unconditionally falling back to the interpreter.
- Done: `UnaryNot` and `PopJumpIf{False,True}` now route through shared truth
  protocol helpers, so branch truth testing no longer keeps a second
  interpreter/tracer stack protocol.
- Done: `BinaryOp`, `CompareOp`, `UnaryNegative`, and `UnaryInvert` now share
  the same interpreter-owned stack protocol helper layer. The tracer no longer
  carries handwritten type-specialized arithmetic semantics for those opcodes;
  it reuses `objspace` semantics through JIT residual helpers instead.
- Done: `LoadConst` and small-int loading now also route through a shared
  constant-loading helper layer. Tuple and code constants follow the same
  recursive construction rules in both the interpreter and tracer instead of
  being handwritten twice.
- Done: traced `int` and `float` constants are now boxed at trace time and
  embedded as object pointer constants. Hot compiled traces no longer call
  `jit_w_int_new` / `jit_w_float_new` for `LOAD_CONST`.
- Done: `ForIter` no longer reimplements range-iterator stepping through
  handwritten field arithmetic in the tracer. It now uses a residual helper
  around the same runtime iterator step function, and the trace-side branch
  decision probes iterator state without mutating the live interpreter object.
- Done: traced `range` iteration no longer calls `jit_range_iter_next_or_null`.
  Hot `ForIter` paths now read `current/stop/step` directly from the iterator
  object, guard the step sign and continuation predicate, update
  `current` in-place, and only leave boxed-result allocation as a residual
  seam.
- Done: hot `LoadName` / `StoreName` for already-materialized globals no longer
  route through namespace helpers. `PyNamespace` now has a stable slot/value
  array layout, and the tracer reads or writes known name slots directly from
  the frame's namespace object.
- Done: hot concrete builtin and Python-function calls no longer have to go
  through the generic `jit_call_callable_*` dispatch helper. The tracer now
  guards the exact callable object and lowers directly to builtin-only or
  function-only call helpers when the concrete callable kind is already known.
- Done: hot negative-index `list` / `tuple` subscription no longer falls back
  to residual `getitem`/`setitem` helpers. The tracer now specializes concrete
  negative indices directly against the container layout, guarding exact length
  where needed so Python's negative-index normalization still matches the
  interpreter state.
- Done: truth testing for `None`, `float`, `str`, `dict`, `list`, and `tuple`
  no longer has to route through the generic `jit_truth_value` helper when the
  concrete object kind is already known at trace time. The tracer now guards
  the concrete object layout directly and emits primitive IR against the same
  runtime payload fields the interpreter reads.
- Done: hot known builtin `len(...)` calls over `str`, `dict`, `list`, and
  `tuple` no longer have to go through builtin callable dispatch when the
  concrete builtin object is already known. The tracer now guards the exact
  builtin object, reads the container length directly from the same object
  layout the interpreter uses, and only leaves integer boxing as a residual
  seam.
- Done: hot known builtin `abs(...)` calls over concrete `int` operands no
  longer have to go through builtin callable dispatch. The tracer now guards
  the exact builtin object and lowers directly to branchless integer payload
  IR, falling back to the residual builtin path only for overflow-promoting
  cases such as `i64::MIN`.
- Done: hot known builtin `isinstance(obj, "type_name")` calls over a traced
  concrete object and concrete type-name string no longer have to go through
  builtin callable dispatch. The tracer now guards the object class and the
  exact type-name object, then folds the simplified builtin to the correct
  bool singleton directly in the trace.
- Done: hot known builtin `type(obj)` calls over a traced concrete object no
  longer have to go through builtin callable dispatch. The simplified builtin
  now reuses the shared constant-string cache for type names, and the tracer
  can guard the object class and return the same cached `W_StrObject`
  directly.
- Done: hot known builtin `min(a, b)` / `max(a, b)` calls over concrete
  small-cache `int` operands no longer have to go through builtin callable
  dispatch. For the subset where returning `w_int_new(result)` preserves the
  same cached object identity as the concrete interpreter result, the tracer
  now emits branchless integer payload IR and reuses the small-int cache for
  the result object.
- Done: string constants are now boxed through a shared constant-string cache
  in both the interpreter and the tracer, so `LOAD_CONST "literal"` reuses
  the same immutable `W_StrObject` identity across executions. This removes a
  spurious mismatch between interpreter and trace-time constant handling and
  makes exact constant-string guards line up with the runtime behavior.
- Done: trace-side `Call` no longer branches between builtin and Python
  function call helpers. The tracer now emits one residual callable-dispatch
  helper that reuses the same runtime callable classification at execution
  time, and the old builtin/function-specific trace helpers have been removed.
- Done: `UnpackSequence` no longer open-codes tuple-vs-list dispatch and class
  guards in the tracer. It now goes through a single residual sequence-item
  helper, so trace-side container-kind branching is no longer duplicated there.
- Done: `Swap` and `UnaryInvert` now have trace-side parity instead of falling
  back to implicit aborts from missing dispatch coverage.
- Done: raw `PyFrame` / `PyObjectRef` live values now travel through the
  compiled `pyre` path as `Type::Int`, and trace-side frame/object memory
  accesses use raw/int descriptors instead of pretending Rust heap pointers are
  `majit` GC-managed refs. Compiled hot range loops now survive a full
  trace/compile/finish cycle under test.
- Done: `pyre` now supplies a trace-shape-specific `JitDriverDescriptor`
  instead of relying on a static red layout. That descriptor marks `frame` as
  the virtualizable red and keeps `majit`'s sync/type plumbing aligned with
  the actual locals/namespace/stack shape of each trace.
- Done: `pyre-interp` no longer reaches into `MetaInterp` directly to run a
  compiled loop. Compiled terminal/jump execution now goes through a
  `JitDriver` helper, so the interpreter loop owns less ad hoc compiled-path
  plumbing.
- Done: `pyre-interp` back-edges now build `PyreJitState` only once and hand
  the full "start tracing or run compiled loop" decision to `JitDriver`.
  `jump_backward()` no longer keeps separate compiled-loop and hot-count paths.
- Done: `PyreJitState` no longer keeps shadow copies of fast locals or the
  operand stack. Those live values are now read from and restored into the
  virtualizable `PyFrame` arrays directly, so the heap frame is the only
  runtime source of truth for locals/stack state.

## Recursive Call Status

- Done: self-recursive `CallAssembler` now uses the same fresh function-entry
  ABI shape as separately traced function-entry loops:
  `[frame, next_instr, stack_depth, locals...]`.
- Done: `trace_box_int` now boxes traced integer results through
  `jit_w_int_new`, so recursive traces reuse the same small-int cache policy as
  the interpreter instead of open-coding `New + Setfield`.
- Done: `DONT_TRACE_HERE` now participates in the actual inlining decision, so
  `pyre` follows the same broad PyPy strategy for recursive traces:
  trace-too-long aborts converge toward separate function-entry traces and then
  `call_assembler`, instead of retrying the same inlining decision forever.
- Remaining: recursive compiled calls still cross the boundary as boxed Python
  ints. The largest remaining `fib_recursive` gap versus PyPy is therefore the
  `call_assembler -> boxed int -> caller unbox` round-trip, not the recursive
  trace-compile path itself.
- Done: module-level `namespace` values no longer travel as a parallel JIT
  live-value vector. `LoadName` / `StoreName` now go through frame-namespace
  helpers, and the heap frame namespace is the only runtime source of truth
  for global/module bindings during tracing and deoptimization.
- Done: trace recording now writes the virtualizable `frame.next_instr` field
  directly. Fallthrough, taken branches, iterator exhaustion, and explicit
  jumps update the same instruction pointer field that the interpreter mutates.
- Done: `eval.rs` no longer snapshots the live value stack into the
  `merge_point` closure. Trace-side branch and iterator hot-path decisions now
  read the concrete `PyFrame` heap object directly, so the interpreter frame is
  also the concrete source of truth during recording.
- Done: `PyreSym` no longer mirrors the frame's local-slot count. The tracer
  reads local shape from the concrete frame when it needs bounds or guard/jump
  fail-args, shrinking the symbolic executor toward control-flow bookkeeping.
- Done: `PyreSym` no longer persists operand-stack depth across trace steps.
  Each `trace_bytecode` call now re-seeds stack depth from the concrete frame,
  keeps depth changes local to the opcode being recorded, and closes loops via
  explicit `jump_args` validation instead of symbolic-depth validation.
- Done: `PyreSym` no longer carries bytecode `ExtendedArg` decode state.
  The tracer now reconstructs the decoded opcode/arg pair from `pc` on demand,
  so symbolic state is down to the live frame handle itself.
- Done: conditional branches, `JumpForward`, `JumpBackward`, `ReturnValue`,
  and `ForIter` now route through shared control-flow helpers in
  `pyre-runtime/src/opcode_step.rs`. `eval.rs` and `trace.rs` no longer keep
  separate branch/loop transition algorithms for those opcodes.
- Done: trace-side frame stack/local mutation, typed guard fail-arg capture,
  and loop-close live-arg materialization now live in
  `pyre-jit/src/state.rs` as `TraceFrameState`. `trace.rs` no longer owns raw
  frame array mutation or live-state snapshot assembly; it delegates those
  operations to the frame-backed JIT state layer.
- Done: shared opcode traits now own checked-load, truth-guard,
  optional-value-guard, and close-loop protocol defaults. `trace.rs` no longer
  chooses `GuardTrue` vs `GuardFalse`, `GuardNonnull` vs `GuardIsnull`, or
  `StepResult::CloseLoop` at the opcode adapter layer; it only provides the
  low-level guard emission hooks that back those shared defaults.
- Done: trace entry finalization is now centralized in
  `pyre-jit/src/state.rs`. `trace_bytecode` no longer open-codes
  `StepResult::{Continue,CloseLoop,Return}` to `TraceAction`
  translation inline; it delegates that last step to the JIT-state helper
  layer.
- Done: the public `pyre-jit/src/trace.rs` module is now just the stable
  trace entry wrapper used by `pyre-interp`. Trace recording executes directly
  through `TraceFrameState`, so there is no separate trace executor module
  anymore.
- Done: helper ABI lowering for trace-time calls, constants, namespace access,
  and arithmetic now lives alongside `TraceFrameState` in
  `pyre-jit/src/state.rs`. The remaining trace-specific lowering is attached
  directly to the frame-backed JIT state layer instead of living in a second
  runtime shim module.
- Done: `OpcodeStepExecutor` now provides the default opcode-arm
  implementations in `pyre-runtime/src/opcode_step.rs`. The concrete
  interpreter executor in `pyre-interp/src/eval.rs` and the frame-backed trace
  executor in `pyre-jit/src/state.rs` no longer carry duplicate one-line
  `load_fast`/`jump`/`build_*`/`call` dispatch bodies around the shared opcode
  stepper.
- Done: the concrete interpreter no longer uses a separate opcode-executor
  wrapper object. `PyFrame` itself now implements the shared opcode-step
  traits in `pyre-interp/src/eval.rs`, so the concrete interpreter path and
  the trace path both execute the shared stepper against frame-shaped state.
- Done: concrete callable/container/namespace semantics are shared instead of
  being open-coded twice. `pyre-runtime/src/runtime_ops.rs` now owns neutral
  helpers such as `dispatch_callable`, function creation, namespace access,
  and sequence/container construction, while `pyre-objspace/src/opcode_ops.rs`
  owns the shared concrete truth/arithmetic/compare semantics reused by both
  the interpreter and the JIT helper ABI layer.
- Done: trace-time helper ABI selection is now centralized in
  `pyre-jit/src/helpers.rs` via selector enums and generic helper lookup.
  `TraceFrameState` no longer names most individual JIT helper entrypoints
  directly; it performs generic backend calls while the helper table owns the
  ABI mapping.
- Done: trace-time object/arithmetic/helper lowering bodies now also live in
  `pyre-jit/src/helpers.rs`. `TraceFrameState` delegates those operations to
  helper-emission APIs and is correspondingly narrower: frame state,
  guard capture, and control-flow bookkeeping stay in `state.rs`, while
  backend call emission lives in the helper layer.
- Done: virtualizable frame reads now go through descriptor-backed GC-style
  field/array ops instead of raw frame loads, and pyre's field/array
  descriptors now carry stable indices so optimizer aliasing can distinguish
  frame locals, value stack, and scalar fields correctly.
- Done: compiled loop entry, loop close, and guard fail args are now frame-only
  in `pyre-jit/src/state.rs`. Red live-state duplication for locals/stack has
  been removed from the JIT boundary: compiled traces re-enter with just the
  live frame handle, while locals/stack/`next_instr`/`stack_depth` are
  recovered from the virtualizable heap frame itself.
- Done: `PyreJitState` no longer caches `local_count` or `stack_capacity` as
  separate interpreter-side shadow fields. Virtualizable array lengths, meta
  compatibility checks, and restore loops now re-read locals/stack shape
  directly from the heap `PyFrame`, so frame shape is no longer duplicated in
  the JIT state object.
- Done: `PyreJitState` no longer caches sorted namespace keys either.
  `build_meta()` and compatibility checks now read the module/global namespace
  shape directly from the heap `PyFrame`, so namespace shape no longer travels
  as a second interpreter-side shadow copy alongside the live frame.
- Done: trace-side helper ABI lowering is no longer open-coded across the
  `TraceFrameState` trait implementations. `pyre-jit/src/helpers.rs` now owns
  a `TraceHelperAccess` boundary with default lowering methods, and
  `TraceFrameState` only supplies frame access plus `GuardNotForced` recording.
- Done: `bool` and `None` singleton constants no longer go through residual
  helper calls during trace recording, and truth-to-bool lowering now records
  direct integer arithmetic that selects the correct bool singleton address
  instead of calling back into a trace helper wrapper.
- Done: pure object-space and iterator helper ABIs for truth / binary /
  compare / unary / range-next now live in `pyre-objspace` and
  `pyre-runtime` instead of `pyre-jit`, so the JIT crate no longer owns those
  concrete runtime wrappers and only references their exported ABIs.
- Done: JIT callable-call bridge registration and the flat callable/list/tuple
  / map helper ABIs now also live in `pyre-runtime`. `pyre-jit` keeps the
  trace emission layer, but the concrete helper bodies for those residual
  runtime calls are no longer implemented in the JIT crate itself.
- Done: namespace/globals helper ABIs (`load_name`, `store_name`,
  `make_function`) now operate on namespace pointers and live in
  `pyre-runtime`, while the tracer reads the namespace pointer from the frame
  via IR field access instead of calling a frame-specific helper wrapper.
- Done: `sequence_getitem` JIT ABI now lives in `pyre-runtime`, and
  `getitem`/`setitem` JIT ABI wrappers now live in `pyre-objspace`, so those
  concrete object-space adapters are no longer implemented in `pyre-jit`.
- Done: trace-time constant boxing and range-iterator allocation wrappers now
  also live in `pyre-object` / `pyre-runtime`, and `pyre-jit/src/helpers.rs`
  no longer defines concrete bigint/str/code boxing or `jit_range_iter_new`
  bodies itself.
- Done: trace-time residual helper emission no longer routes through a
  `TraceIntHelper` / `TraceVoidHelper` enum catalog in `pyre-jit`. The tracer
  now emits calls to the shared ABI entrypoints directly, so one more
  pyre-specific helper-dispatch layer has been removed from the recording path.
- Done: `PyObjectArray` now lives in `pyre-object` and is shared by
  `PyFrame`, `list`, and `tuple` storage. Container object layout is now using
  the same stable pointer-backed array wrapper as the frame virtualizable
  storage, instead of a separate `Vec<PyObjectRef>` representation.
- Done: trace-side `UNPACK_SEQUENCE` no longer always residual-calls
  `jit_sequence_getitem` for tuple/list inputs. When the concrete object is a
  list or tuple with the traced length, the tracer now records `GuardClass` +
  `GuardValue(len == count)` and reads items by direct raw-array access against
  the shared `PyObjectArray` layout.
- Done: trace-side `BinaryOp::Subscr` and `StoreSubscr` now have direct
  list/tuple fast paths for concrete `int` indices, and `ListAppend` can also
  mutate list storage directly when the traced list has spare capacity. Those
  cases now record guard + raw-array access against `PyObjectArray` instead of
  always falling back to residual helper calls.
- Done: trace-side `int` arithmetic (`Add`, `Sub`, `Mul`), integer
  comparisons, and `int` / `bool` truth testing no longer always residual-call
  `pyre-objspace` helpers. For concrete small-object cases the tracer now
  records `GuardClass` on the object layout, reads the payload fields
  directly, emits primitive integer IR (`IntAddOvf`, `IntSubOvf`, `IntMulOvf`,
  `IntLt`, `IntEq`, `IntNe`, ...), and only uses a helper call for the final
  boxed `int` result when needed.
- Done: trace-side integer bitwise ops (`And`, `Or`, `Xor`, `Lshift`,
  `Rshift`) and unary integer ops (`-`, `~`) now also lower to primitive IR
  when the traced operands are concrete `int` objects with safe machine-int
  semantics. The tracer still falls back to the shared object-space helper path
  for overflow, negative shifts, or non-`int` cases.
- Done: positive-`int` `//` and `%` cases now also lower directly to
  `IntFloorDiv` / `IntMod` in traces when the traced operands are known to
  match machine-int semantics (`lhs >= 0`, `rhs > 0`). Negative or exceptional
  cases still fall back to the shared Python object-space path.
- Done: `pyre` no longer synthesizes a dynamic per-trace driver descriptor for
  locals/stack red arguments. With frame-only red live state, the JIT driver
  descriptor is now a fixed `frame`-virtualizable descriptor.
- Done: reconstructed root-frame resume slots are also frame-only now. Bridge
  and guard-failure reconstruction for the top `PyFrame` refreshes
  locals/stack/shape from the heap frame instead of serializing those values
  back through explicit reconstructed-frame slots.
- Pending: `PyreJitState` now uses the frame arrays as the single locals/stack
  source of truth, `PyreSym` is down to the live frame handle, and
  `TraceFrameState` owns the remaining frame-backed trace mutation while
  `TraceHelperAccess` centralizes helper ABI lowering. The main remaining gap
  from PyPy's interpreter-level tracing model is now the residual helper calls
  for callable dispatch, namespace access, container operations outside the
  direct list/tuple fast paths, and the remaining object-space fallbacks that
  still sit between the trace recorder and concrete interpreter/object-space
  steps.

1. Collapse the remaining symbolic bookkeeping at the JIT boundary.
   - `PyreJitState` should eventually carry only the frame plus runtime state
     that truly lives outside the virtualizable frame.
   - Current status: `PyreSym` is now just the live frame handle, and
     `TraceFrameState` owns frame-backed stack/local/guard plumbing.
   - Current status: the shared opcode layer also owns checked-load and branch
     / iteration / close-loop guard protocol defaults.
   - Remaining work: collapse the trace-only helper ABI lowering in
     `TraceFrameState` further so the opcode recorder is backend glue instead
     of interpreter-aware adapter code.

2. Change trace recording to operate on the virtualizable frame.
   - `LoadFast` / `StoreFast` should go through virtualizable array accesses.
   - Stack push/pop and `stack_depth` updates should go through virtualizable
     array/field accesses.

3. Add deopt-focused regression coverage.
   - A hot loop should preserve locals and stack through compiled execution.
   - A guard failure should force the virtualizable frame back to the heap.
   - Nested Python calls should preserve execution-context and frame ownership
     correctly.

## Meta-Tracing Refactor

The long-term target is to eliminate `pyre-jit/src/trace.rs` as a second
bytecode interpreter.

The required sequence is:

1. Move hot bytecode semantics behind interpreter-owned helper functions.
   - `Call`, locals access, stack mutation, and jump decisions should stop being
     duplicated in both `eval.rs` and `trace.rs`.
   - Current status: the supported opcode dispatch table is shared, and
     call/container/subscript/list-append/unpack stack protocol is shared.
     Remaining duplication is now concentrated in the trace executor methods
     for locals, arithmetic, truth testing, guards, and symbolic stack state.

2. Make tracing consume interpreter-owned step semantics.
   - Either through a generated trace view of the interpreter step function or
     through an interpreter-local abstraction layer that both concrete execution
     and tracing implement.

3. Shrink `pyre-jit` to backend-facing glue.
   - Helper ABI shims, frame layout metadata, and backend descriptors are fine.
   - Bytecode dispatch semantics should live with `pyre-interp`.

4. Delete semantics-changing trace shortcuts.
   - Tracing must never substitute one Python object kind for another
     (`tuple` as `list`, placeholder user-function calls, etc.).

## Porting Sequence

The preferred order is:

1. execution context and module/global layering
2. code/function/argument semantics
3. closures and exceptions
4. generators and suspended frames
5. virtualizable-frame migration in the portal and tracer
6. JIT parity against the stabilized frame model

This order keeps the interpreter model from drifting away from the eventual
meta-tracing portal.

## Success Criteria

- `pyre-object` contains standard object layout and object storage concerns only.
- `pyre-runtime` owns interpreter-level environment and runtime services.
- `pyre-interp` can be read as a direct Rust analogue of the hot path in
  `pypy/interpreter/pyframe.py` and adjacent interpreter modules.
- JIT tracing logic operates on the same frame model that the interpreter uses,
  instead of a parallel ad hoc representation.
