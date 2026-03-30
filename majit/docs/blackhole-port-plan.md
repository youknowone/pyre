# Blackhole Port Plan: RPython resume_in_blackhole → majit

## 1. Goal

Replace the current guard failure recovery path (shared mutable state
restoration) with RPython's orthodox architecture:

```
compiled code → deadframe/fail_args → resume data decode
  → blackhole frame chain → register-array dispatch
  → merge_point or finish
```

The blackhole interpreter must have **independent register arrays**
(`registers_i`, `registers_r`, `registers_f`) and must NOT share mutable
heap state with the main interpreter.

## 2. Current Path vs Target Path

### Guard Failure Flow Comparison

| Step | Current (majit) | Target (RPython parity) |
|------|-----------------|-------------------------|
| Guard fails | Compiled code exits, returns fail_values in DeadFrame | Same |
| Dispatch | `jitdriver.rs` calls `run_compiled_detailed()` | Same |
| Decision | `must_compile_with_values()` → bridge or restore | `must_compile()` → bridge or `resume_in_blackhole()` |
| **Bridge path** | `start_bridge_tracing()` with `guard_resume_pc` | `_trace_and_compile_from_bridge()` via MetaInterp |
| **Fallback path** | `restore_guard_failure()` — directly patches interpreter state | `resume_in_blackhole()` → frame chain → `_run_forever()` |
| State source | fail_values + exit_layout → `restore_guard_failure_values()` | resume data → `ResumeDataDirectReader` → register arrays |
| Execution | Main interpreter resumes at loop header (`target_pc`) | Blackhole runs from guard PC to merge_point/finish |
| Mid-loop resume | **NOT SUPPORTED** — always returns to header | Supported — blackhole has independent registers |

### Why Current Path Is Wrong

1. **Double mutation**: Compiled code already mutated heap (SetfieldGc).
   Restoring interpreter state and re-running from loop header repeats
   those mutations.
2. **No mid-loop resume**: Guard at PC=42 always jumps back to PC=0,
   losing work done between PC=0..42.
3. **Resume data unused**: `ResumeData` / `EncodedResumeData` structures
   exist but `restore_guard_failure()` bypasses them entirely.

## 3. Ownership Layers

| Data | Owner (current) | Owner (target) |
|------|-----------------|----------------|
| `resume_pc` (guard's bytecode position) | Appended to fail_args (last value) | `ResumeGuardDescr.rd_numb` via `read_jitcode_pos_pc()` |
| Frame chain (call stack at guard) | Not reconstructed | `blackhole_from_resumedata()` builds `nextblackholeinterp` chain |
| Register values at guard | fail_args → `restore_guard_failure_values()` → shared state | fail_args → `ResumeDataDirectReader` → `registers_i/r/f` |
| Virtualizable fields | Restored via `sync_after()` from exit_meta | `consume_vable_info()` writes directly into virtualizable |
| Virtual objects | `MaterializedVirtual` in `GuardRecovery` | Lazy allocation in `ResumeDataDirectReader.getvirtual_ptr()` |
| Pending field writes | `ResolvedPendingFieldWrite` in `GuardRecovery` | `consume_pendingfields()` applied during reconstruction |
| Exception state | `ExceptionState` in `BlackholeResult` | `_prepare_resume_from_failure(deadframe)` extracts from CPU |
| Virtual references | Not tracked | `consume_virtualref_info()` restores vref→virtual mapping |

## 4. Architecture Diagram

```
                    ┌──────────────────────┐
                    │   Compiled Code      │
                    │   (Cranelift JIT)    │
                    └──────┬───────────────┘
                           │ guard fails
                           ▼
                    ┌──────────────────────┐
                    │   DeadFrame          │
                    │   fail_values[]      │
                    └──────┬───────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼                             ▼
    ┌─────────────────┐          ┌─────────────────────┐
    │ must_compile()   │          │ resume_in_blackhole()│
    │ → bridge trace   │          │                     │
    └─────────────────┘          └──────┬──────────────┘
                                        │
                                        ▼
                              ┌──────────────────────┐
                              │ blackhole_from_       │
                              │ resumedata()          │
                              │                       │
                              │ 1. ResumeDataDirect-  │
                              │    Reader(storage,    │
                              │    deadframe)         │
                              │ 2. consume_vref_and_  │
                              │    vable()            │
                              │ 3. Loop: acquire_     │
                              │    interp, setposition│
                              │    consume_one_section│
                              └──────┬──────────────┘
                                     │
                                     ▼
                              ┌──────────────────────┐
                              │ _run_forever()        │
                              │                       │
                              │ while chain:          │
                              │   bh._resume_mainloop │
                              │   release_interp(bh)  │
                              │   bh = bh.next        │
                              └──────┬──────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                                 ▼
          ┌─────────────────┐              ┌─────────────────┐
          │ merge_point     │              │ DoneWithThisFrame│
          │ → return to JIT │              │ → interpreter    │
          │   dispatch loop │              │   exit           │
          └─────────────────┘              └─────────────────┘
```

## 5. Key RPython Structures to Port

### 5.1 BlackholeInterpBuilder (blackhole.py:52)

Pool-based factory for `BlackholeInterpreter` instances. Caches
released interpreters for reuse. Generates `dispatch_loop` from
instruction table (`setup_insns`).

**majit gap**: No builder/pool pattern. `BlackholeInterpreter::new()`
allocates fresh each time.

### 5.2 BlackholeInterpreter Register Layout (blackhole.py:282)

```
registers_i = [val0, val1, ..., valN, const0, const1, ...]
               ├── num_regs_i() ──┤├── constants ──────┤
```

Three separate register banks. Constants appended after working
registers by `setposition()` → `copy_constants()`.

**majit status**: Structure exists (`BlackholeInterpreter` at
blackhole.rs:781). Registers, tmpregs, jitcode, position all present.
Missing: builder pool, `copy_constants` integration with resume path.

### 5.3 ResumeDataDirectReader (resume.py:1354)

Reads encoded resume data (tagged numbering) directly into blackhole
interpreter registers. Key method: `consume_one_section(blackholeinterp)`
uses liveness info to determine which registers are live, then fills
them from tagged values (TAGCONST/TAGINT/TAGBOX/TAGVIRTUAL).

**majit gap**: `ResumeLayoutSummary.reconstruct_state()` exists but
returns `ReconstructedState` instead of writing directly into a
`BlackholeInterpreter`. The indirection through `ReconstructedFrame`
is an unnecessary intermediate step.

### 5.4 blackhole_from_resumedata (resume.py:1312)

Builds the entire blackhole frame chain:
1. Create `ResumeDataDirectReader`
2. Consume virtualref and virtualizable sections
3. Loop: `read_jitcode_pos_pc()` → `setposition()` → `consume_one_section()`
4. Return top-of-chain interpreter

**majit gap**: No equivalent function. Guard failure goes through
`handle_guard_failure_in_trace_with_savedata()` → `GuardRecovery` →
`restore_guard_failure_with_resume_layout()` which patches shared state
instead of building a blackhole chain.

### 5.5 handle_fail (compile.py:701)

Decision point: `must_compile()` → bridge, else `resume_in_blackhole()`.
Simple two-branch dispatch. The `assert 0, "unreachable"` after both
branches confirms that both paths diverge permanently (bridge compiles
and re-enters JIT; blackhole runs to completion and raises
`DoneWithThisFrame` or `ContinueRunningNormally`).

**majit gap**: `jitdriver.rs:1088` has the decision but the fallback
path returns to shared interpreter state instead of diverging into
blackhole execution.

## 6. Vertical Slice Definition

The minimum viable change to validate the architecture:

1. **Input**: A single guard failure in a simple loop (e.g., `sum_loop`
   with overflow guard)
2. **Resume data**: One frame, no virtuals, no pending fields, no
   virtualizable
3. **Expected flow**:
   - Guard fails → extract fail_values from DeadFrame
   - `blackhole_from_resumedata()` builds 1-frame chain
   - `consume_one_section()` fills `registers_i` from fail_args
   - `run()` executes jitcode from guard PC to merge_point
   - Return to JIT dispatch loop with correct values
4. **Validation**: Same final result as current restore path, but via
   blackhole registers instead of shared state mutation

## 7. Risks

1. **proc macro jitcode generation**: The `#[jit_interp]` macro must
   produce jitcode that the blackhole can execute. This is the hardest
   part — every interpreter operation needs a blackhole bytecode encoding.
2. **Runtime stack operations**: Aheui uses mutable stacks
   (`push`/`pop`). The blackhole must either have shadow stacks or defer
   stack ops to the real runtime.
3. **Heap mutations in compiled code**: Compiled code has already mutated
   the heap. The blackhole must not re-execute those mutations. This is
   solved by resuming at the guard PC (not the loop header) with
   independent register state.
4. **Bridge tracing start point**: Bridge must start from the
   blackhole-resumed state, not the shared interpreter state.
