# Preamble Peeling TODO (RPython compile.py parity)

## Status
- [x] Phase 1: export virtual state at JUMP
- [x] Phase 2: build_body_trace_with_virtual_import + re-optimize
- [x] combine_preamble_and_body
- [ ] **Phase 2 body trace uses optimized Phase 1 body, not raw trace**
- [ ] **Eliminate remaining New ops in Phase 2 body**
- [ ] **Fix OpRef position collisions in combine_preamble_and_body**
- [ ] **Connect short preamble to Phase 2 virtual state for bridges**
- [ ] **Clean up unused flatten_virtuals_in_peeled_trace**

## 1. Phase 2 body trace size (PRIORITY)
Currently `build_body_trace_with_virtual_import` feeds the ORIGINAL
unoptimized trace to Phase 2. RPython feeds the SAME trace to both
phases — but Phase 2 imports the exported state from Phase 1, so
OptVirtualize starts with knowledge that certain inputargs are virtual.

Fix: Phase 2 should process the original trace (same as Phase 1),
but the optimizer should be pre-populated with imported virtual state.
This means adding `import_state()` to the Optimizer that tells
OptVirtualize "inputarg N is a VirtualStruct with these fields".

RPython ref: unroll.py:479-504 import_state()

## 2. Eliminate remaining New in Phase 2
Phase 2 currently produces 2 New ops in logo trace (should be 0).
Root cause: OptVirtualize sees the reconstruction New+SetfieldGc
but the subsequent body ops create additional New nodes that escape
through the JUMP.

Fix: ensure the JUMP in Phase 2 body also flattens virtual fields
(same as Phase 1). The JUMP should carry field VALUES, not virtual
pointers. OptVirtualize's JUMP handler should flatten instead of
force when in Phase 2 context.

RPython ref: unroll.py _jump_to_existing_trace()

## 3. OpRef position collisions in combine
combine_preamble_and_body naively concatenates preamble and body ops.
OpRef positions from Phase 1 and Phase 2 can collide, causing the
Cranelift backend to produce wrong code.

Fix: remap Phase 2 OpRef positions to not overlap with Phase 1.

## 4. Short preamble for bridges
When a guard fails in the body and a bridge is compiled, the bridge
needs to enter the loop body with the correct virtual state.
RPython stores the short preamble + virtual state in TargetToken.

Fix: store Phase 2 virtual state in the compiled loop metadata.
When compiling a bridge, use `generalization_of()` to check
compatibility and `inline_short_preamble()` to replay facts.

RPython ref: unroll.py:320-362 _jump_to_existing_trace()

## 5. Cleanup
Remove `flatten_virtuals_in_peeled_trace` if no longer used,
or integrate its logic into `build_body_trace_with_virtual_import`.
