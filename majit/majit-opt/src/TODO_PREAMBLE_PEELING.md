# Preamble Peeling TODO (RPython compile.py parity)

## Status
- [x] Phase 1: export virtual state at JUMP
- [x] Phase 2: import_state + re-optimize with imported virtual knowledge
- [x] combine_preamble_and_body (assemble_peeled_trace)
- [x] pre_force_field_refs removed (RPython parity: field info preserved on alloc_ref)
- [x] Forwarded/InfoArena scaffolding removed
- [ ] **JUMP handler: don't force VirtualStruct (RPython has no optimize_JUMP)**
- [ ] **Phase 2 body trace uses optimized Phase 1 body, not raw trace**
- [ ] **Connect short preamble to Phase 2 virtual state for bridges**

## 1. JUMP handler RPython parity (BLOCKED)
RPython's OptVirtualize has no optimize_JUMP — JUMP passes through without
forcing. force_box_for_end_of_preamble handles discrimination: VirtualStruct
stays virtual, Virtual (instance) gets forced.

Currently majit forces all virtuals in the JUMP handler. This works but
requires compensating mechanisms: expand_info for Instance/Struct, constant
OpRef 10100+ remapping, pre_force_virtual_state/pre_force_jump_args.

**Blocked by**: flush/guard/virtual-reconstruction coupling. See
memory/jump_handler_refactoring.md for details on 3 coupled systems.

RPython ref: optimizer.py:536-556, unroll.py:452-477

## 2. Phase 2 body trace
Phase 2 processes the original trace with imported virtual state. Currently
correct but could benefit from sharing more short preamble information.

RPython ref: unroll.py:479-504 import_state()

## 3. Short preamble for bridges
When a guard fails in the body and a bridge is compiled, the bridge needs
to enter the loop body with the correct virtual state. RPython stores the
short preamble + virtual state in TargetToken.

RPython ref: unroll.py:320-362 _jump_to_existing_trace()
