# codegen_state.rs Patch for Preamble Peeling

## Linked list path: extract_live, live_value_types, create_sym

RPython parity: virtual head pointer replaced by field values in inputargs.
Layout: [pool, selected, value_0, next_0, value_1, next_1, ...]
(no head pointer — fields directly)

### extract_live (replace ~line 1180)

```rust
fn extract_live(&self, meta: &__JitMeta) -> Vec<i64> {
    // RPython make_inputargs_and_virtuals: head → [value, next]
    let mut values = vec![
        (&self.#pool_field as *const _ as usize) as i64,
        self.#sel_field as i64,
    ];
    for &(sidx, _) in &meta.storage_layout {
        let head = self.#pool_field.get(sidx).head_ptr();
        if head != 0 {
            let node = head as *const aheui_runtime::storage::StackNode;
            unsafe {
                values.push(aheui_runtime::value::val_to_i64(&(*node).value));
                values.push((*node).next as usize as i64);
            }
        } else {
            values.push(0);
            values.push(0);
        }
    }
    values
}
```

### live_value_types (replace ~line 1193)

```rust
fn live_value_types(&self, meta: &__JitMeta) -> Vec<majit_ir::Type> {
    let mut types = vec![majit_ir::Type::Int, majit_ir::Type::Int];
    for _ in &meta.storage_layout {
        types.push(majit_ir::Type::Int); // value
        types.push(majit_ir::Type::Ref); // next
    }
    types
}
```

### create_sym (replace ~line 1202)

```rust
fn create_sym(meta: &__JitMeta, header_pc: usize) -> __JitSym {
    // Heads loaded lazily via ensure_linked_list_head (Phase 1)
    // or injected as VirtualStruct via imported_virtuals (Phase 2)
    let heads = std::collections::HashMap::new();
    __JitSym {
        pool_ref: majit_ir::OpRef(0),
        current_selected: meta.initial_selected,
        current_selected_value: Some(majit_ir::OpRef(1)),
        storage_layout: meta.storage_layout.clone(),
        loop_header_pc: header_pc,
        header_selected: meta.initial_selected,
        trace_started: false,
        meta_storage_count: meta.storage_layout.len(),
        linked_list_heads: heads,
        stacks: std::collections::HashMap::new(),
    }
}
```

### collect_jump_args (replace ~line 1252)

Should include field values for heads that were loaded (for Phase 1 JUMP):

```rust
fn collect_jump_args(sym: &__JitSym) -> Vec<majit_ir::OpRef> {
    let selected = sym.current_selected_value.unwrap_or(majit_ir::OpRef::NONE);
    let mut args = vec![sym.pool_ref, selected];
    for &(sidx, _) in sym.storage_layout.iter().take(sym.meta_storage_count) {
        args.push(
            sym.linked_list_heads.get(&sidx).copied().unwrap_or(majit_ir::OpRef::NONE),
        );
    }
    args
}
```

Note: Phase 1's collect_jump_args returns [pool, selected, head].
Phase 2's flatten_virtuals_at_jump replaces head with [value, next] in the JUMP.
export_flatten_jump_args in unroll.rs does the same for the Label.
So Label and body JUMP have matching arity.

### fail_args and fail_args_types (~line 1072)

Phase 1: fail_args = [pool, selected, head_0, ...] (head as Ref)
Phase 2: fail_args = [pool, selected, value_0, next_0, ...] (flattened)

The flatten_virtuals_at_jump flag controls which layout the JUMP uses.
fail_args_types must match.

## Summary of invariants

- extract_live returns [pool, selected, value_0, next_0, ...]
- Phase 1 JUMP: [pool, selected, head_0] (collect_jump_args)
- export_flatten: head_0 → [value_0, next_0] → label_args = [pool, selected, value_0, next_0]
- Phase 2 JUMP: [pool, selected, value_0, next_0] (flatten_virtuals_at_jump)
- Label args = label_args = Phase 2 JUMP args → same arity ✓
- extract_live = label_args layout → compiled entry matches ✓
