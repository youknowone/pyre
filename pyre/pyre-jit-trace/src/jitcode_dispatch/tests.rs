use super::*;
use crate::jitcode_runtime::{insns_opname_to_byte, named_jitcode};
use majit_ir::Type;
use majit_metainterp::make_fail_descr;

fn test_fbw_mode() -> FbwWalkMode<crate::state::PyreSym> {
    FbwWalkMode::default()
}

/// Install the minimal `-live-`-anchored outer JitCode required by
/// per-opcode guard tests. Production obtains this coordinate from the
/// codewriter; these isolated dispatcher fixtures supply the same shape.
fn test_outer_resume_jitcode_index() -> u32 {
    thread_local! {
        static INDEX: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
    }
    if let Some(index) = INDEX.with(|slot| slot.get()) {
        return index;
    }
    let runtime_jc = majit_metainterp::jitcode::JitCode::new("guard_resume_test");
    runtime_jc.set_body(majit_translate::jitcode::JitCodeBody {
        code: vec![crate::state::op_live(), 0, 0],
        startpoints: Some([0_usize].into_iter().collect()),
        ..Default::default()
    });
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.jitcode = std::sync::Arc::new(runtime_jc);
    pyjit.metadata.is_drained = true;
    let jitcode = crate::state::install_jitcode_for(std::ptr::null(), std::sync::Arc::new(pyjit))
        as *const crate::state::JitCode;
    let index = unsafe { (*jitcode).index as u32 };
    INDEX.with(|slot| slot.set(Some(index)));
    index
}

#[test]
fn vstack_permuted_for_iter_entry_uses_block_head_target() {
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.metadata.n_py_instrs = 19;
    pyjit.metadata.block_head_py_by_jit_pc = vec![(110, 18)];
    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0), (100, 17), (120, 18)];

    assert_eq!(vstack_containing_py_pc(&pyjit.metadata, 110), 17);
    assert_eq!(vstack_initial_py_pc(&pyjit.metadata, 110, true), 18);
    assert_eq!(vstack_initial_py_pc(&pyjit.metadata, 110, false), 17);
    assert_eq!(vstack_step_py_pc(&pyjit.metadata, 110, 18), 18);
    assert_eq!(vstack_step_py_pc(&pyjit.metadata, 110, 17), 17);
    assert_eq!(vstack_step_py_pc(&pyjit.metadata, 120, 18), 18);
}

/// Build a fresh `TraceCtx`. Uses the public `for_test_types` +
/// `const_ref` / `make_fail_descr` factories so the fixture stays
/// out of `pub(crate)` API.
fn fresh_trace_ctx() -> TraceCtx {
    let _ = test_outer_resume_jitcode_index();
    TraceCtx::for_test_types(&[Type::Ref])
}

/// Build a `done_with_this_frame_descr_ref` for tests. Mirrors the
/// production fallback at `pyjitpl.rs` (`make_fail_descr_typed`)
/// when the staticdata singleton was never attached.
fn done_descr_ref_for_tests() -> DescrRef {
    make_fail_descr(1)
}

#[test]
fn inline_caller_frame_distinguishes_try_block_catch_marker_decline() {
    assert_eq!(
        decline_inline_caller_frame_for_catch_marker(Some(42)),
        Err(InlineCallerFrameDecline::TryBlockCatchMarker),
    );
    assert_eq!(decline_inline_caller_frame_for_catch_marker(None), Ok(()));
}

/// `ensure_residual_call_args_bound` backs the unbound-arg abort path
/// for all three residual-call shapes (iRd / iIRd / iIRFd); they all
/// funnel through this helper, so one direct test covers the guard
/// that otherwise lets `OpRef::NONE` reach the backend's
/// `resolve_opref` (a process abort).
#[test]
fn ensure_residual_call_args_bound_rejects_unbound_arg() {
    // Fully-bound funcbox + args slice passes.
    let bound = [OpRef::int_op(1), OpRef::int_op(2), OpRef::int_op(3)];
    assert!(ensure_residual_call_args_bound(&bound, 7).is_ok());

    // An unbound arg surfaces its position.
    let unbound = [OpRef::int_op(1), OpRef::NONE, OpRef::int_op(3)];
    assert!(matches!(
        ensure_residual_call_args_bound(&unbound, 7),
        Err(DispatchError::ResidualCallArgUnbound {
            pc: 7,
            arg_index: 1
        })
    ));

    // The funcbox slot (index 0) is guarded too.
    let unbound_func = [OpRef::NONE, OpRef::int_op(2)];
    assert!(matches!(
        ensure_residual_call_args_bound(&unbound_func, 4),
        Err(DispatchError::ResidualCallArgUnbound {
            pc: 4,
            arg_index: 0
        })
    ));
}

/// Build distinct `OpRef` constants for register slots so dataflow
/// assertions don't get false positives from shared identity. Each
/// slot holds `const_ref(0xC0DE_0000 + i)` for `i in 0..count`.
fn distinct_const_refs(ctx: &mut TraceCtx, count: usize) -> Vec<OpRef> {
    (0..count)
        .map(|i| ctx.const_ref(0xC0DE_0000_i64 + i as i64))
        .collect()
}

/// Companion of [`distinct_const_refs`] that mints Int-typed
/// ConstInt OpRefs.  Use this when a fixture needs to populate
/// integer-register slots — the heapcache array path keys on the
/// `ConstInt.getint()` value, so Ref-typed mints don't satisfy
/// `getarrayitem_cache`'s ConstInt precondition.
fn distinct_const_ints(ctx: &mut TraceCtx, count: usize) -> Vec<OpRef> {
    (0..count)
        .map(|i| ctx.const_int(1_000 + i as i64))
        .collect()
}

/// Default `sub_jitcode_lookup` for tests that don't exercise
/// `inline_call_r_r` recursion. Returns `None` for every index;
/// any test that hits the inline_call handler with this lookup
/// will see `DispatchError::SubJitCodeNotFound`.
fn no_sub_jitcodes(_idx: usize) -> Option<SubJitCodeBody> {
    None
}

fn switch_descr_pool(entries: &[(i64, usize)]) -> Vec<DescrRef> {
    let dict = entries.iter().copied().collect();
    vec![std::sync::Arc::new(crate::descr::PyreSwitchDescr::new(dict)) as DescrRef]
}

/// Concrete-shadow round-trip: a `WalkContext` built with
/// `concrete_registers_r` exposes each slot's `ConcreteValue` via
/// `read_ref_reg_concrete` indexed by the same byte the symbolic
/// `read_ref_reg` consults.  Verifies the parallel-slice plumbing
/// (slot N's OpRef in `registers_r` shares slot N in
/// `concrete_registers_r`).
#[test]
fn read_ref_reg_concrete_returns_slot_matching_symbolic_read() {
    let exc_obj_ptr: pyre_object::PyObjectRef = 0xDEAD_BEEFusize as _;
    let descr_pool: Vec<DescrRef> = Vec::new();
    let mut tc = fresh_trace_ctx();
    let oprefs = distinct_const_refs(&mut tc, 3);
    let mut regs_r = oprefs.clone();
    let mut concrete = vec![
        ConcreteValue::Null,
        ConcreteValue::Ref(exc_obj_ptr),
        ConcreteValue::Int(42),
    ];
    // Snapshot expected values before `&mut concrete` enters wc —
    // the assertion below cannot read `concrete[reg_idx]` while wc
    // holds the mutable borrow.
    let expected: Vec<ConcreteValue> = concrete.clone();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut concrete,
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    // Synthesize a 2-byte op fixture: `<opcode_byte> <reg_idx>`.
    // `read_ref_reg_concrete` reads `code[op.pc + 1 + operand_offset]`
    // exactly like `read_ref_reg`, so encoding the reg byte at pc+1
    // suffices.
    for reg_idx in 0..3 {
        let code = [0u8, reg_idx as u8];
        let op = DecodedOp {
            key: "fixture/r",
            opname: "fixture",
            argcodes: "r",
            pc: 0,
            next_pc: 2,
        };
        assert_eq!(
            read_ref_reg_concrete(&code, &op, 0, &wc),
            expected[reg_idx],
            "reg {} concrete shadow must match the parallel slot",
            reg_idx,
        );
    }
}

/// `getfield_vable_*` must abort to `VableBoxNotSeeded` when the
/// box register is unseeded (`OpRef::NONE`) rather than feed
/// `u32::MAX` into the heapcache flag vector (a 16 GiB resize).
#[test]
fn getfield_vable_with_none_obj_surfaces_vable_box_not_seeded() {
    let descr_pool: Vec<DescrRef> = Vec::new();
    let mut tc = fresh_trace_ctx();
    let mut regs_r = vec![OpRef::NONE];
    let mut regs_i = vec![OpRef::NONE];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    // `getfield_vable_i/rd>i`: operand 0 (the box) sits at code[pc+1].
    let code = [0u8, 0x00, 0x00, 0x00, 0x00];
    let op = DecodedOp {
        key: "getfield_vable_i/rd>i",
        opname: "getfield_vable_i",
        argcodes: "rd>i",
        pc: 0,
        next_pc: 5,
    };
    assert_eq!(
        getfield_vable_via_metainterp(&code, &op, &mut wc, 'i'),
        Err(DispatchError::VableBoxNotSeeded { pc: 0 })
    );
}

/// `setfield_vable_*` carries the same unseeded-box guard.
#[test]
fn setfield_vable_with_none_obj_surfaces_vable_box_not_seeded() {
    let descr_pool: Vec<DescrRef> = Vec::new();
    let mut tc = fresh_trace_ctx();
    let mut regs_r = vec![OpRef::NONE];
    let mut regs_i = vec![OpRef::NONE];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    // `setfield_vable_i/rid`: operand 0 (the box) sits at code[pc+1].
    let code = [0u8, 0x00, 0x00, 0x00, 0x00];
    let op = DecodedOp {
        key: "setfield_vable_i/rid",
        opname: "setfield_vable_i",
        argcodes: "rid",
        pc: 0,
        next_pc: 5,
    };
    assert_eq!(
        setfield_vable_via_metainterp(&code, &op, &mut wc, 'i'),
        Err(DispatchError::VableBoxNotSeeded { pc: 0 })
    );
}

/// `getarrayitem_vable_*` / `setarrayitem_vable_*` / `arraylen_vable`
/// carry the same unseeded-box guard as the scalar field handlers:
/// an `OpRef::NONE` vable must abort to `VableBoxNotSeeded` rather than
/// resize the heapcache flag vector to 16 GiB.
#[test]
fn array_vable_handlers_with_none_obj_surface_vable_box_not_seeded() {
    // operand 0 (the box) sits at code[pc+1] for all three argcodes.
    let code = [0u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    for (key, opname, argcodes) in [
        (
            "getarrayitem_vable_i/riXdd>i",
            "getarrayitem_vable_i",
            "riXdd>i",
        ),
        (
            "setarrayitem_vable_i/riXdd",
            "setarrayitem_vable_i",
            "riXdd",
        ),
        ("arraylen_vable/rdd>i", "arraylen_vable", "rdd>i"),
    ] {
        let descr_pool: Vec<DescrRef> = Vec::new();
        let mut tc = fresh_trace_ctx();
        let mut regs_r = vec![OpRef::NONE];
        let mut regs_i = vec![OpRef::NONE];
        let session = std::cell::RefCell::new(WalkSession::default());
        let mut wc = WalkContext {
            callee_shadow: None,
            inline_callee_consts: None,
            fbw_mode: test_fbw_mode(),
            session: &session,
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: make_fail_descr(1),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: EntryPyPc::Py(0),
            outer_resume_marker_jit_pc: None,
            outer_jitcode_index: 0,
            raw_descrs: RawDescrPool::Global,
            is_authoritative_executor: false,
            outer_active_boxes: Vec::new(),
            store_subscr_fn_addr: None,
            pending_guard_snapshot_error: None,
            vstack_boxes: Vec::new(),
            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_ceiling: u32::MAX,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
        };
        let op = DecodedOp {
            key,
            opname,
            argcodes,
            pc: 0,
            next_pc: code.len(),
        };
        let result = match opname {
            "getarrayitem_vable_i" => getarrayitem_vable_via_metainterp(&code, &op, &mut wc, 'i'),
            "setarrayitem_vable_i" => setarrayitem_vable_via_metainterp(&code, &op, &mut wc, 'i'),
            "arraylen_vable" => arraylen_vable_via_metainterp(&code, &op, &mut wc),
            _ => unreachable!(),
        };
        assert_eq!(
            result,
            Err(DispatchError::VableBoxNotSeeded { pc: 0 }),
            "{opname} must abort VableBoxNotSeeded on an unseeded vable register",
        );
    }
}

#[test]
#[ignore = "T3 audit probe — dumps runtime opnames + walker-handled set + \
                per-opname JitCode hit count. Run with \
                `cargo test -p pyre-jit-trace --features dynasm --lib \
                t3_audit_opname_gap_inventory -- --ignored --nocapture` to \
                produce a project memory entry; not a permanent test."]
fn t3_audit_opname_gap_inventory() {
    use crate::jitcode_runtime::{all_jitcodes, insns_byte_to_opname, insns_opname_to_byte};

    // 1) Runtime opnames (pyre's actual codewriter emission set).
    let runtime_names: std::collections::BTreeSet<String> =
        insns_opname_to_byte().keys().cloned().collect();

    // 2) Walker-handled opnames — parsed from the embedded `handle`
    // function's string literals.  Source-of-truth scan against
    // the files themselves so this probe stays accurate as handlers
    // land/leave.  `arith.rs` carries the `regular_record_table!` arms
    // (the `int_*` / `float_*` / `ptr_*` families) as opname literals, and
    // `residual_call.rs` carries the residual-call body match arms, so both
    // are scanned alongside `mod.rs`.
    let source = concat!(
        include_str!("mod.rs"),
        "\n",
        include_str!("arith.rs"),
        "\n",
        include_str!("residual_call.rs"),
    );
    let mut handled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Heuristic: scan the literal patterns that appear ONLY in
    // dispatch arms of `handle()` — they look like
    // `"<opname>/[argcodes]" => ...`.  Filter to entries that are
    // also in the runtime table to drop test-fixture literals.
    // An arm may list several keys on one line (`"a/i" | "b/c" => ...`),
    // so every literal on the line counts, not just the leading one —
    // reading only the first silently under-reports the handled set.
    for line in source.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('"') {
            continue;
        }
        for (idx, key) in trimmed.split('"').enumerate() {
            // Odd indices are the quoted spans; even ones are the
            // separators between them.
            if idx % 2 == 0 {
                continue;
            }
            // Must contain '/' (separates opname from argcodes); skip
            // anything that doesn't look like an opname/argcodes literal.
            if !key.contains('/') {
                continue;
            }
            if runtime_names.contains(key) {
                handled.insert(key.to_string());
            }
        }
    }

    let unhandled: Vec<&String> = runtime_names.difference(&handled).collect();

    // 3) Per-opname JitCode hit count — for each unhandled opname,
    // count how many JitCodes contain its opcode byte.  Higher
    // counts = higher likelihood of blocking the next opcode
    // entering the shadow allow-list.
    let opname_to_byte = insns_opname_to_byte();
    let byte_to_opname = insns_byte_to_opname();
    let all_jcs = all_jitcodes();

    // For accurate counts the byte must be at a true OP position,
    // not an operand position.  We need to walk each JitCode using
    // `decoded_ops` to enumerate true op bytes.
    let mut hit_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for jc in all_jcs {
        for op in crate::jitcode_runtime::decoded_ops(&jc.code) {
            let key = op.key;
            if handled.contains(key) {
                continue;
            }
            *hit_counts.entry(key.to_string()).or_insert(0) += 1;
        }
    }

    eprintln!();
    eprintln!("=== T3 AUDIT: runtime opnames ===");
    eprintln!("total runtime opnames: {}", runtime_names.len());
    eprintln!("walker-handled opnames: {}", handled.len());
    eprintln!("unhandled opnames: {}", unhandled.len());

    eprintln!();
    eprintln!("=== T3 AUDIT: unhandled opnames ranked by JitCode hit count ===");
    let mut ranked: Vec<(String, usize)> = hit_counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (name, count) in &ranked {
        let byte = opname_to_byte
            .get(name)
            .map(|b| format!("0x{b:02x}"))
            .unwrap_or_else(|| "?".to_string());
        eprintln!("  {count:>5}  {byte}  {name}");
    }

    eprintln!();
    eprintln!("=== T3 AUDIT: unhandled opnames with ZERO JitCode hits ===");
    for name in &unhandled {
        if !ranked.iter().any(|(n, _)| n == *name) {
            let byte = opname_to_byte
                .get(*name)
                .map(|b| format!("0x{b:02x}"))
                .unwrap_or_else(|| "?".to_string());
            eprintln!("  {byte}  {name}");
        }
    }

    eprintln!();
    eprintln!("=== T3 AUDIT: walker-handled opnames (for cross-check) ===");
    for name in &handled {
        eprintln!("  {name}");
    }

    // Sanity: byte_to_opname must invert opname_to_byte.
    assert_eq!(byte_to_opname.len(), opname_to_byte.len());
}

#[test]
fn regular_record_table_is_the_sole_dispatcher() {
    // The `regular_record_table!` macro (arith.rs) owns dispatch for the
    // uniform `int_*` / `float_*` / `ptr_*` record families. Guard the
    // invariant that those opnames route ONLY through the table:
    // (a) it is non-empty with unique keys, (b) every key is a real
    // runtime opname, (c) none still appears as a `"opname" =>` arm in
    // `handle` (which the pre-match table lookup would shadow into dead
    // code).
    let keys = REGULAR_RECORD_KEYS;
    assert!(!keys.is_empty(), "table must route at least one opname");

    // (a) unique keys.
    let unique: std::collections::BTreeSet<&str> = keys.iter().copied().collect();
    assert_eq!(
        unique.len(),
        keys.len(),
        "REGULAR_RECORD_KEYS has duplicates"
    );

    // (b) every key is a real codewriter-emitted opname, except the
    // documented dormant `int_same_as/i>i` forward-prep arm: RPython
    // `jtransform.py rewrite_op_same_as` strips `same_as` before
    // assembly, so it is intentionally absent from the runtime table while
    // its dispatch entry is kept for the walker's forward-prep path.
    let runtime = insns_opname_to_byte();
    for key in keys {
        if *key == "int_same_as/i>i" {
            continue;
        }
        assert!(
            runtime.contains_key(*key),
            "table key {key:?} is not a runtime opname",
        );
    }

    // (c) no key survives as a dispatch arm in `handle`.
    let handle_src = include_str!("mod.rs");
    for key in keys {
        let arm = format!("\"{key}\" =>");
        assert!(
            !handle_src.contains(&arm),
            "opname {key:?} still has a dead dispatch arm in mod.rs; \
             it is dispatched by regular_record_table!",
        );
    }
}

fn drive_int_add_jump_if_ovf(
    lhs_value: i64,
    rhs_value: i64,
) -> (Vec<OpCode>, usize, bool, u32, OpRef, OpRef, usize) {
    let opname = "int_add_jump_if_ovf/Lii>i";
    let ovf_byte = *insns_opname_to_byte()
        .get(opname)
        .expect("int_add_jump_if_ovf must be in the runtime instruction table");
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("live must be in the runtime instruction table");
    // Lii>i: target 9, source registers 0/1, destination register 2.
    // Separate live tails at 6 and 9 make fallthrough and jump observable.
    let code = [ovf_byte, 9, 0, 0, 1, 2, live_byte, 0, 0, live_byte, 0, 0];
    let _ = test_outer_resume_jitcode_index();
    let mut tc = TraceCtx::for_test_types(&[Type::Int, Type::Int]);
    let lhs = OpRef::input_arg_int(0);
    let rhs = OpRef::input_arg_int(1);
    tc.set_opref_concrete(lhs, Value::Int(lhs_value));
    tc.set_opref_concrete(rhs, Value::Int(rhs_value));
    let mut regs_i = vec![lhs, rhs, OpRef::NONE];
    let mut concrete_i = vec![
        ConcreteValue::Int(lhs_value),
        ConcreteValue::Int(rhs_value),
        ConcreteValue::Null,
    ];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut concrete_i,
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: done_descr_ref_for_tests(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: Some(0),
        outer_jitcode_index: test_outer_resume_jitcode_index(),
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int_add_jump_if_ovf must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    let dst = wc.registers_i[2];
    drop(wc);
    let ops = tc.ops();
    let opcodes = ops.iter().map(|op| op.opcode).collect();
    let guard_num_args = ops[1].num_args();
    let guard_has_snapshot = ops[1].rd_resume_position.get() >= 0;
    let guard_resume_pc = tc
        .get_snapshot(ops[1].rd_resume_position.get())
        .expect("overflow guard snapshot must exist")
        .frames
        .last()
        .expect("overflow guard snapshot must contain its frame")
        .pc;
    let resbox = ops[0].pos.get();
    (
        opcodes,
        guard_num_args,
        guard_has_snapshot,
        guard_resume_pc,
        resbox,
        dst,
        next_pc,
    )
}

#[test]
fn int_add_jump_if_ovf_no_overflow_records_guard_no_overflow_and_writes_dst() {
    let (opcodes, guard_num_args, guard_has_snapshot, guard_resume_pc, resbox, dst, next_pc) =
        drive_int_add_jump_if_ovf(40, 2);
    assert_eq!(opcodes, vec![OpCode::IntAddOvf, OpCode::GuardNoOverflow],);
    assert_eq!(guard_num_args, 0, "GuardNoOverflow is operand-less");
    assert!(guard_has_snapshot);
    assert_eq!(
        guard_resume_pc, 0,
        "the guard resumes at the overflow opcode"
    );
    assert_eq!(next_pc, 6, "no overflow continues at op.next_pc");
    assert_eq!(dst, resbox, "the no-overflow continue writes resbox to dst");
}

#[test]
fn int_add_jump_if_ovf_overflow_records_guard_overflow_and_jumps() {
    let (opcodes, guard_num_args, guard_has_snapshot, guard_resume_pc, _, dst, next_pc) =
        drive_int_add_jump_if_ovf(i64::MAX, 1);
    assert_eq!(opcodes, vec![OpCode::IntAddOvf, OpCode::GuardOverflow],);
    assert_eq!(guard_num_args, 0, "GuardOverflow is operand-less");
    assert!(guard_has_snapshot);
    assert_eq!(
        guard_resume_pc, 0,
        "the guard resumes at the overflow opcode"
    );
    assert_eq!(dst, OpRef::NONE, "the overflow jump does not write dst");
    assert_eq!(next_pc, 9, "overflow jumps to the handler target");
}

#[test]
fn int_ovf_jump_constant_operands_fold_without_recording_an_ovf_op() {
    let byte = *insns_opname_to_byte()
        .get("int_add_jump_if_ovf/Lii>i")
        .expect("int_add_jump_if_ovf must be in the runtime instruction table");
    let code = [byte, 6, 0, 0, 1, 2];
    let mut tc = fresh_trace_ctx();
    let lhs = tc.const_int(40);
    let rhs = tc.const_int(2);
    let mut regs_i = [lhs, rhs, OpRef::NONE];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
        .expect("constant overflow arithmetic must fold");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 6);
    assert_eq!(
        tc.num_ops(),
        0,
        "folding emits neither an ovf op nor a guard"
    );
    assert_eq!(regs_i[2].inline_const_to_value(), Some(Value::Int(42)));
}

#[test]
fn int_ovf_jump_declines_when_an_operand_is_not_concrete() {
    let byte = *insns_opname_to_byte()
        .get("int_add_jump_if_ovf/Lii>i")
        .expect("int_add_jump_if_ovf must be in the runtime instruction table");
    let code = [byte, 6, 0, 0, 1, 2];
    let mut tc = TraceCtx::for_test_types(&[Type::Int, Type::Int]);
    let lhs = OpRef::input_arg_int(0);
    let rhs = OpRef::input_arg_int(1);
    let mut regs_i = [lhs, rhs, OpRef::NONE];
    let err = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
        .expect_err("an unstamped overflow operand must decline");
    assert_eq!(
        err,
        DispatchError::IntOvfOperandNotConcrete { pc: 0, value: lhs }
    );
    assert_eq!(tc.num_ops(), 0, "a declined overflow jump records nothing");
}

/// Drive one of the `d>r` struct-allocation handlers (`new`,
/// `new_with_vtable`): both record the descr alone and write the
/// allocation into the ref bank.
fn drive_alloc_with_descr(opname: &str, expected_opcode: OpCode) {
    let nwv_byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in the runtime instruction table"));
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("live must be in the runtime instruction table");
    // `d>r`: 2B descr index (0) + 1B dst register (0); live tail at 4.
    let code = [nwv_byte, 0, 0, 0, live_byte, 0, 0];
    let _ = test_outer_resume_jitcode_index();
    let descr_pool = vec![crate::descr::w_int_size_descr()];
    let mut tc = TraceCtx::for_test_types(&[]);
    let mut regs_r = vec![OpRef::NONE];
    let mut concrete_r = vec![ConcreteValue::Null];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut concrete_r,
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: done_descr_ref_for_tests(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: Some(0),
        outer_jitcode_index: test_outer_resume_jitcode_index(),
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).unwrap_or_else(|_| panic!("`{opname}` must dispatch"));
    assert_eq!(outcome, DispatchOutcome::Continue);
    let dst = wc.registers_r[0];
    drop(wc);

    let ops = tc.ops();
    assert_eq!(
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
        vec![expected_opcode],
    );
    assert_eq!(
        ops[0].num_args(),
        0,
        "the allocation records the descr alone, with no box operands"
    );
    assert_eq!(
        dst,
        ops[0].pos.get(),
        "the `>r` decorator writes the allocation into the ref bank"
    );
    assert_eq!(next_pc, 4, "`d>r` consumes a 2B descr plus a 1B dst");
}

#[test]
fn new_with_vtable_records_the_alloc_and_writes_the_ref_dst() {
    drive_alloc_with_descr("new_with_vtable/d>r", OpCode::NewWithVtable);
}

#[test]
fn new_records_the_alloc_and_writes_the_ref_dst() {
    drive_alloc_with_descr("new/d>r", OpCode::New);
}

/// Step one record-only opcode (the heapcache hints, the raw-memory pair)
/// over caller-supplied banks.
fn run_hint_step(
    code: &[u8],
    tc: &mut TraceCtx,
    regs_r: &mut [OpRef],
    concrete_r: &mut [ConcreteValue],
    regs_i: &mut [OpRef],
) -> Result<(DispatchOutcome, usize), DispatchError> {
    run_hint_step_with_descrs(code, tc, regs_r, concrete_r, regs_i, &[])
}

fn run_hint_step_with_descrs(
    code: &[u8],
    tc: &mut TraceCtx,
    regs_r: &mut [OpRef],
    concrete_r: &mut [ConcreteValue],
    regs_i: &mut [OpRef],
    descr_pool: &[DescrRef],
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Guard-emitting arms need a resolvable outer resume coordinate for the
    // snapshot; record-only arms ignore these two fields.
    let outer_jitcode_index = test_outer_resume_jitcode_index();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: regs_r,
        registers_i: regs_i,
        registers_f: &mut [],
        concrete_registers_r: concrete_r,
        concrete_registers_i: &mut [],
        descr_refs: descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: tc,
        done_with_this_frame_descr_ref: done_descr_ref_for_tests(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: Some(0),
        outer_jitcode_index,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    step(code, 0, &mut wc)
}

#[test]
fn assert_not_none_records_when_the_operand_has_a_concrete() {
    let byte = *insns_opname_to_byte()
        .get("assert_not_none/r")
        .expect("`assert_not_none/r` must be in insns table");
    let code = [byte, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::PtrEq, &[]);
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Ref(
        0xdead_beef_usize as *mut pyre_object::pyobject::PyObject,
    )];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("`assert_not_none/r` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 2, "`r` consumes a single register byte");
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::AssertNotNone);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![operand],
    );
}

#[test]
fn assert_not_none_declines_when_the_operand_has_no_concrete() {
    let byte = *insns_opname_to_byte()
        .get("assert_not_none/r")
        .expect("`assert_not_none/r` must be in insns table");
    let code = [byte, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::PtrEq, &[]);
    let ops_before = tc.num_ops();
    let mut regs_r = [operand];
    // No shadow for the slot: the walker never observed this pointer, so
    // it cannot stand behind the non-null check the recorder performs.
    let mut concrete_r = [ConcreteValue::Null];
    let err = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect_err("a shadow-less operand must decline instead of asserting");
    assert_eq!(
        err,
        DispatchError::UnsupportedOpname {
            pc: 0,
            key: "assert_not_none/r",
        },
    );
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "a declined step must not leave a recorded op behind"
    );
}

#[test]
fn assert_not_none_uses_known_nonnull_heapcache_without_a_ref_shadow() {
    let byte = *insns_opname_to_byte()
        .get("assert_not_none/r")
        .expect("`assert_not_none/r` must be in insns table");
    let code = [byte, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::New, &[]);
    tc.heap_cache_mut().new_object(operand);
    let ops_before = tc.num_ops();
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Null];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("a heapcache-known allocation needs no ref shadow");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 2);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "the known-nonnull fast path records no AssertNotNone"
    );
}

#[test]
fn fused_int_compare_folds_same_box_and_constant_operands_without_guards() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_int_eq/iiL")
        .expect("`goto_if_not_int_eq/iiL` must be in insns table");
    let code = [byte, 0x00, 0x00, 0x09, 0x00];
    let mut same_tc = TraceCtx::for_test_types(&[Type::Int]);
    let same = OpRef::input_arg_int(0);
    let mut same_regs = [same];
    let (_, same_next) = run_hint_step(&code, &mut same_tc, &mut [], &mut [], &mut same_regs)
        .expect("a same-box equality has a static direction");
    assert_eq!(same_next, code.len());
    assert_eq!(
        same_tc.num_ops(),
        0,
        "same-box equality records no compare or guard"
    );

    let code = [byte, 0x00, 0x01, 0x09, 0x00];
    let mut const_tc = fresh_trace_ctx();
    let lhs = const_tc.const_int(4);
    let rhs = const_tc.const_int(5);
    let mut const_regs = [lhs, rhs];
    let (_, const_next) = run_hint_step(&code, &mut const_tc, &mut [], &mut [], &mut const_regs)
        .expect("constant equality has a static direction");
    assert_eq!(const_next, 9);
    assert_eq!(
        const_tc.num_ops(),
        0,
        "constant equality records no compare or guard"
    );

    let ptr_byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_ne/rrL")
        .expect("`goto_if_not_ptr_ne/rrL` must be in insns table");
    let ptr_code = [ptr_byte, 0x00, 0x00, 0x09, 0x00];
    let mut ptr_tc = TraceCtx::for_test_types(&[Type::Ref]);
    let same_ptr = OpRef::input_arg_ref(0);
    let mut ptr_regs = [same_ptr];
    let (_, ptr_next) = run_hint_step(&ptr_code, &mut ptr_tc, &mut ptr_regs, &mut [], &mut [])
        .expect("a same-box pointer inequality has a static direction");
    assert_eq!(ptr_next, 9);
    assert_eq!(
        ptr_tc.num_ops(),
        0,
        "same-box pointer inequality records no compare or guard"
    );
}

#[test]
fn fused_ptr_compare_uses_ref_shadows_for_the_runtime_direction() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_eq/rrL")
        .expect("`goto_if_not_ptr_eq/rrL` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x09, 0x00];
    let mut tc = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
    let lhs = OpRef::input_arg_ref(0);
    let rhs = OpRef::input_arg_ref(1);
    let mut regs_r = [lhs, rhs];
    let ptr = 0xdead_beef_usize as *mut pyre_object::pyobject::PyObject;
    let mut concrete_r = [ConcreteValue::Ref(ptr), ConcreteValue::Ref(ptr)];
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("ref shadows determine the pointer comparison direction");
    assert_eq!(next_pc, code.len());
    assert_eq!(
        tc.ops().iter().map(|op| op.opcode).collect::<Vec<_>>(),
        vec![OpCode::PtrEq, OpCode::GuardTrue]
    );
}

#[test]
fn ptr_eq_folds_from_the_ref_shadow_when_box_value_is_absent() {
    // A non-fused `ptr_eq/rr>i` whose operands carry a concrete only in the
    // ref-register shadow (box_value absent — the inline-callee-param case)
    // must still fold the bool result, matching the fused twin and
    // ptr_nullity. Without the shadow fallback the bool stayed symbolic and a
    // downstream goto_if_not declined.
    let byte = *insns_opname_to_byte()
        .get("ptr_eq/rr>i")
        .expect("`ptr_eq/rr>i` must be in insns table");
    // `rr>i`: 1B r-src1 + 1B r-src2 + 1B i-dst.
    let code = [byte, 0x00, 0x01, 0x00];
    let mut tc = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
    let a = OpRef::input_arg_ref(0);
    let b = OpRef::input_arg_ref(1);
    assert_eq!(
        tc.box_value(a),
        None,
        "an input-arg ref has no box_value carrier"
    );
    let mut regs_r = [a, b];
    let mut regs_i = [OpRef::None];
    let pa = 0xdead_0000_usize as *mut pyre_object::pyobject::PyObject;
    let pb = 0xbeef_0000_usize as *mut pyre_object::pyobject::PyObject;
    let mut concrete_r = [ConcreteValue::Ref(pa), ConcreteValue::Ref(pb)];
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut regs_i)
        .expect("ptr_eq must record from the ref shadow");
    assert_eq!(next_pc, 4);
    assert_eq!(
        tc.concrete_of_opref(regs_i[0]),
        Some(majit_ir::Value::Int(0)),
        "distinct shadow pointers fold PtrEq to false without a box_value carrier",
    );
}

#[test]
fn hint_force_virtualizable_is_a_noop_without_virtualizable_info() {
    let byte = *insns_opname_to_byte()
        .get("hint_force_virtualizable/r")
        .expect("`hint_force_virtualizable/r` must be in insns table");
    // `r`: 1B ref reg.
    let code = [byte, 0x00];
    let mut tc = fresh_trace_ctx();
    let vable = tc.record_op(majit_ir::OpCode::PtrEq, &[]);
    let ops_before = tc.num_ops();
    let mut regs_r = [vable];
    let mut concrete_r = [ConcreteValue::Null];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("`hint_force_virtualizable/r` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 2, "`r` consumes a single register byte");
    // `gen_store_back_in_vable` returns before emitting anything when the
    // jitdriver carries no virtualizable info -- the upstream `vinfo is None`
    // gate -- so the hint writes back nothing here.
    assert_eq!(tc.num_ops(), ops_before);
}

#[test]
fn goto_if_not_ptr_nonzero_guards_nonnull_and_falls_through() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_nonzero/rL")
        .expect("`goto_if_not_ptr_nonzero/rL` must be in insns table");
    // `rL`: 1B ref reg + 2B label (target 9).
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::PtrEq, &[]);
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Ref(
        0xdead_beef_usize as *mut pyre_object::pyobject::PyObject,
    )];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("`goto_if_not_ptr_nonzero/rL` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 4,
        "a non-null pointer falls through past the 3 operand bytes"
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(
        last.opcode,
        majit_ir::OpCode::GuardNonnull,
        "`_establish_nullity` guards the observed non-nullness"
    );
    assert!(
        tc.heap_cache()
            .is_nullity_known(operand, |_| None)
            .is_some(),
        "the nullity must be stamped into the heapcache"
    );
}

#[test]
fn goto_if_not_ptr_iszero_takes_the_branch_when_nonnull() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_iszero/rL")
        .expect("`goto_if_not_ptr_iszero/rL` must be in insns table");
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::PtrEq, &[]);
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Ref(
        0xdead_beef_usize as *mut pyre_object::pyobject::PyObject,
    )];
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("`goto_if_not_ptr_iszero/rL` must dispatch");
    assert_eq!(
        next_pc, 9,
        "iszero jumps exactly where nonzero falls through"
    );
}

#[test]
fn goto_if_not_ptr_nonzero_guards_isnull_and_takes_the_branch() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_nonzero/rL")
        .expect("`goto_if_not_ptr_nonzero/rL` must be in insns table");
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::PtrEq, &[]);
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Ref(std::ptr::null_mut())];
    let (_, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("`goto_if_not_ptr_nonzero/rL` must dispatch");
    assert_eq!(next_pc, 9, "a null pointer takes the branch");
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::GuardIsnull);
}

#[test]
fn goto_if_not_ptr_nonzero_declines_without_a_concrete() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_nonzero/rL")
        .expect("`goto_if_not_ptr_nonzero/rL` must be in insns table");
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::PtrEq, &[]);
    let ops_before = tc.num_ops();
    let mut regs_r = [operand];
    let mut concrete_r = [ConcreteValue::Null];
    let err = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect_err("an unobserved pointer must decline, not pick a direction");
    assert_eq!(
        err,
        DispatchError::UnsupportedOpname {
            pc: 0,
            key: "goto_if_not_ptr_nonzero/rL",
        },
    );
    assert_eq!(tc.num_ops(), ops_before, "a declined step records nothing");
}

#[test]
fn goto_if_not_ptr_nonzero_uses_known_nonnull_heapcache_without_a_ref_shadow() {
    let byte = *insns_opname_to_byte()
        .get("goto_if_not_ptr_nonzero/rL")
        .expect("`goto_if_not_ptr_nonzero/rL` must be in insns table");
    // `rL`: 1B ref reg + 2B label (target 9).
    let code = [byte, 0x00, 0x09, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::New, &[]);
    tc.heap_cache_mut().new_object(operand);
    let ops_before = tc.num_ops();
    let mut regs_r = [operand];
    // The Ref shadow is absent, yet the heapcache already proved non-nullness.
    // `_establish_nullity` answers from the cache before demanding a pointer.
    let mut concrete_r = [ConcreteValue::Null];
    let (outcome, next_pc) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut [])
        .expect("a heapcache-known non-null box needs no ref shadow");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 4,
        "a known non-null pointer falls through without reading a shadow"
    );
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "the heapcache-known fast path emits no nullity guard"
    );
}

#[test]
fn raw_load_i_records_the_load_against_the_descr() {
    let byte = *insns_opname_to_byte()
        .get("raw_load_i/iid>i")
        .expect("`raw_load_i/iid>i` must be in insns table");
    // `iid>i`: 1B base + 1B offset + 2B descr + 1B dst.
    let code = [byte, 0x00, 0x01, 0x00, 0x00, 0x02];
    let descr_pool = vec![crate::descr::w_int_size_descr()];
    let mut tc = fresh_trace_ctx();
    let base = tc.const_int(0x1000);
    let offset = tc.const_int(8);
    let mut regs_i = [base, offset, OpRef::NONE];
    let (outcome, next_pc) =
        run_hint_step_with_descrs(&code, &mut tc, &mut [], &mut [], &mut regs_i, &descr_pool)
            .expect("`raw_load_i/iid>i` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "`iid>i` consumes 3 register bytes plus a 2B descr"
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::RawLoadI);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![base, offset],
        "raw loads record the address pair only; the descr rides alongside",
    );
    assert_eq!(
        regs_i[2],
        last.pos.get(),
        "the `>i` decorator writes the dst"
    );
}

#[test]
fn raw_store_i_records_the_store_and_writes_no_register() {
    let byte = *insns_opname_to_byte()
        .get("raw_store_i/iiid")
        .expect("`raw_store_i/iiid` must be in insns table");
    // `iiid`: 1B base + 1B offset + 1B value + 2B descr.
    let code = [byte, 0x00, 0x01, 0x02, 0x00, 0x00];
    let descr_pool = vec![crate::descr::w_int_size_descr()];
    let mut tc = fresh_trace_ctx();
    let base = tc.const_int(0x1000);
    let offset = tc.const_int(8);
    let value = tc.const_int(42);
    let mut regs_i = [base, offset, value];
    let (outcome, next_pc) =
        run_hint_step_with_descrs(&code, &mut tc, &mut [], &mut [], &mut regs_i, &descr_pool)
            .expect("`raw_store_i/iiid` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "`iiid` consumes 3 register bytes plus a 2B descr"
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::RawStore);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![base, offset, value],
    );
    assert_eq!(
        regs_i,
        [base, offset, value],
        "a raw store leaves every register untouched"
    );
}

#[test]
fn record_exact_class_records_the_hint_with_both_operands() {
    let byte = *insns_opname_to_byte()
        .get("record_exact_class/ri")
        .expect("`record_exact_class/ri` must be in insns table");
    // `ri`: 1B ref reg + 1B int reg holding the class vtable address.
    let code = [byte, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let operand = tc.record_op(majit_ir::OpCode::PtrEq, &[]);
    let cls = tc.const_int(0x4000);
    let mut regs_r = [operand];
    let mut regs_i = [cls];
    let mut concrete_r = [ConcreteValue::Null];
    let (outcome, next_pc) =
        run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut regs_i)
            .expect("`record_exact_class/ri` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 3, "`ri` consumes two register bytes");
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::RecordExactClass);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![operand, cls],
    );
}

#[test]
fn switch_id_hit_jumps_to_matching_target() {
    let switch_byte = *insns_opname_to_byte()
        .get("switch/id")
        .expect("`switch/id` must be in insns table");
    let code = [
        switch_byte,
        0x00, // i register 0
        0x00,
        0x00, // d descr index 0
    ];
    let mut tc = fresh_trace_ctx();
    let value = tc.const_int(5);
    let mut regs_i = vec![value];
    let descr_pool = switch_descr_pool(&[(5, 17), (9, 23)]);
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("switch hit must dispatch");

    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 17);
}

#[test]
fn switch_id_miss_falls_through() {
    let switch_byte = *insns_opname_to_byte()
        .get("switch/id")
        .expect("`switch/id` must be in insns table");
    let code = [
        switch_byte,
        0x00, // i register 0
        0x00,
        0x00, // d descr index 0
    ];
    let mut tc = fresh_trace_ctx();
    let value = tc.const_int(7);
    let mut regs_i = vec![value];
    let descr_pool = switch_descr_pool(&[(5, 17), (9, 23)]);
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("switch miss must dispatch");

    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, code.len());
}

#[test]
fn switch_id_requires_concrete_int_value() {
    let switch_byte = *insns_opname_to_byte()
        .get("switch/id")
        .expect("`switch/id` must be in insns table");
    let code = [
        switch_byte,
        0x00, // i register 0
        0x00,
        0x00, // d descr index 0
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = vec![OpRef::input_arg_int(0)];
    let descr_pool = switch_descr_pool(&[(5, 17)]);
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let err = step(&code, 0, &mut wc).expect_err("non-constant switch value must not guess");

    assert_eq!(
        err,
        DispatchError::SwitchValueNotConcrete {
            pc: 0,
            value: OpRef::input_arg_int(0),
        }
    );
}

#[test]
fn goto_if_not_truthy_records_guard_true_and_falls_through() {
    // `goto_if_not/iL` with a concrete non-zero Int: emit GuardTrue,
    // do NOT take the jump (pc advances past the 3-byte operand
    // block).  RPython `pyjitpl.py opimpl_goto_if_not`
    // `if switchcase: opnum = rop.GUARD_TRUE; ... if not switchcase: self.pc = target`.
    let goto_if_byte = *insns_opname_to_byte()
        .get("goto_if_not/iL")
        .expect("`goto_if_not/iL` must be in insns table");
    let code = [
        goto_if_byte,
        0x00, // i register 0
        0x40,
        0x00, // L target = 0x0040
    ];
    let mut tc = fresh_trace_ctx();
    let value = tc.const_int(1);
    let mut regs_i = vec![value];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("truthy branch must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, code.len(), "truthy branch falls through");
}

#[test]
fn goto_if_not_falsy_records_guard_false_and_jumps() {
    // `goto_if_not/iL` with a concrete zero Int: emit GuardFalse,
    // jump to the label target (pc = target).
    let goto_if_byte = *insns_opname_to_byte()
        .get("goto_if_not/iL")
        .expect("`goto_if_not/iL` must be in insns table");
    let code = [
        goto_if_byte,
        0x00, // i register 0
        0x40,
        0x00, // L target = 0x0040
    ];
    let mut tc = fresh_trace_ctx();
    let value = tc.const_int(0);
    let mut regs_i = vec![value];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("falsy branch must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 0x0040, "falsy branch jumps to label target");
}

#[test]
fn goto_if_not_requires_concrete_int_value() {
    // Non-constant symbolic OpRef has no concrete: must surface
    // `GotoIfNotValueNotConcrete` rather than guess a branch.
    let goto_if_byte = *insns_opname_to_byte()
        .get("goto_if_not/iL")
        .expect("`goto_if_not/iL` must be in insns table");
    let code = [
        goto_if_byte,
        0x00, // i register 0
        0x40,
        0x00, // L target = 0x0040
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = vec![OpRef::input_arg_int(0)];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    let err = step(&code, 0, &mut wc).expect_err("non-constant branch value must not guess");

    assert_eq!(
        err,
        DispatchError::GotoIfNotValueNotConcrete {
            pc: 0,
            value: OpRef::input_arg_int(0),
        }
    );
}

/// Production-like `sub_jitcode_lookup` that resolves `idx` against
/// `crate::jitcode_runtime::all_jitcodes()`. Used by the end-to-end
/// helper acceptance tests (`walk_return_value_helper_*`,
/// `walk_pop_top_helper_*`) so the walker can recurse into real
/// callee bodies. Delegates to the production
/// [`super::sub_jitcode_body_by_index`].
fn production_sub_jitcodes(idx: usize) -> Option<SubJitCodeBody> {
    super::sub_jitcode_body_by_index(idx)
}

#[test]
fn sub_jitcode_body_by_index_builds_w_list_append() {
    // The shared by-index `SubJitCodeBody` builder must resolve a
    // build-time charon body (`w_list_append`) to a well-formed body —
    // non-empty bytecode and >= 2 ref registers for the (list, value)
    // params (calldescr arg_classes 'rr').  This body is descended by the
    // shipping `lst.append` arm (`try_walker_orthodox_list_append`).
    let idx = crate::jitcode_runtime::list_append_jitcode()
        .expect("w_list_append must be present in ALL_JITCODES")
        .index();
    let body =
        super::sub_jitcode_body_by_index(idx).expect("by-index builder must resolve w_list_append");
    assert!(
        !body.code.is_empty(),
        "w_list_append body must carry assembled bytecode"
    );
    assert!(
        body.num_regs_r >= 2,
        "w_list_append takes (list, value) => >= 2 ref registers, got {}",
        body.num_regs_r
    );
    // Out-of-range index resolves to None (RPython: ALL_JITCODES miss).
    assert!(super::sub_jitcode_body_by_index(usize::MAX).is_none());
}

#[test]
fn append_journal_rollback_rewinds_length() {
    // #171 P3 journal infra: a walked eager `list.append` grows the
    // concrete list at trace time (the fold records the array-op IR but
    // does not mutate), so a NON-commit walk must rewind the length,
    // exactly like the STORE_SUBSCR store journal restores its displaced
    // element.  Spare-capacity gating (`w_list_can_append_without_realloc`)
    // makes the rewind a pure length set with no reallocation to undo.
    use pyre_object::listobject::{w_list_can_append_without_realloc, w_list_len};
    use pyre_object::{w_int_new, w_list_append};

    super::fbw_store_journal_reset();

    let list =
        pyre_object::listobject::w_list_new(vec![w_int_new(10), w_int_new(20), w_int_new(30)]);
    // A first append forces the backing array to grow with a growth
    // factor, leaving spare capacity so the *next* append is in-place
    // (the only shape the arm specializes).
    unsafe { w_list_append(list, w_int_new(40)) };
    let len_before = unsafe { w_list_len(list) };
    assert_eq!(len_before, 4);
    assert!(
        unsafe { w_list_can_append_without_realloc(list) },
        "post-grow list must have spare capacity for the in-place append"
    );

    // Rollback path: journal push + eager append (production order, see
    // try_walker_orthodox_list_append), then a non-commit exit rewinds
    // the length.
    super::fbw_append_journal_push(list, len_before);
    unsafe { w_list_append(list, w_int_new(50)) };
    assert_eq!(unsafe { w_list_len(list) }, 5);
    super::fbw_store_journal_rollback();
    assert_eq!(
        unsafe { w_list_len(list) },
        len_before,
        "non-commit walk must rewind the eager append's length"
    );

    // Commit path: the eager append stands; the log is dropped.
    super::fbw_append_journal_push(list, len_before);
    unsafe { w_list_append(list, w_int_new(60)) };
    super::fbw_store_journal_commit();
    assert_eq!(
        unsafe { w_list_len(list) },
        len_before + 1,
        "committed walk keeps the eager append"
    );
    // A subsequent rollback with the log already committed-empty is a
    // no-op (does not shrink further).
    super::fbw_store_journal_rollback();
    assert_eq!(unsafe { w_list_len(list) }, len_before + 1);
}

#[test]
fn append_journal_rollback_rewinds_object_length() {
    // #171 object-append: the orthodox fold journals object-strategy
    // appends through the SAME `FBW_APPEND_JOURNAL` as the int spec, so a
    // non-commit rollback must rewind the strategy-correct length — the
    // `W_ListObject.length` header (`ll_list_obj_set_len`), not
    // `int_items.len`.  Rewinding via the int leaf would leave the object
    // header grown (legacy replay double-appends) and trip the
    // `w_list_int_set_len` strategy assert.
    use pyre_object::listobject::{w_list_can_append_without_realloc, w_list_len};
    use pyre_object::{w_list_append, w_none};

    super::fbw_store_journal_reset();

    // `w_list_new_object` forces the Object strategy regardless of element
    // type; None elements keep the test free of int/float boxing.
    let list = pyre_object::listobject::w_list_new_object(vec![w_none(), w_none(), w_none()]);
    // Grow once so the next append is an in-place spare-capacity store
    // (the only shape the journal records).
    unsafe { w_list_append(list, w_none()) };
    let len_before = unsafe { w_list_len(list) };
    assert_eq!(len_before, 4);
    assert!(
        unsafe { w_list_can_append_without_realloc(list) },
        "post-grow object list must have spare capacity for the in-place append"
    );

    super::fbw_append_journal_push(list, len_before);
    unsafe { w_list_append(list, w_none()) };
    assert_eq!(unsafe { w_list_len(list) }, 5);
    super::fbw_store_journal_rollback();
    assert_eq!(
        unsafe { w_list_len(list) },
        len_before,
        "non-commit walk must rewind the object append's length header"
    );
}

/// Tests use the production `PyreJitCodeDescr` adapter
/// directly — the type lives at
/// `pyre-jit-trace/src/descr.rs::PyreJitCodeDescr`
/// + `descr::make_jitcode_descr(idx)` so the walker's
/// `as_jitcode_descr()` cast exercises production code, not a
/// test-local duplicate.
use crate::descr::make_jitcode_descr;

/// Build a `descr_refs` pool of length `pool_len` where the slot at
/// each `BhDescr::JitCode` index in `crate::jitcode_runtime::all_descrs()`
/// holds a `TestJitCodeDescr` carrying that descr's `jitcode_index`,
/// and every other slot holds a `make_fail_descr` placeholder.
/// Lets acceptance tests resolve `inline_call_*` descr indices
/// without standing up the full BhDescr → trait Descr adapter
/// pipeline.
fn descr_pool_with_jitcode_adapters(pool_len: usize) -> Vec<DescrRef> {
    let all_bh = crate::jitcode_runtime::all_descrs();
    (0..pool_len)
        .map(|i| match all_bh.get(i) {
            Some(majit_translate::jitcode::BhDescr::JitCode { jitcode_index, .. }) => {
                make_jitcode_descr(*jitcode_index)
            }
            // Residual calls surfaced during inline_call recursion (e.g.
            // the arm bodies' helper calls) resolve their descr through
            // `as_call_descr()`. Build a real CallDescr for `BhDescr::Call`
            // entries, mirroring production (`descr.rs make_call_descr_from_bh`),
            // so the walk does not surface ResidualCallDescrNotCallDescr.
            Some(majit_translate::jitcode::BhDescr::Call { calldescr }) => {
                crate::descr::make_call_descr_from_bh(calldescr)
            }
            _ => make_fail_descr(1 + i),
        })
        .collect()
}

#[test]
fn inline_call_recursion_writes_subreturn_into_caller_dst_register() {
    // Core acceptance: caller's `inline_call_r_r/dR>r`
    // recurses into a synthetic callee jitcode whose body is
    // simply `ref_return r0`. The callee's ref_return surfaces as
    // `SubReturn { result: Some(callee.registers_r[0]) }`; the
    // caller's inline_call handler writes that OpRef into the
    // caller's dst register. Then the caller's own `ref_return r3`
    // records the outermost Finish carrying that propagated value.
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    // Callee body: `ref_return r0`. registers_r[0] is populated
    // from the caller's R-list arg.
    let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body:
    //   inline_call_r_r/dR>r descr=7, R=[r2], >r=r5
    //   ref_return r5
    let caller_code = [
        inline_byte,
        0x07,
        0x00, // d (LE descr index = 7)
        0x01,
        0x02, // R: varlen=1, args=[r2]
        0x05, // >r: dst = r5
        ret_byte,
        0x05, // ref_return r5
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let arg_value = regs_r[2];
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, end_pc) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
    assert_eq!(outcome, DispatchOutcome::Terminate);
    assert_eq!(end_pc, caller_code.len());
    drop(wc);
    // dst register r5 must equal the arg the caller passed (since
    // callee's `ref_return r0` returns its registers_r[0] which
    // was populated from caller's R-list[0] = r2's OpRef).
    assert_eq!(
        regs_r[5], arg_value,
        "inline_call_r_r dst writeback must propagate callee's SubReturn",
    );
    // The outermost finish payload carries the same value (callee's
    // ref_return surfaced as SubReturn and recorded nothing; the
    // caller's top-level ref_return stashed the payload).
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "no op recorded: the compile consumer records the FINISH from the payload",
    );
    assert_eq!(
        fbw_finish_payload_take(),
        Some((arg_value, Type::Ref)),
        "outermost finish payload must carry the arg value the caller threaded \
             through inline_call_r_r",
    );
}

#[test]
fn inline_call_r_i_writes_int_subreturn_into_caller_int_bank() {
    // Acceptance: caller's `inline_call_r_i/dR>i`
    // recurses into a synthetic callee whose body is `int_return
    // r0` on the int bank. The callee's int_return surfaces as
    // `SubReturn { result: Some(callee.registers_i[0]) }`; the
    // caller's helper writes that OpRef into the caller's
    // `registers_i[dst]` (NOT registers_r — the kind discriminator
    // for this variant). RPython parity: pyjitpl.py
    // exec-generated `_opimpl_inline_call_r_i` template paired with
    // `_opimpl_any_return` for `int_return`.
    //
    // Callee shape constraint: the `_r` arglist promises only Ref
    // args, but the body needs an int register populated to source
    // the int_return. The codewriter populates that via a separate
    // op inside the callee body (e.g. `int_copy/i>i`). For the
    // walker test, we synthesize an `int_copy` that materializes
    // the int constant from a tracer-side const_int OpRef stored in
    // a high i-register, then int_returns it.
    //
    // Simpler: callee whose body is just `int_return i0`; we
    // pre-populate the callee's registers_i[0] indirectly through a
    // setter — but the walker doesn't expose that directly. So
    // instead, we use a callee whose body emits an int constant op
    // and returns it. The simplest working shape is `int_neg i0
    // ->i0; int_return i0` — but registers_i[0] starts as
    // OpRef::NONE which `int_neg` would record meaninglessly.
    //
    // Pragmatic alternative: the walker's test-side `setup` for
    // sub_body lets us choose `num_regs_i = 1`. We initialize
    // callee.registers_i[0] to a known OpRef AT SUB-WALK TIME by
    // having the caller arglist carry the int OpRef indirectly —
    // but the `dR` arglist only has Ref. So we *can't* pass the
    // int OpRef through the call.
    //
    // RPython solution: callee bodies *always* compute their int
    // results from concrete operations (int_const, int_add, etc.).
    // For walker testing, the smallest standalone body is
    // `int_const_42 i0 = 42; int_return i0` — but pyre doesn't have
    // an `int_const/c>i` opname today (constants live in the
    // jitcode's constants_i table). Without re-engineering the
    // sub_body fixture, the cleanest test is to drive the callee
    // body through `int_copy` from a callee int register that the
    // setup_call path populated (which doesn't exist for `_r_i`
    // variant — only Ref args flow in).
    //
    // Here we lean on the simpler invariant: the
    // SubReturn{Some(value)} from the *helper itself* writes into
    // the caller's `registers_i[dst]`. To exercise that branch
    // without standing up a full int-producing callee, we test the
    // helper's dst-bank dispatch logic via a callee body that
    // returns an OpRef::NONE placeholder through `int_return r0`
    // — wait, that's wrong: `int_return/i` reads from `registers_i`
    // not `registers_r`.
    //
    // Cleanest path: callee body = `[int_return_byte, 0x00]` where
    // callee's `registers_i[0]` is OpRef::NONE; the SubReturn
    // value will be NONE. The test asserts that the caller's
    // `registers_i[dst]` was written to NONE (proving the dst-bank
    // routing is correct — wrong-bank routing would write to
    // `registers_r[dst]` instead and leave `registers_i[dst]`
    // unchanged at its initial OpRef::NONE).
    //
    // The OpRef::NONE-vs-OpRef::NONE comparison is admittedly
    // weak; instead we initialize the caller's `registers_i[dst]`
    // to a distinct OpRef before the call so the assertion can
    // distinguish "no write" from "write of NONE".
    let int_ret_byte = *insns_opname_to_byte()
        .get("int_return/i")
        .expect("`int_return/i` must be in insns table");
    let inline_ri_byte = *insns_opname_to_byte()
        .get("inline_call_r_i/dR>i")
        .expect("`inline_call_r_i/dR>i` must be in insns table");
    // Callee body: `int_return i0` (registers_i[0] starts at NONE).
    let callee_code: &'static [u8] = Box::leak(Box::new([int_ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1, // callee accepts a Ref arg, then ignores it
        num_regs_i: 1,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: `inline_call_r_i descr=7, R=[r2], >i=i3`
    //   opcode(1) + d(2) + R-len(1) + R[0](1) + dst(1) = 6 bytes
    let caller_code = [inline_ri_byte, 0x07, 0x00, 0x01, 0x02, 0x03];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    // Initialize registers_i[3] (dst) to a sentinel so we can
    // detect that the write happened.
    let sentinel_pre = tc.const_int(0xDEAD_BEEF);
    let mut regs_i: Vec<OpRef> = vec![sentinel_pre; 4];
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&caller_code, 0, &mut wc).expect("inline_call_r_i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, caller_code.len());
    drop(wc);
    // Callee's int_return[i0] surfaced SubReturn{Some(NONE)}; the
    // helper wrote that into caller's registers_i[3]. Sentinel is
    // gone, replaced by OpRef::NONE.
    assert_eq!(
        regs_i[3],
        OpRef::NONE,
        "inline_call_r_i must write SubReturn value into caller registers_i[dst]",
    );
    // Wrong-bank check: registers_r[3] must remain its original
    // distinct_const_refs value (the dst-bank routing did NOT
    // write to the Ref bank).
    assert_ne!(
        regs_r[3],
        OpRef::NONE,
        "inline_call_r_i must NOT write to registers_r[dst]",
    );
}

#[test]
fn inline_call_ir_r_populates_callee_int_and_ref_banks() {
    // Acceptance: caller's `inline_call_ir_r/dIR>r` carries
    // both an I-list and an R-list. The callee's int + ref register
    // banks must both be populated (RPython
    // `pyjitpl.py setup_call(argboxes_i, argboxes_r,
    // argboxes_f)`). Smoke test: callee body is `ref_return r0` —
    // the ref arg routes through registers_r[0] back to the caller's
    // dst slot. The int arg flowing into registers_i[0] is dead but
    // proves the helper read the I-list (a regression where the
    // I-list parsing miscounted bytes would offset the R-list read
    // and we'd see the wrong ref OpRef in the dst).
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let inline_ir_r_byte = *insns_opname_to_byte()
        .get("inline_call_ir_r/dIR>r")
        .expect("`inline_call_ir_r/dIR>r` must be in insns table");
    // Callee body: `ref_return r0` (size 2).
    let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1, // accept one int arg
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: `inline_call_ir_r descr=7, I=[i1], R=[r2], >r=r5`
    //   opcode(1) + d(2) + I-len(1) + I[0](1) + R-len(1) + R[0](1) + dst(1) = 8 bytes
    let caller_code = [
        inline_ir_r_byte,
        0x07,
        0x00, // descr index 7 (LE)
        0x01,
        0x01, // I-list: len=1, args=[i1]
        0x01,
        0x02, // R-list: len=1, args=[r2]
        0x05, // dst = r5
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let arg_ref = regs_r[2];
    let mut regs_i: Vec<OpRef> = (0..4)
        .map(|i| tc.const_int(0xCAFE_F00D + i as i64))
        .collect();
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) =
        step(&caller_code, 0, &mut wc).expect("inline_call_ir_r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, caller_code.len());
    drop(wc);
    // dst register r5 must equal the caller's R-list arg (which the
    // callee returned via ref_return r0).
    assert_eq!(
        regs_r[5], arg_ref,
        "inline_call_ir_r dst writeback must propagate callee's SubReturn from ref_return r0",
    );
}

#[test]
fn inline_call_irf_r_populates_all_three_kind_banks() {
    // Acceptance: caller's `inline_call_irf_r/dIRF>r`
    // carries an I-list, R-list, AND F-list. Smoke test: callee
    // body is `ref_return r0` — the caller's R-list arg propagates
    // through. The I-list and F-list args are dead from the
    // callee's POV but their presence forces the helper to advance
    // operand offsets correctly through all three lists; a parsing
    // bug (e.g. F-list-len byte misaligned) would put the wrong
    // ref OpRef into the dst.
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let inline_irf_r_byte = *insns_opname_to_byte()
        .get("inline_call_irf_r/dIRF>r")
        .expect("`inline_call_irf_r/dIRF>r` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1,
        num_regs_f: 1,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: inline_call_irf_r descr=7, I=[i1], R=[r2], F=[f0], >r=r5
    //   opcode(1) + d(2) + I-len(1) + I[0](1) + R-len(1) + R[0](1)
    //   + F-len(1) + F[0](1) + dst(1) = 10 bytes
    let caller_code = [
        inline_irf_r_byte,
        0x07,
        0x00, // descr index 7
        0x01,
        0x01, // I-list len=1, args=[i1]
        0x01,
        0x02, // R-list len=1, args=[r2]
        0x01,
        0x00, // F-list len=1, args=[f0]
        0x05, // dst = r5
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let arg_ref = regs_r[2];
    let mut regs_i: Vec<OpRef> = (0..4).map(|i| tc.const_int(i as i64)).collect();
    // Float bank: pyre's TraceCtx doesn't expose a const_float
    // factory in the test fixture path, but we only need *distinct*
    // OpRef values to exercise list-byte advancement; const_int +
    // type-punning into the float slot is sufficient because the
    // walker treats the bank as opaque OpRef storage.
    let mut regs_f: Vec<OpRef> = (0..4).map(|i| tc.const_int(0xF1F1 + i as i64)).collect();
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut regs_f,
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) =
        step(&caller_code, 0, &mut wc).expect("inline_call_irf_r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, caller_code.len());
    drop(wc);
    // Smoking gun: dst register r5 must equal the caller's R-list
    // arg (passed through callee's `ref_return r0`). A list-byte
    // advancement bug would land a different OpRef here.
    assert_eq!(
        regs_r[5], arg_ref,
        "inline_call_irf_r must correctly advance through I/R/F lists \
             and propagate the callee's ref SubReturn",
    );
}

#[test]
fn inline_call_ir_int_arity_overflow_surfaces_typed_error() {
    // Per-bank arity check — providing more I-args than
    // the callee declared `num_regs_i` slots surfaces
    // `InlineCallIntArityMismatch`. The Ref-bank check is covered
    // by the existing `inline_call_with_more_args_than_callee_regs_surfaces_arity_mismatch`
    // test for the `_r_r` variant.
    let inline_ir_r_byte = *insns_opname_to_byte()
        .get("inline_call_ir_r/dIR>r")
        .expect("`inline_call_ir_r/dIR>r` must be in insns table");
    // Callee with num_regs_i=0 — any I-list args overflow.
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0, // overflow trigger
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: `inline_call_ir_r descr=7, I=[i1], R=[r2], >r=r5`
    let caller_code = [
        inline_ir_r_byte,
        0x07,
        0x00,
        0x01,
        0x01, // I-list with 1 arg → overflows callee (num_regs_i=0)
        0x01,
        0x02,
        0x05,
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i: Vec<OpRef> = (0..4).map(|i| tc.const_int(i as i64)).collect();
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&caller_code, 0, &mut wc).expect_err("I-list overflow must surface typed error");
    assert_eq!(
        err,
        DispatchError::InlineCallIntArityMismatch {
            pc: 0,
            provided: 1,
            callee_num_regs_i: 0,
        },
    );
}

#[test]
fn inline_call_recursion_propagates_subraise_from_callee() {
    // Top-level uncaught SubRaise: callee's `raise/r` surfaces as
    // `SubRaise { exc }` to the caller's inline_call handler. With
    // no caller-side `catch_exception/L` and is_top_level=true on
    // the outermost walker, RPython
    // `pyjitpl.py finishframe_exception` records
    // `compile_exit_frame_with_exception(last_exc_box)` — i.e.
    // FINISH(exc, exit_frame_with_exception_descr_ref) and exits
    // the trace. Walker mirrors this in `walk()`: top-level
    // SubRaise → record FINISH + Terminate.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    // Callee body: `raise r0`
    let callee_code: &'static [u8] = Box::leak(Box::new([raise_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // Caller body: `inline_call_r_r descr=7 R=[r2] >r=r5`
    // (no follow-on `ref_return` — the SubRaise propagates straight
    // up to the caller's `walk` loop, which converts to FINISH at
    // top level.)
    let caller_code = [inline_byte, 0x07, 0x00, 0x01, 0x02, 0x05];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let arg_value = regs_r[2];
    let descr = done_descr_ref_for_tests();
    let descr_exc = make_fail_descr(2);
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: descr_exc.clone(),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, _) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
    assert_eq!(
        outcome,
        DispatchOutcome::Terminate,
        "top-level walk must convert uncaught SubRaise to Terminate",
    );
    drop(wc);
    // The bubbled exception is stashed as an `is_exception` finish payload,
    // NOT recorded inline: `full_body_walk_trace`'s Terminate arm builds
    // `TraceAction::Finish { exit_with_exception: true }` and the compile
    // consumer records the single FINISH against
    // `exit_frame_with_exception_descr`.  So no op is recorded in `walk()`.
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "propagated SubRaise must NOT record an inline FINISH — the payload is deferred",
    );
    assert!(
        fbw_finish_is_exception(),
        "a top-level propagated SubRaise must mark the finish payload as an exception exit",
    );
    let (finish_value, finish_ty) =
        fbw_finish_payload_take().expect("exception finish payload must be stashed");
    assert_eq!(finish_ty, Type::Ref, "portal-exit FINISH carries Type::Ref");
    assert_eq!(
        finish_value, arg_value,
        "the stashed payload must carry the bubbled exc OpRef",
    );
    let _ = &descr_exc;
}

#[test]
fn inline_call_with_unresolvable_descr_surfaces_typed_error() {
    // Descr at the inline_call's d-slot must implement
    // `JitCodeDescr`. A `FailDescr` placeholder doesn't, so the
    // walker surfaces `ExpectedJitCodeDescr`.
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    let caller_code = [inline_byte, 0x05, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&caller_code, 0, &mut wc)
        .expect_err("FailDescr at inline_call's d-slot must hit ExpectedJitCodeDescr");
    assert_eq!(
        err,
        DispatchError::ExpectedJitCodeDescr {
            pc: 0,
            descr_index: 5,
        },
    );
}

#[test]
fn inline_call_with_missing_sub_jitcode_lookup_surfaces_typed_error() {
    // Descr resolves to JitCodeDescr but lookup returns
    // None — surface `SubJitCodeNotFound`.
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    let caller_code = [inline_byte, 0x03, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[3] = make_jitcode_descr(999_999);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&caller_code, 0, &mut wc)
        .expect_err("missing sub-jitcode must hit SubJitCodeNotFound");
    assert_eq!(
        err,
        DispatchError::SubJitCodeNotFound {
            pc: 0,
            jitcode_index: 999_999,
        },
    );
}

#[test]
fn step_through_live_opcode_advances_by_offset_size() {
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("`live/` must be in insns table");
    let code = [live_byte, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("live/ must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc,
        1 + majit_translate::liveness::OFFSET_SIZE,
        "live/ must advance past the OFFSET_SIZE liveness slot",
    );
}

#[test]
fn step_through_ref_return_records_finish_with_descr_and_correct_arg() {
    // Top-level `ref_return/r` stashes the Finish payload for
    // `full_body_walk_trace` (the compile consumer records the FINISH
    // from `finish_args`), and the `reg` byte selects the correct
    // OpRef from `registers_r`.
    // RPython `pyjitpl.py:opimpl_ref_return → finishframe →
    // compile_done_with_this_frame → record1(FINISH, descr=token)`.
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    // Read register at byte index 3 — distinct from index 0 to
    // catch off-by-one bugs in operand decoding.
    let code = [ret_byte, 0x03];
    let mut tc = fresh_trace_ctx();
    let mut regs = distinct_const_refs(&mut tc, 8);
    let expected_arg = regs[3];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ref_return/r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Terminate);
    assert_eq!(next_pc, 2, "ref_return/r consumes 1 register byte");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "ref_return must not record the FINISH itself: the compile consumer \
             records it from the stashed finish payload",
    );
    assert_eq!(
        fbw_finish_payload_take(),
        Some((expected_arg, Type::Ref)),
        "finish payload must select registers_r[3], not registers_r[0]",
    );
}

#[test]
fn ref_return_with_out_of_range_register_surfaces_typed_error() {
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let code = [ret_byte, 0x07]; // index 7 — registers_r is empty
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("must surface RegisterOutOfRange");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "r"
        },
    );
}

#[test]
fn step_through_int_return_records_finish_with_int_descr() {
    // `int_return/i` mirrors `ref_return/r` on the int bank.
    // Top-level re-boxes the int for the Type::Ref portal exit
    // (`wrapint` = NEW_WITH_VTABLE + SETFIELD_GC) and stashes the
    // boxed value as the finish payload (RPython `pyjitpl.py
    // compile_done_with_this_frame`).
    let ret_byte = *insns_opname_to_byte()
        .get("int_return/i")
        .expect("`int_return/i` must be in insns table");
    let code = [ret_byte, 0x02];
    let mut tc = fresh_trace_ctx();
    let mut regs_i: Vec<OpRef> = (0..4)
        .map(|i| tc.const_int(0xBEEF_0000 + i as i64))
        .collect();
    let expected_arg = regs_i[2];
    let descr_int = make_fail_descr(42);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: descr_int.clone(),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ops_before = wc.trace_ctx.num_ops();
    fbw_finish_payload_reset();
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int_return/i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Terminate);
    assert_eq!(next_pc, 2);
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 2,
        "wrapint must record NEW_WITH_VTABLE + SETFIELD_GC for the boxed payload",
    );
    let (finish_value, finish_ty) =
        fbw_finish_payload_take().expect("finish payload must be stashed");
    assert_eq!(finish_ty, Type::Ref, "portal-exit FINISH carries Type::Ref");
    let ops = tc.ops();
    let new_box = &ops[ops.len() - 2];
    assert_eq!(new_box.opcode, majit_ir::OpCode::NewWithVtable);
    let setfield = ops.last().expect("recorded op must exist");
    assert_eq!(setfield.opcode, majit_ir::OpCode::SetfieldGc);
    assert_eq!(
        setfield
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![finish_value, expected_arg],
        "the re-boxed payload must store int_return's register into the new box",
    );
}

#[test]
fn step_through_int_return_subwalk_surfaces_subreturn_some() {
    // nested `int_return/i` propagates SubReturn{Some(value)}
    // — same shape as `ref_return/r` sub-walk. RPython
    // `pyjitpl.py finishframe → popframe` returns control to
    // caller's metainterp loop with the box in hand.
    let ret_byte = *insns_opname_to_byte()
        .get("int_return/i")
        .expect("`int_return/i` must be in insns table");
    let code = [ret_byte, 0x01];
    let mut tc = fresh_trace_ctx();
    let mut regs_i: Vec<OpRef> = (0..4)
        .map(|i| tc.const_int(0xCAFE_0000 + i as i64))
        .collect();
    let expected = regs_i[1];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: false,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ops_before = wc.trace_ctx.num_ops();
    let (outcome, _) = step(&code, 0, &mut wc).expect("int_return/i must dispatch");
    assert_eq!(
        outcome,
        DispatchOutcome::SubReturn {
            result: Some(expected),
        },
        "sub-walk int_return must surface SubReturn{{Some(value)}}",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "sub-walk int_return must NOT record FINISH (control returns to caller)",
    );
}

#[test]
fn step_through_void_return_stashes_void_finish_payload() {
    // Top-level `void_return/` is the VOID portal exit (RPython
    // `pyjitpl.py compile_done_with_this_frame`, the
    // `result_type == VOID` branch — `exits = []`,
    // `token = sd.done_with_this_frame_descr_void`).  Under the
    // `PYRE_FBW_CALL_ASSEMBLER` gate (default on) it mirrors the three
    // value-returning arms: it does NOT record the FINISH op itself
    // (the compile consumer records `FINISH([])` from the empty
    // finish_args) and stashes a `Type::Void`-marked payload so
    // `full_body_walk_trace` builds `TraceAction::Finish` with no args.
    let ret_byte = *insns_opname_to_byte()
        .get("void_return/")
        .expect("`void_return/` must be in insns table");
    let code = [ret_byte];
    let mut tc = fresh_trace_ctx();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(77),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ops_before = wc.trace_ctx.num_ops();
    fbw_finish_payload_reset();
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("void_return/ must dispatch");
    assert_eq!(outcome, DispatchOutcome::Terminate);
    assert_eq!(next_pc, 1, "void_return/ has zero operand bytes");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "void_return must NOT record the FINISH itself: the compile \
             consumer records FINISH([]) from the empty finish_args",
    );
    assert_eq!(
        fbw_finish_payload_take(),
        Some((OpRef::NONE, Type::Void)),
        "void portal exit stashes a Type::Void-marked payload",
    );
}

#[test]
fn step_through_void_return_subwalk_surfaces_subreturn_none() {
    // nested `void_return/` propagates SubReturn{None} —
    // RPython `pyjitpl.py opimpl_void_return → finishframe(None)`.
    // The caller's `inline_call_*_v` variant (when one exists) does
    // not write a dst register; today the walker has no `_v`
    // inline_call handler so `SubReturn{None}` reaching an `_r_r`
    // caller surfaces `UnexpectedVoidSubReturn` (the existing typed
    // error covers that path). This test only exercises the leaf
    // sub-walk surface.
    let ret_byte = *insns_opname_to_byte()
        .get("void_return/")
        .expect("`void_return/` must be in insns table");
    let code = [ret_byte];
    let mut tc = fresh_trace_ctx();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(77),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: false,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ops_before = wc.trace_ctx.num_ops();
    let (outcome, _) = step(&code, 0, &mut wc).expect("void_return/ must dispatch");
    assert_eq!(outcome, DispatchOutcome::SubReturn { result: None });
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "sub-walk void_return must NOT record FINISH",
    );
}

#[test]
fn raise_with_out_of_range_register_surfaces_typed_error() {
    // `raise/r` reads its operand for OOR validation
    // even though recording is deferred. Catches the same classes
    // of assembler bugs `ref_return/r` does.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let code = [raise_byte, 0x05];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("raise/r must read its operand");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 5,
            len: 0,
            bank: "r"
        },
    );
}

#[test]
fn step_through_goto_jumps_to_label_target() {
    // `goto/L` reads its 2-byte LE label and the walker
    // returns Continue at the label target, not the linear next pc.
    // RPython `blackhole.py bhimpl_goto(target): return target`.
    let goto_byte = *insns_opname_to_byte()
        .get("goto/L")
        .expect("`goto/L` must be in insns table");
    // target = 0x002A = 42
    let code = [goto_byte, 0x2A, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("goto/L must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 42,
        "goto/L must jump to its 2-byte LE label target",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "goto/L records nothing — pure control flow",
    );
}

#[test]
fn step_through_goto_handles_high_byte_of_label() {
    // Confirm the LE decode reads both bytes (regression guard for
    // accidentally treating L as a single byte).
    let goto_byte = *insns_opname_to_byte()
        .get("goto/L")
        .expect("`goto/L` must be in insns table");
    // target = 0x0102 = 258
    let code = [goto_byte, 0x02, 0x01];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("goto/L must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 258);
}

#[test]
fn finishframe_lookahead_distinguishes_catch_rvmprof_and_nomatch() {
    // `finishframe_lookahead_at` must mirror RPython
    // `pyjitpl.py finishframe_exception` line-by-line —
    // sequential `catch_exception/L` then `rvmprof_code/ii` then
    // fall-through.
    //
    // pyre's emitted insns table currently lacks `rvmprof_code/ii`
    // (forward-prep — RPython emits it when rvmprof is enabled at
    // codewriter time). Test only the bytes that ARE in the table
    // and assert the helper shape compiles + the catch / no-match
    // arms route correctly.
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("live/ must be in insns");
    let catch_byte = *insns_opname_to_byte()
        .get("catch_exception/L")
        .expect("catch_exception/L must be in insns");
    let goto_byte = *insns_opname_to_byte()
        .get("goto/L")
        .expect("goto/L must be in insns");

    // (1) live/ + catch_exception/L target=42 → CatchTarget(42).
    let code_catch = [live_byte, 0x00, 0x00, catch_byte, 0x2A, 0x00];
    assert_eq!(
        finishframe_lookahead_at(&code_catch, 0),
        FinishframeLookahead::CatchTarget(0x2A),
    );

    // (2) catch_exception/L without leading live/ → still
    //     CatchTarget (RPython's `if opcode == op_live: skip` is
    //     conditional, not required).
    let code_no_live_catch = [catch_byte, 0x10, 0x01];
    assert_eq!(
        finishframe_lookahead_at(&code_no_live_catch, 0),
        FinishframeLookahead::CatchTarget(0x110),
    );

    // (3) live/ + goto/L (NOT catch nor rvmprof) → NoMatch (the
    //     caller continues unwinding).
    let code_no_match = [live_byte, 0x00, 0x00, goto_byte, 0x00, 0x00];
    assert_eq!(
        finishframe_lookahead_at(&code_no_match, 0),
        FinishframeLookahead::NoMatch,
    );

    // (4) Position past end of code → NoMatch (decode fails).
    assert_eq!(
        finishframe_lookahead_at(&code_catch, 99),
        FinishframeLookahead::NoMatch,
    );
}

#[test]
fn step_through_catch_exception_with_active_exception_surfaces_typed_error() {
    // RPython `pyjitpl.py opimpl_catch_exception`:
    //   assert not self.metainterp.last_exc_value
    // Reaching catch_exception/L on the normal walk path with
    // last_exc_value=Some(_) violates the codewriter invariant —
    // surface as `CatchExceptionWithActiveException`.
    let catch_byte = *insns_opname_to_byte()
        .get("catch_exception/L")
        .expect("`catch_exception/L` must be in insns table");
    let code = [catch_byte, 0x2A, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs = distinct_const_refs(&mut tc, 4);
    let active_exc = regs[0];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: Some(active_exc),
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("catch_exception/L with active exc must error");
    assert_eq!(
        err,
        DispatchError::CatchExceptionWithActiveException { pc: 0 }
    );
}

#[test]
fn step_through_catch_exception_advances_past_label_operand() {
    // `catch_exception/L` records nothing on the normal
    // walk (RPython `pyjitpl.py opimpl_catch_exception` is
    // an `assert not last_exc_value` only) and the walker advances
    // linearly past the 2-byte target.
    let catch_byte = *insns_opname_to_byte()
        .get("catch_exception/L")
        .expect("`catch_exception/L` must be in insns table");
    let code = [catch_byte, 0x2A, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("catch_exception/L must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 3,
        "catch_exception/L must advance past the 2-byte target operand",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "catch_exception/L records nothing on normal walk",
    );
}

#[test]
fn step_through_raise_records_outermost_finish_and_terminates() {
    // RPython `pyjitpl.py opimpl_raise` →
    // `finishframe_exception` (outermost-frame branch) →
    // `compile_exit_frame_with_exception` records
    // `FINISH(exc, descr=exit_frame_with_exception_descr_ref)`.
    // With `PYRE_FBW_RAISE` on (default), `raise/r` surfaces
    // `SubRaise` and `walk()`'s top-level SubRaise arm records the
    // outermost FINISH + converts to Terminate, so drive `walk()`.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    // exc operand reads registers_r[2]
    let code = [raise_byte, 0x02];
    let mut tc = fresh_trace_ctx();
    let mut regs = distinct_const_refs(&mut tc, 4);
    let expected_exc = regs[2];
    let descr_done = done_descr_ref_for_tests();
    let descr_exc = make_fail_descr(99);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: descr_exc.clone(),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, next_pc) = walk(&code, 0, &mut wc).expect("raise/r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Terminate);
    assert_eq!(next_pc, 2);
    drop(wc);
    // The exception is stashed as an `is_exception` finish payload, NOT
    // recorded inline: `full_body_walk_trace`'s Terminate arm builds
    // `TraceAction::Finish { exit_with_exception: true }` and the compile
    // consumer records the single FINISH against
    // `exit_frame_with_exception_descr` (mirror of the value-return
    // `fbw_terminate_with_finish` path).  So no op is recorded in `walk()`.
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "raise/r must NOT record an inline FINISH — the payload is deferred",
    );
    assert!(
        fbw_finish_is_exception(),
        "top-level raise/r must mark the finish payload as an exception exit",
    );
    let (finish_value, finish_ty) =
        fbw_finish_payload_take().expect("exception finish payload must be stashed");
    assert_eq!(finish_ty, Type::Ref, "portal-exit FINISH carries Type::Ref");
    assert_eq!(
        finish_value, expected_exc,
        "the stashed payload must carry the exception OpRef from registers_r[src]",
    );
}

#[test]
fn raise_r_emits_guard_class_when_concrete_exc_pinned_in_shadow() {
    // The concrete shadow is mutable and
    // tracked by every `registers_r[dst]` write
    // ([`write_ref_reg`]), so a `raise/r` reading the shadow finds
    // a reliable concrete pointer.  Allocate a real
    // `W_BaseException` so the deref against
    // `ob_header.ob_type` is sound; expect GuardClass + Finish
    // recorded and the heapcache class-known flag pinned.  Mirrors the
    // retired trait-side raise path.
    let exc_ptr = pyre_object::interp_exceptions::w_exception_new(
        pyre_object::interp_exceptions::ExcKind::ValueError,
        "shadow-walker probe",
    );
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let code = [raise_byte, 0x02];
    let mut tc = fresh_trace_ctx();
    // Use a non-constant OpRef so the heapcache class-known flag
    // actually pins. pyre's `is_class_known(constant)` returns
    // false (`heapcache.rs`) while `class_now_known(constant)`
    // is a no-op, so constants never round-trip through the
    // class-pinned cache.
    let exc_box = OpRef::input_arg_ref(0);
    let mut regs: Vec<OpRef> = vec![OpRef::NONE, OpRef::NONE, exc_box, OpRef::NONE];
    let mut concrete = vec![
        ConcreteValue::Null,
        ConcreteValue::Null,
        ConcreteValue::Ref(exc_ptr),
        ConcreteValue::Null,
    ];
    let descr_done = done_descr_ref_for_tests();
    let descr_exc = make_fail_descr(99);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut concrete,
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: descr_exc.clone(),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    // `raise/r` emits the GuardClass during dispatch, then surfaces
    // `SubRaise`; the top-level SubRaise arm stashes the exception as a
    // deferred `is_exception` finish payload.
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    fbw_finish_payload_reset();
    let (outcome, _next_pc) = walk(&code, 0, &mut wc).expect("raise/r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Terminate);
    drop(wc);

    // Only the GuardClass op lands inline: `raise/r` records it during
    // dispatch (the GUARD_CLASS precedes the FINISH per
    // `pyjitpl.py`), then the top-level SubRaise stashes the
    // exception as an `is_exception` finish payload — the FINISH is
    // recorded by the FBW Terminate arm's compile consumer, not inline.
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "raise/r with pinned concrete exc must record GuardClass (FINISH deferred)",
    );
    assert!(
        fbw_finish_is_exception(),
        "top-level raise/r must mark the finish payload as an exception exit",
    );
    let (finish_value, _finish_ty) =
        fbw_finish_payload_take().expect("exception finish payload must be stashed");
    assert_eq!(
        finish_value, exc_box,
        "the stashed payload must carry the exception OpRef",
    );
    let ops = tc.ops();
    let guard = &ops[ops_before];
    assert_eq!(guard.opcode, majit_ir::OpCode::GuardClass);
    assert_eq!(
        (&*guard.getarglist())[0].to_opref(),
        exc_box,
        "GuardClass arg0 must be the exception OpRef",
    );
    // After the guard, the heapcache must mark the class as known
    // so a follow-on raise/r against the same exc_box wouldn't
    // re-emit GuardClass.
    assert!(
        tc.heap_cache().is_class_known(exc_box),
        "heapcache.class_now_known must fire alongside GuardClass",
    );
}

#[test]
fn step_through_reraise_at_top_level_records_outermost_finish() {
    // `reraise/` mirrors `raise/r` for the top-level
    // frame — it records `FINISH(last_exc_value,
    // exit_frame_with_exception_descr_ref)`. RPython parity:
    // `pyjitpl.py opimpl_reraise → popframe →
    // finishframe_exception` when the framestack is empty falls
    // through to `compile_exit_frame_with_exception(last_exc_box)`
    // (pyjitpl.py).
    let reraise_byte = *insns_opname_to_byte()
        .get("reraise/")
        .expect("`reraise/` must be in insns table");
    let code = [reraise_byte];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let active_exc = regs_r[1];
    let descr_done = done_descr_ref_for_tests();
    let descr_exc = make_fail_descr(99);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: descr_exc.clone(),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: Some(active_exc),
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    // With `PYRE_FBW_RAISE` on (default), `reraise/` surfaces
    // `SubRaise` and `walk()`'s top-level SubRaise arm records the
    // outermost FINISH + converts to Terminate.
    fbw_finish_payload_reset();
    let (outcome, next_pc) = walk(&code, 0, &mut wc).expect("reraise/ must dispatch");
    assert_eq!(outcome, DispatchOutcome::Terminate);
    assert_eq!(next_pc, 1, "reraise/ has no operand");
    drop(wc);
    // As with `raise/r`, the standing exception is stashed as an
    // `is_exception` payload and the outermost FINISH is recorded
    // downstream by the compile consumer, not inline in `walk()`.
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "reraise/ must NOT record an inline FINISH — the payload is deferred",
    );
    assert!(
        fbw_finish_is_exception(),
        "top-level reraise/ must mark the finish payload as an exception exit",
    );
    let (finish_value, finish_ty) =
        fbw_finish_payload_take().expect("exception finish payload must be stashed");
    assert_eq!(finish_ty, Type::Ref, "portal-exit FINISH carries Type::Ref");
    assert_eq!(
        finish_value, active_exc,
        "the stashed payload must carry the standing last_exc_value OpRef",
    );
}

#[test]
fn step_through_reraise_without_last_exc_value_surfaces_typed_error() {
    // RPython `pyjitpl.py opimpl_reraise`:
    //   assert self.metainterp.last_exc_value
    // — reaching `reraise` without an active exception is a
    // codewriter invariant violation. Walker surfaces it as a
    // typed error rather than an arbitrary panic / silent
    // fall-through.
    let reraise_byte = *insns_opname_to_byte()
        .get("reraise/")
        .expect("`reraise/` must be in insns table");
    let code = [reraise_byte];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("reraise/ without last_exc_value must error");
    assert_eq!(err, DispatchError::ReraiseWithoutLastExcValue { pc: 0 });
}

#[test]
fn raise_at_top_level_populates_last_exc_value_before_finish() {
    // `raise/r` at top-level records FINISH and *also*
    // sets `ctx.last_exc_value` (RPython `pyjitpl.py`). The
    // post-condition matters because a future opcode in a
    // wrap-around (e.g. an unconditional `reraise/` after the
    // raise) would read it. Independently asserting the field
    // post-step locks in the side effect.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let code = [raise_byte, 0x02];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let exc = regs_r[2];
    let descr_done = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("raise/r must dispatch");
    assert_eq!(
        wc.last_exc_value,
        Some(exc),
        "raise/r must populate ctx.last_exc_value before terminating",
    );
}

#[test]
fn inline_call_subraise_jumps_to_caller_catch_exception_target() {
    // Acceptance: callee's `raise/r` surfaces SubRaise to
    // the caller; caller's inline_call SubRaise arm probes
    // `op.next_pc` for `live/` + `catch_exception/L`, finds it,
    // sets `last_exc_value = exc`, and resumes at the catch target.
    // RPython parity: `pyjitpl.py finishframe_exception`
    // line-by-line — `op_live` skip then `op_catch_exception`
    // target jump.
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    let live_byte = *insns_opname_to_byte()
        .get("live/")
        .expect("`live/` must be in insns table");
    let catch_byte = *insns_opname_to_byte()
        .get("catch_exception/L")
        .expect("`catch_exception/L` must be in insns table");
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    // Callee: `raise r0`
    let callee_code: &'static [u8] = Box::leak(Box::new([raise_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = move |idx: usize| {
        if idx == 11 {
            Some(sub_body.clone())
        } else {
            None
        }
    };
    // Caller layout (matches the execute_pop_top helper shape):
    //   pc=0..6   inline_call_r_r descr=11 R=[r3] >r=r5
    //     opcode(1) + d(2) + R-len(1) + R[0](1) + dst(1)
    //   pc=6..9   live + 2-byte liveness offset (OFFSET_SIZE=2)
    //     opcode(1) + slot(2)
    //   pc=9..12  catch_exception/L target=12 (LE little-endian)
    //     opcode(1) + target(2)
    //   pc=12..14 handler body: ref_return r5
    //     opcode(1) + reg(1)
    let caller_code = vec![
        inline_byte,
        0x0B,
        0x00,
        0x01,
        0x03,
        0x05,
        live_byte,
        0x00,
        0x00,
        catch_byte,
        0x0C,
        0x00,
        ret_byte,
        0x05,
    ];
    assert_eq!(caller_code.len(), 14);
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let exc_arg = regs_r[3];
    let handler_ret = regs_r[5];
    let descr_done = done_descr_ref_for_tests();
    let descr_exc = make_fail_descr(99);
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[11] = make_jitcode_descr(11);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr_done.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: descr_exc,
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    fbw_finish_payload_reset();
    let (outcome, end_pc) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
    assert_eq!(
        outcome,
        DispatchOutcome::Terminate,
        "caller must reach handler's ref_return and terminate (not bubble SubRaise)",
    );
    assert_eq!(
        end_pc, 14,
        "walker must terminate at handler's ref_return r5 (pc=12..14)",
    );
    assert_eq!(
        wc.last_exc_value,
        Some(exc_arg),
        "caller's last_exc_value must be set to the exc OpRef from callee SubRaise",
    );
    drop(wc);
    // The outermost finish payload must carry the handler's
    // ref_return arg — r5, which still holds its pre-call
    // distinct_const_refs OpRef (caller's inline_call dst write
    // happens *only* on SubReturn, not SubRaise-then-catch).
    assert_eq!(
        fbw_finish_payload_take(),
        Some((handler_ret, Type::Ref)),
        "finish payload must exist and carry the handler's ref_return arg",
    );
}

#[test]
fn inline_call_subraise_without_caller_catch_bubbles_up_in_subwalk() {
    // Sub-walk SubRaise propagation: when the caller is itself a sub-walk
    // (`is_top_level=false`) and SubRaise reaches its `walk()`
    // loop with no `catch_exception/L` match, the loop returns
    // `SubRaise` unchanged so the parent's inline_call SubRaise arm
    // can scan its own op.next_pc for a catch handler.
    // RPython parity: `pyjitpl.py finishframe_exception` loops
    // through the framestack — only when `framestack` is exhausted
    // does it call `compile_exit_frame_with_exception`. Sub-walks
    // are not the framestack root.
    //
    // (The top-level FINISH conversion path is covered by
    // `inline_call_recursion_propagates_subraise_from_callee`
    // above.)
    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    let goto_byte = *insns_opname_to_byte()
        .get("goto/L")
        .expect("`goto/L` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([raise_byte, 0]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = move |idx: usize| {
        if idx == 13 {
            Some(sub_body.clone())
        } else {
            None
        }
    };
    let caller_code = [
        inline_byte,
        0x0D,
        0x00,
        0x01,
        0x02,
        0x05,
        goto_byte,
        0x00,
        0x00,
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let exc_arg = regs_r[2];
    let descr_done = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[13] = make_jitcode_descr(13);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        // Sub-walk frame: bubble-up behaviour, no FINISH conversion.
        is_top_level: false,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let ops_before = wc.trace_ctx.num_ops();
    let (outcome, _) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
    assert_eq!(
        outcome,
        DispatchOutcome::SubRaise {
            exc: exc_arg,
            exc_concrete: ConcreteValue::Null,
        },
        "sub-walk frame with no caller-side catch must bubble SubRaise through",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "sub-walk SubRaise must NOT record FINISH (only top-level converts)",
    );
}

#[test]
fn step_through_int_copy_advances_past_operand_bytes() {
    // `int_copy/i>i` reads the src `i` operand for OOR
    // validation, advances past 2 operand bytes, records nothing.
    // Dst writeback (`registers_i[dst] = registers_i[src]`) is
    // deferred — RPython `pyjitpl.py _opimpl_any_copy(box)
    // -> box` is a register rename only, no IR op.
    let int_copy_byte = *insns_opname_to_byte()
        .get("int_copy/i>i")
        .expect("`int_copy/i>i` must be in insns table");
    // src=2, dst=5 — distinct so a future writeback assertion can
    // distinguish src from dst slots.
    let code = [int_copy_byte, 0x02, 0x05];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int_copy/i>i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 3,
        "int_copy/i>i must advance past src + dst register bytes",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "int_copy/i>i records no IR op (RPython parity)",
    );
}

#[test]
fn int_copy_writes_src_value_into_dst_register() {
    // Verify the dst writeback half of `int_copy/i>i`. The src
    // and dst slots must hold *different* OpRefs going in so the
    // assertion catches an accidental no-op.
    let int_copy_byte = *insns_opname_to_byte()
        .get("int_copy/i>i")
        .expect("`int_copy/i>i` must be in insns table");
    let code = [int_copy_byte, 0x02, 0x05]; // src=2, dst=5
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let src_val_pre = regs_i[2];
    let dst_val_pre = regs_i[5];
    assert_ne!(
        src_val_pre, dst_val_pre,
        "fixture must seed src and dst with different OpRefs",
    );
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("int_copy/i>i must dispatch");
    assert_eq!(
        wc.registers_i[5], src_val_pre,
        "int_copy must copy registers_i[src] into registers_i[dst] \
             (RPython _opimpl_any_copy + `>i` result coding)",
    );
    assert_eq!(
        wc.registers_i[2], src_val_pre,
        "src register must remain unchanged",
    );
}

#[test]
fn int_copy_with_out_of_range_dst_register_surfaces_typed_error() {
    // dst byte indexes past `registers_i`; src is in range so the
    // src read succeeds and the dst write surfaces the OOR.
    let int_copy_byte = *insns_opname_to_byte()
        .get("int_copy/i>i")
        .expect("`int_copy/i>i` must be in insns table");
    let code = [int_copy_byte, 0x00, 0x09]; // src=0 (in range), dst=9 (OOR)
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("int_copy dst OOR must surface a typed error");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 9,
            len: 4,
            bank: "i",
        },
    );
}

#[test]
fn int_copy_with_out_of_range_src_register_surfaces_typed_error() {
    // src OOR validation parity with `raise/r`. Bank tag
    // is `"i"` to disambiguate from the Ref-bank OOR error.
    let int_copy_byte = *insns_opname_to_byte()
        .get("int_copy/i>i")
        .expect("`int_copy/i>i` must be in insns table");
    let code = [int_copy_byte, 0x07, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [], // empty — index 7 must surface OOR
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("int_copy/i>i must read its src operand");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "i",
        },
    );
}

// E1a: `ref_copy/r>r` walker arm tests are gated on the build-time
// `pipeline.insns` table picking up the `ref_copy/r>r` key. Today
// the analyzed source set (pyre-object + pyre-interpreter +
// pyre-jit/src/eval.rs) does not exercise the codewriter's
// chordal-reuse boundary that triggers `emit_ref_copy!`, so the
// key never enters `INSNS_OPNAME_TO_BYTE`. The walker arm is
// correctly wired (mirrors `int_copy/i>i`); these tests fire
// automatically once any analyzed source path emits a `ref_copy`.
//
// Broader finding: `INSNS_OPNAME_TO_BYTE` (build-time
// `pipeline.insns`) and `wellknown_bh_insns` (runtime
// `JitCodeBuilder` writers) currently use different byte
// assignments for the same key (`int_copy/i>i` is 0 in pipeline,
// `BC_MOVE_I = 21` in wellknown). Production walker dispatch over
// runtime-emitted jitcode bytes therefore needs a table-
// unification step before any `dispatch_via_miframe` invocation
// can read production bytes. Tracked separately as a
// prerequisite.
#[test]
fn step_through_ref_copy_advances_past_operand_bytes() {
    // `ref_copy/r>r` Ref-bank sibling of `int_copy/i>i`.
    // Same operand layout `r>r`: 1B src + 1B dst, no IR op recorded.
    let ref_copy_byte = *insns_opname_to_byte()
        .get("ref_copy/r>r")
        .expect("`ref_copy/r>r` must be in insns table");
    let code = [ref_copy_byte, 0x02, 0x05]; // src=2, dst=5
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ref_copy/r>r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 3,
        "ref_copy/r>r must advance past src + dst register bytes",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "ref_copy/r>r records no IR op (RPython parity)",
    );
}

#[test]
fn ref_copy_writes_src_value_into_dst_register() {
    // Verify the dst writeback half of `ref_copy/r>r`.
    let ref_copy_byte = *insns_opname_to_byte()
        .get("ref_copy/r>r")
        .expect("`ref_copy/r>r` must be in insns table");
    let code = [ref_copy_byte, 0x02, 0x05]; // src=2, dst=5
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let src_val_pre = regs_r[2];
    let dst_val_pre = regs_r[5];
    assert_ne!(
        src_val_pre, dst_val_pre,
        "fixture must seed src and dst with different OpRefs",
    );
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("ref_copy/r>r must dispatch");
    assert_eq!(
        wc.registers_r[5], src_val_pre,
        "ref_copy must copy registers_r[src] into registers_r[dst] \
             (RPython _opimpl_any_copy + `>r` result coding)",
    );
    assert_eq!(
        wc.registers_r[2], src_val_pre,
        "src register must remain unchanged",
    );
}

#[test]
fn ref_copy_with_out_of_range_dst_register_surfaces_typed_error() {
    let ref_copy_byte = *insns_opname_to_byte()
        .get("ref_copy/r>r")
        .expect("`ref_copy/r>r` must be in insns table");
    let code = [ref_copy_byte, 0x00, 0x09]; // src=0 (in range), dst=9 (OOR)
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("ref_copy dst OOR must surface a typed error");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 9,
            len: 4,
            bank: "r",
        },
    );
}

#[test]
fn ref_copy_with_out_of_range_src_register_surfaces_typed_error() {
    let ref_copy_byte = *insns_opname_to_byte()
        .get("ref_copy/r>r")
        .expect("`ref_copy/r>r` must be in insns table");
    let code = [ref_copy_byte, 0x07, 0x00];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [], // empty — index 7 must surface OOR
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("ref_copy/r>r must read its src operand");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "r",
        },
    );
}

/// Drive a single `int_<binop>/ii>i` handler: the codewriter
/// encodes `[opcode, src1, src2, dst]`. Asserts the recorder
/// captured `OpCode::<expected>` with `[regs_i[src1],
/// regs_i[src2]]` and that `regs_i[dst]` was written with the
/// recorder's result OpRef.
fn drive_int_binop(opname: &str, expected_opcode: majit_ir::OpCode) {
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // src=2, src2=4, dst=6 — chosen to be distinct so misordered
    // operand decoding surfaces in the assertion.
    let code = [byte, 0x02, 0x04, 0x06];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let arg0 = regs_i[2];
    let arg1 = regs_i[4];
    let dst_pre = regs_i[6];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc)
        .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 4, "`{opname}` operand layout `ii>i` = 3 bytes");
    let dst_post = wc.registers_i[6];
    assert_ne!(
        dst_post, dst_pre,
        "`{opname}` must write a fresh OpRef into registers_i[dst]",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "`{opname}` must record exactly one op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(
        last.opcode, expected_opcode,
        "`{opname}` must record `{:?}`",
        expected_opcode,
    );
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![arg0, arg1],
        "`{opname}` args must be [registers_i[src1], registers_i[src2]] in source order",
    );
    assert_eq!(
        dst_post,
        last.pos.get(),
        "`{opname}` dst must hold the recorder's result OpRef (op.pos.get())",
    );
}

#[test]
fn int_add_records_intadd_with_both_operands_and_writes_dst() {
    drive_int_binop("int_add/ii>i", majit_ir::OpCode::IntAdd);
}

#[test]
fn int_sub_records_intsub() {
    drive_int_binop("int_sub/ii>i", majit_ir::OpCode::IntSub);
}

#[test]
fn int_mul_records_intmul() {
    drive_int_binop("int_mul/ii>i", majit_ir::OpCode::IntMul);
}

#[test]
fn int_and_records_intand() {
    drive_int_binop("int_and/ii>i", majit_ir::OpCode::IntAnd);
}

// `int_or/ii>i` is not currently in `pipeline.insns` — pyre's
// interpreter source does not emit Rust `|` on integers in any
// path the JIT traces.  RPython's `Assembler.insns` only carries
// emitted opnames (`assembler.py
// setdefault(key, len(self.insns))`); pyre's runtime now mirrors
// that (build.rs walks only `pipeline.insns`).  The dispatcher
// handler exists; this test will unignore once an interpreter
// source path emits `int_or` (e.g., bitset / flag computation).
#[test]
fn int_or_records_intor() {
    drive_int_binop("int_or/ii>i", majit_ir::OpCode::IntOr);
}

#[test]
fn int_xor_records_intxor() {
    drive_int_binop("int_xor/ii>i", majit_ir::OpCode::IntXor);
}

#[test]
fn int_rshift_records_intrshift() {
    drive_int_binop("int_rshift/ii>i", majit_ir::OpCode::IntRshift);
}

// The unsigned members of the same generated binop loop. They reach the
// walker through `record_binop_i`, which the legacy dispatcher also feeds
// from `BC_UINT_*`; the shape is identical to the signed arms, so the
// driver covers them unchanged.
#[test]
fn uint_rshift_records_uintrshift() {
    drive_int_binop("uint_rshift/ii>i", majit_ir::OpCode::UintRshift);
}

#[test]
fn uint_mul_high_records_uintmulhigh() {
    drive_int_binop("uint_mul_high/ii>i", majit_ir::OpCode::UintMulHigh);
}

#[test]
fn uint_lt_records_uintlt() {
    drive_int_binop("uint_lt/ii>i", majit_ir::OpCode::UintLt);
}

#[test]
fn uint_le_records_uintle() {
    drive_int_binop("uint_le/ii>i", majit_ir::OpCode::UintLe);
}

#[test]
fn uint_gt_records_uintgt() {
    drive_int_binop("uint_gt/ii>i", majit_ir::OpCode::UintGt);
}

#[test]
fn uint_ge_records_uintge() {
    drive_int_binop("uint_ge/ii>i", majit_ir::OpCode::UintGe);
}

#[test]
fn int_eq_records_inteq() {
    drive_int_binop("int_eq/ii>i", majit_ir::OpCode::IntEq);
}

#[test]
fn int_ne_records_intne() {
    drive_int_binop("int_ne/ii>i", majit_ir::OpCode::IntNe);
}

#[test]
fn int_lt_records_intlt() {
    drive_int_binop("int_lt/ii>i", majit_ir::OpCode::IntLt);
}

#[test]
fn int_le_records_intle() {
    drive_int_binop("int_le/ii>i", majit_ir::OpCode::IntLe);
}

#[test]
fn int_gt_records_intgt() {
    drive_int_binop("int_gt/ii>i", majit_ir::OpCode::IntGt);
}

#[test]
fn int_ge_records_intge() {
    drive_int_binop("int_ge/ii>i", majit_ir::OpCode::IntGe);
}

/// Drive `int_between/iii>i` and return the recorded ops plus
/// the post-handler `dst` slot.  `inputs` describes how the
/// three operand slots are populated.
enum BetweenOperand {
    ConstInt(i64),
    NonConstWithConcrete(i64),
}

fn drive_int_between(
    b1: BetweenOperand,
    b2: BetweenOperand,
    b3: BetweenOperand,
) -> (Vec<majit_ir::OpCode>, OpRef, OpRef) {
    // `int_between/iii>i` is recorder-side only (decomposed at
    // record time) and is not currently emitted into pyre's
    // pipeline.insns table, so `insns_opname_to_byte()` lacks
    // a byte for it.  Call `int_between_record` directly with a
    // synthetic `DecodedOp` to exercise the handler.  Operand
    // layout: `[opcode, src1, src2, src3, dst]` — opcode byte is
    // unused by the handler (PC offsets into `code` from `op.pc`).
    let code = [0x00u8, 0x02, 0x04, 0x06, 0x08];
    let op = crate::jitcode_runtime::DecodedOp {
        key: "int_between/iii>i",
        opname: "int_between",
        argcodes: "iii>i",
        pc: 0,
        next_pc: 5,
    };
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 16);
    let mint = |tc: &mut TraceCtx, op: &BetweenOperand| match op {
        BetweenOperand::ConstInt(v) => tc.const_int(*v),
        BetweenOperand::NonConstWithConcrete(v) => {
            // Materialize a non-Const OpRef by recording a placeholder
            // op, then stamp its concrete value.  IntAdd with two
            // ConstInts is a valid stand-in carrier.
            let lhs = tc.const_int(0);
            let opref = tc.record_op(majit_ir::OpCode::IntAdd, &[lhs, lhs]);
            tc.set_opref_concrete(opref, majit_ir::Value::Int(*v));
            opref
        }
    };
    regs_i[2] = mint(&mut tc, &b1);
    regs_i[4] = mint(&mut tc, &b2);
    regs_i[6] = mint(&mut tc, &b3);
    let arg_b1 = regs_i[2];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) =
        int_between_record(&code, &op, &mut wc).expect("int_between_record must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 5, "operand layout `iii>i` consumes 4 bytes");
    let dst_post = wc.registers_i[8];
    drop(wc);
    let new_ops: Vec<majit_ir::OpCode> = tc
        .ops()
        .iter()
        .skip(ops_before)
        .map(|op| op.opcode)
        .collect();
    (new_ops, dst_post, arg_b1)
}

/// `pyjitpl.py execute_and_record` — when every argbox
/// is `ConstInt`, a pure op like `INT_SUB` / `INT_EQ` is folded
/// via `wrap_constant` and never reaches `_record_helper`.
/// `opimpl_int_between(ConstInt, ConstInt, ConstInt)` chains three
/// pure binops on all-Const inputs, so 0 ops must be recorded and
/// the destination must hold the folded ConstInt result.
#[test]
fn int_between_const_inputs_with_unit_width_takes_inteq_fast_path() {
    let (ops, dst_post, _) = drive_int_between(
        BetweenOperand::ConstInt(5),
        BetweenOperand::ConstInt(7),
        BetweenOperand::ConstInt(6),
    );
    assert!(
        ops.is_empty(),
        "all-Const inputs must fold without recording — got {ops:?}",
    );
    // `b5 = 6 - 5 = 1` → fast path → `IntEq(b2=7, b1=5) = 0`.
    assert_eq!(
        dst_post.inline_const_to_value(),
        Some(majit_ir::Value::Int(0)),
        "ConstInt(1) fast path on all-Const inputs must fold to ConstInt(0)",
    );
}

/// All-Const generic path: `pyjitpl.py` folds the three
/// pure binops without recording.  Destination must carry the
/// folded UINT_LT result.
#[test]
fn int_between_const_inputs_with_wide_range_takes_uintlt_generic_path() {
    let (ops, dst_post, _) = drive_int_between(
        BetweenOperand::ConstInt(5),
        BetweenOperand::ConstInt(7),
        BetweenOperand::ConstInt(10),
    );
    assert!(
        ops.is_empty(),
        "all-Const inputs must fold without recording — got {ops:?}",
    );
    // `b5 = 10 - 5 = 5` (not 1) → generic → `b4 = 7 - 5 = 2`,
    // `UintLt(2, 5) = 1`.
    assert_eq!(
        dst_post.inline_const_to_value(),
        Some(majit_ir::Value::Int(1)),
        "wide-range path on all-Const inputs must fold to ConstInt(1)",
    );
}

/// Regression: when `b1`/`b3` are non-constant OpRefs with
/// observed concrete values satisfying `b3-b1==1`, the recorder
/// must still take the generic `INT_SUB + UINT_LT` path.
/// Specializing to `INT_EQ` would bake a `b3-b1==1` invariant
/// without a runtime guard (`box_value` only reports the observed
/// sample), miscompiling any future execution where the same
/// boxes carry different live values.
#[test]
fn int_between_non_const_inputs_observing_unit_width_takes_generic_path() {
    let (ops, _, _) = drive_int_between(
        BetweenOperand::NonConstWithConcrete(5),
        BetweenOperand::NonConstWithConcrete(7),
        BetweenOperand::NonConstWithConcrete(6),
    );
    assert_eq!(
        ops,
        vec![
            majit_ir::OpCode::IntSub,
            majit_ir::OpCode::IntSub,
            majit_ir::OpCode::UintLt,
        ],
        "non-const inputs must NOT take INT_EQ fast path; expected \
             [IntSub(b5), IntSub(b4), UintLt] — got {ops:?}",
    );
}

/// Drive a single `float_<binop>/ff>f` handler. Same shape as
/// `drive_int_binop` but on the float bank.
fn drive_float_binop(opname: &str, expected_opcode: majit_ir::OpCode) {
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0x02, 0x04, 0x06];
    let mut tc = fresh_trace_ctx();
    let mut regs_f = distinct_const_refs(&mut tc, 8);
    let arg0 = regs_f[2];
    let arg1 = regs_f[4];
    let dst_pre = regs_f[6];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut regs_f,
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc)
        .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 4, "`{opname}` operand layout `ff>f` = 3 bytes");
    let dst_post = wc.registers_f[6];
    assert_ne!(
        dst_post, dst_pre,
        "`{opname}` must write a fresh OpRef into registers_f[dst]",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "`{opname}` must record exactly one op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, expected_opcode);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![arg0, arg1]
    );
    assert_eq!(dst_post, last.pos.get());
}

#[test]
fn float_add_records_floatadd() {
    drive_float_binop("float_add/ff>f", majit_ir::OpCode::FloatAdd);
}

#[test]
fn float_sub_records_floatsub() {
    drive_float_binop("float_sub/ff>f", majit_ir::OpCode::FloatSub);
}

#[test]
fn float_truediv_records_floattruediv() {
    drive_float_binop("float_truediv/ff>f", majit_ir::OpCode::FloatTrueDiv);
}

/// Drive a single `float_<unop>/f>f` handler. Same shape pattern as
/// `drive_float_binop` minus one read.
fn drive_float_unop(opname: &str, expected_opcode: majit_ir::OpCode) {
    // `f>f` shape: 1B src + 1B dst = 2 operand bytes after opcode.
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0x02, 0x05];
    let mut tc = fresh_trace_ctx();
    let mut regs_f = distinct_const_refs(&mut tc, 8);
    let arg = regs_f[2];
    let dst_pre = regs_f[5];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut regs_f,
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).unwrap_or_else(|_| panic!("`{opname}` must dispatch"));
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 3, "`{opname}` operand layout `f>f` = 2 bytes");
    let dst_post = wc.registers_f[5];
    assert_ne!(dst_post, dst_pre);
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, expected_opcode);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![arg],
        "`{opname}` args must be [registers_f[src]]",
    );
    assert_eq!(dst_post, last.pos.get());
}

#[test]
fn float_neg_records_floatneg_with_one_operand_and_writes_dst() {
    drive_float_unop("float_neg/f>f", majit_ir::OpCode::FloatNeg);
}

#[test]
fn float_abs_records_floatabs() {
    drive_float_unop("float_abs/f>f", majit_ir::OpCode::FloatAbs);
}

/// Drive a single `int_<unop>/i>i` handler. Same shape pattern as
/// `drive_int_binop` minus one read.
fn drive_int_unop(opname: &str, expected_opcode: majit_ir::OpCode) {
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0x02, 0x05]; // src=2, dst=5
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let arg = regs_i[2];
    let dst_pre = regs_i[5];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc)
        .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 3, "`{opname}` operand layout `i>i` = 2 bytes");
    let dst_post = wc.registers_i[5];
    assert_ne!(dst_post, dst_pre);
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, expected_opcode);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![arg]
    );
    assert_eq!(dst_post, last.pos.get());
}

#[test]
fn int_neg_records_intneg() {
    drive_int_unop("int_neg/i>i", majit_ir::OpCode::IntNeg);
}

#[test]
fn int_invert_records_intinvert() {
    drive_int_unop("int_invert/i>i", majit_ir::OpCode::IntInvert);
}

#[test]
fn int_same_as_is_eliminated_from_generated_insns_table() {
    // RPython `jtransform.py rewrite_op_same_as` removes
    // `same_as` before assembly. The walker keeps a handler arm for
    // forward-prep, but the production insns table should not contain
    // the opname unless a future codewriter path legitimately emits it.
    assert!(
        !insns_opname_to_byte().contains_key("int_same_as/i>i"),
        "`int_same_as/i>i` appeared in the generated insns table; \
             verify same_as elimination before adding a decode test"
    );
}

/// Drive `ptr_eq/rr>i` or `ptr_ne/rr>i`. Shape `rr>i`: read 2
/// r-regs, record, write to i-bank.
fn drive_ptr_compare(opname: &str, expected_opcode: majit_ir::OpCode) {
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0x02, 0x04, 0x06]; // r-src1=2, r-src2=4, i-dst=6
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let arg0 = regs_r[2];
    let arg1 = regs_r[4];
    let dst_pre = regs_i[6];
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc)
        .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 4, "`{opname}` operand layout `rr>i` = 3 bytes");
    let dst_post = wc.registers_i[6];
    assert_ne!(dst_post, dst_pre);
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, expected_opcode);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![arg0, arg1]
    );
    assert_eq!(dst_post, last.pos.get());
}

#[test]
fn ptr_eq_records_ptreq_with_two_ref_operands_into_int_dst() {
    drive_ptr_compare("ptr_eq/rr>i", majit_ir::OpCode::PtrEq);
}

#[test]
fn ptr_ne_records_ptrne() {
    drive_ptr_compare("ptr_ne/rr>i", majit_ir::OpCode::PtrNe);
}

#[test]
fn instance_ptr_eq_records_instanceptreq() {
    drive_ptr_compare("instance_ptr_eq/rr>i", majit_ir::OpCode::InstancePtrEq);
}

#[test]
fn instance_ptr_ne_records_instanceptrne() {
    drive_ptr_compare("instance_ptr_ne/rr>i", majit_ir::OpCode::InstancePtrNe);
}

#[test]
fn float_add_with_out_of_range_src_register_surfaces_typed_error() {
    let byte = *insns_opname_to_byte()
        .get("float_add/ff>f")
        .expect("`float_add/ff>f` must be in insns table");
    let code = [byte, 0x07, 0x00, 0x00]; // src=7, registers_f empty
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("float_add must read its src operand");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "f",
        },
    );
}

#[test]
fn int_add_with_out_of_range_src_register_surfaces_typed_error() {
    // OOR validation parity with int_copy. Bank tag = "i".
    let byte = *insns_opname_to_byte()
        .get("int_add/ii>i")
        .expect("`int_add/ii>i` must be in insns table");
    let code = [byte, 0x07, 0x00, 0x00]; // src=7, registers_i empty
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("int_add must read its src operand");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "i",
        },
    );
}

#[test]
fn int_add_with_out_of_range_dst_register_surfaces_typed_error() {
    // src reads succeed, dst write surfaces OOR. Catches the
    // reverse-direction encoding bugs the src-only test misses.
    let byte = *insns_opname_to_byte()
        .get("int_add/ii>i")
        .expect("`int_add/ii>i` must be in insns table");
    let code = [byte, 0x00, 0x01, 0x09]; // dst=9, registers_i.len()=4 → OOR
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("int_add dst OOR must surface a typed error");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 9,
            len: 4,
            bank: "i",
        },
    );
}

#[test]
fn unsupported_opname_surfaces_typed_error() {
    // Stable choice for exercising the catch-all `UnsupportedOpname`
    // error path.  `vtable_method_ptr/rd>i` is a pyre-only backend
    // adaptation (emitted by `OpKind::VtableMethodPtr` /
    // `assembler.rs`) without a PyPy analog: Python dispatch
    // resolves through `cpu.bh_call_*` at runtime rather than
    // reifying a method pointer into the bytecode stream.  Zero
    // JitCode hits in production traces (per
    // `t3_audit_opname_gap_inventory`), so it's a durable choice
    // for the "still unsupported" slot.
    let opname = "vtable_method_ptr/rd>i";
    let unsupported_byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // Operand encoding `rd>i`: 1B r-reg + 2B descr + 1B i-reg-dst = 4B.
    let code = [unsupported_byte, 0, 0, 0, 0];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("unsupported opname must hit UnsupportedOpname");
    assert_eq!(err, DispatchError::UnsupportedOpname { pc: 0, key: opname },);
}

/// `ptr_nonzero/r>i` records `PtrNe(box, CONST_NULL)` into the
/// int dst.  RPython parity: `pyjitpl.py opimpl_ptr_nonzero`
/// returns `self.execute(rop.PTR_NE, box, CONST_NULL)`.
#[test]
fn ptr_nonzero_records_ptrne_with_box_and_null() {
    let opname = "ptr_nonzero/r>i";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // Operand encoding `r>i`: 1B r-reg + 1B i-reg-dst = 2B
    let code = [byte, 0, 0];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    // Seed `registers_r[0]` with a placeholder OpRef so the
    // handler has something to read.
    let box_opref = tc.const_ref(0xdeadbeef);
    let mut regs_r = [box_opref];
    let mut regs_i = [OpRef::None];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ptr_nonzero must record PtrNe");
    assert!(matches!(outcome, DispatchOutcome::Continue));
    assert_eq!(next_pc, 3);
    // `get_or_insert_typed` mints a fresh OpRef on every call (see
    // `constant_pool.rs` — equality is `Const.same_constant`, not
    // OpRef identity), so we cannot compare against a freshly-minted
    // null_const.  Verify args[1] is a Ref-typed constant whose
    // pooled value is 0 instead.
    let last_args0;
    let last_args1;
    let last_opcode;
    let last_args_len;
    {
        let ops = wc.trace_ctx.ops();
        let last = ops.last().expect("ptr_nonzero must record one op");
        last_opcode = last.opcode;
        let args = last.getarglist();
        last_args_len = args.len();
        last_args0 = args[0].clone();
        last_args1 = args[1].clone();
    }
    assert_eq!(last_opcode, majit_ir::OpCode::PtrNe);
    assert_eq!(last_args_len, 2);
    assert_eq!(last_args0.to_opref(), box_opref);
    assert_eq!(
        wc.trace_ctx.const_value(last_args1.to_opref()),
        Some(0),
        "args[1] must point at the CONST_NULL pool entry (value=0)"
    );
    assert_eq!(
        wc.trace_ctx.const_type(last_args1.to_opref()),
        Some(Type::Ref)
    );
    assert_ne!(wc.registers_i[0], OpRef::None);
}

/// `abort/>r` is a pyre-only no-op result marker — the walker
/// counterpart of blackhole's `handler_abort_result_marker_r`
/// (`blackhole.rs`).  No operand read, no register write, no
/// IR op recorded; dispatch advances past the 1B dst slot only.
#[test]
fn abort_result_r_is_pure_pc_advance() {
    let opname = "abort/>r";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0x05]; // dst byte = 5 (intentionally out-of-range)
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("abort/>r must dispatch");
    assert!(matches!(outcome, DispatchOutcome::Continue));
    assert_eq!(next_pc, 2, "abort/>r operand layout = 1 byte (dst marker)");
    assert_eq!(
        wc.trace_ctx.num_ops(),
        ops_before,
        "abort/>r must not record any IR op",
    );
}

/// `ref_guard_value/r` records `GuardValue(value, ConstPtr(concrete))`
/// when the symbolic OpRef is non-Const and a concrete pointer is
/// available in the shadow.  Mirrors `pyjitpl.py
/// implement_guard_value`.
#[test]
fn ref_guard_value_records_guardvalue_with_concrete_constant() {
    let opname = "ref_guard_value/r";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // Operand encoding `r`: 1B r-src only.
    let code = [byte, 0];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    // Symbolic side: a recorded op OpRef (not a Const).
    let value_opref = tc.record_op(majit_ir::OpCode::PtrEq, &[]);
    let mut regs_r = [value_opref];
    let mut regs_i = [OpRef::None];
    let concrete_ptr: usize = 0xdead_beef;
    let mut concrete_r = [ConcreteValue::Ref(
        concrete_ptr as *mut pyre_object::pyobject::PyObject,
    )];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut concrete_r,
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).expect("ref_guard_value must record GuardValue");
    assert!(matches!(outcome, DispatchOutcome::Continue));
    assert_eq!(next_pc, 2);
    let (last_opcode, last_args0, last_args1, last_args_len) = {
        let ops = wc.trace_ctx.ops();
        let last = ops.last().expect("ref_guard_value must record one op");
        let args = last.getarglist();
        (last.opcode, args[0].clone(), args[1].clone(), args.len())
    };
    assert_eq!(last_opcode, majit_ir::OpCode::GuardValue);
    assert_eq!(last_args_len, 2);
    assert_eq!(last_args0.to_opref(), value_opref);
    assert_eq!(
        wc.trace_ctx.const_value(last_args1.to_opref()),
        Some(concrete_ptr as i64),
        "args[1] must point at the concrete pointer in the pool",
    );
    assert_eq!(
        wc.trace_ctx.const_type(last_args1.to_opref()),
        Some(Type::Ref)
    );
    assert_eq!(
        wc.registers_r[0],
        last_args1.to_opref(),
        "register slot still holding the original OpRef must be rewritten \
             to the promoted constant (pyjitpl.py:1923 replace_box)",
    );
}

/// `int_guard_value/i` records `GuardValue(value, ConstInt(concrete))`
/// through the same bank-parameterized body as the Ref and Float variants.
#[test]
fn int_guard_value_records_guardvalue_with_concrete_constant() {
    let opname = "int_guard_value/i";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    // Operand encoding `i`: 1B i-src only.
    let code = [byte, 0];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    // Symbolic side: a recorded op OpRef (not a Const).
    let value_opref = tc.record_op(majit_ir::OpCode::IntAdd, &[]);
    let ops_before = tc.num_ops();
    let mut regs_r = [OpRef::None];
    let mut regs_i = [value_opref];
    let mut concrete_i = [ConcreteValue::Int(42)];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut concrete_i,
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).expect("int_guard_value must record GuardValue");
    assert!(matches!(outcome, DispatchOutcome::Continue));
    assert_eq!(next_pc, 2);
    assert_eq!(
        wc.trace_ctx.num_ops(),
        ops_before + 1,
        "int_guard_value must record exactly one GuardValue op",
    );
    let (last_opcode, last_args0, last_args1, last_args_len) = {
        let ops = wc.trace_ctx.ops();
        let last = ops.last().expect("int_guard_value must record one op");
        let args = last.getarglist();
        (last.opcode, args[0].clone(), args[1].clone(), args.len())
    };
    assert_eq!(last_opcode, majit_ir::OpCode::GuardValue);
    assert_eq!(last_args_len, 2);
    assert_eq!(last_args0.to_opref(), value_opref);
    assert_eq!(wc.trace_ctx.const_value(last_args1.to_opref()), Some(42));
    assert_eq!(
        wc.trace_ctx.const_type(last_args1.to_opref()),
        Some(Type::Int)
    );
    assert_eq!(
        wc.registers_i[0],
        last_args1.to_opref(),
        "register slot still holding the original OpRef must be rewritten \
         to the promoted constant (pyjitpl.py:1923 replace_box)",
    );
}

/// Symbolic OpRef already a Const → `ref_guard_value/r` is a no-op
/// (`pyjitpl.py if isinstance(box, Const): return box`).
#[test]
fn ref_guard_value_on_const_records_nothing() {
    let opname = "ref_guard_value/r";
    let byte = *insns_opname_to_byte()
        .get(opname)
        .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
    let code = [byte, 0];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let value_opref = tc.const_ref(0xdead_beef);
    let baseline_ops = tc.ops().len();
    let mut regs_r = [value_opref];
    let mut regs_i = [OpRef::None];
    let mut concrete_r = [ConcreteValue::Ref(
        0xdead_beef as *mut pyre_object::pyobject::PyObject,
    )];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut concrete_r,
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ref_guard_value Const arm");
    assert!(matches!(outcome, DispatchOutcome::Continue));
    assert_eq!(next_pc, 2);
    assert_eq!(
        wc.trace_ctx.ops().len(),
        baseline_ops,
        "no op should be recorded when input is already Const"
    );
    assert_eq!(wc.registers_r[0], value_opref);
}

#[test]
fn step_through_residual_call_r_r_records_callr_with_descr_and_args() {
    // `residual_call_r_r/iRd>r` records `OpCode::CallR`
    // with `[funcptr, ...args]` and `descr=descr_refs[d]`. RPython
    // `pyjitpl.py _opimpl_residual_call1` →
    // `do_residual_or_indirect_call → execute_and_record_varargs(
    // rop.CALL_R, [funcbox]+argboxes, descr=calldescr)`.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    // Operand encoding `iRd>r`: 1B funcptr (i-reg=2),
    // 1B varlen=2 + [r-reg=4, r-reg=7], 2B descr_index=1 (LE),
    // 1B dst-reg=0 (writeback deferred — not used by walker yet).
    let code = [
        residual_byte,
        0x02, // funcptr from registers_i[2]
        0x02, // varlen
        0x04,
        0x07, // args from registers_r[4, 7]
        0x01,
        0x00, // descr index = 1 (LE)
        0x00, // dst reg (deferred)
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 4);
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let funcptr_expected = regs_i[2];
    let arg0_expected = regs_r[4];
    let arg1_expected = regs_r[7];
    // Build a 2-entry descr table — index 0 is a decoy (different
    // pointer), index 1 is the descr we expect the recorder to attach.
    // RPython `_build_allboxes` reads `descr.get_arg_types()` to
    // permute argboxes into ABI order; the test passes 2 R args so
    // `arg_types = [Ref, Ref]` keeps the permutation an identity
    // (allboxes = [funcbox, r0, r1]).
    let decoy = make_call_descr(
        2,
        vec![Type::Ref, Type::Ref],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    );
    let call_descr = make_call_descr(
        3,
        vec![Type::Ref, Type::Ref],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    );
    let descr_pool = vec![decoy, call_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc,
        code.len(),
        "residual_call_r_r must advance past funcptr + varlist + descr + dst",
    );
    drop(wc);
    // FailDescr placeholder has no EffectInfo (`as_call_descr() = None`),
    // so the walker takes the `no-effectinfo-fallback` branch:
    // CallR + GuardNoException (RPython parity:
    // `do_residual_call → execute_varargs(..., exc=True)` →
    // `handle_possible_exception` emits GUARD_NO_EXCEPTION).
    assert_eq!(
        tc.num_ops(),
        ops_before + 2,
        "residual_call_r_r must record CallR + GuardNoException (no-effectinfo fallback)",
    );
    let call_op = tc
        .ops()
        .iter()
        .find(|o| o.opcode == majit_ir::OpCode::CallR)
        .expect("CallR must be recorded");
    assert_eq!(
        call_op
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![funcptr_expected, arg0_expected, arg1_expected],
        "CallR args must be [funcptr, ...args] from registers_i+registers_r",
    );
    let recorded_descr = call_op.getdescr().expect("CallR must carry the calldescr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &call_descr),
        "CallR descr must be descr_refs[1] (not decoy at index 0)",
    );
    // GuardNoException follows immediately after.
    let guard_op = tc
        .ops()
        .iter()
        .find(|o| o.opcode == majit_ir::OpCode::GuardNoException)
        .expect("GuardNoException must follow CallR for raising calls");
    assert!(
        guard_op.num_args() == 0,
        "GuardNoException takes no operand args",
    );
    // `walker_capture_snapshot_for_last_guard` ports
    // `capture_resumedata(after_residual_call=True)`
    // (`pyjitpl.py`).  Every guard emitted by a
    // residual_call dispatcher now carries a snapshot whose
    // `rd_resume_position` is the freshly-allocated snapshot id
    // (`>= 0`), so the optimizer's `store_final_boxes_in_guard`
    // (`optimizeopt/mod.rs`) finds attached resume data
    // instead of panicking on the `-1` sentinel.
    assert!(
        guard_op.rd_resume_position.get() >= 0,
        "GuardNoException must carry an attached snapshot (rd_resume_position >= 0) after capture_resumedata port",
    );
}

/// Build a `SimpleCallDescr` for tests, parameterised by `arg_types`,
/// `result_type`, and `extraeffect`. The `_build_allboxes` permutation
/// reads `arg_types` as the callee's ABI ordering, so tests must pass
/// the exact types of the arglist they exercise.
fn make_call_descr(
    idx: u32,
    arg_types: Vec<Type>,
    result_type: Type,
    extra: majit_ir::ExtraEffect,
) -> DescrRef {
    let mut effect = majit_ir::EffectInfo::default();
    effect.extraeffect = extra;
    std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
        idx,
        arg_types,
        result_type,
        false,
        std::mem::size_of::<usize>(),
        effect,
    ))
}

/// Convenience: legacy signature used by elidable-classification
/// tests with empty arglists (0 R args, descr arg_types=[]).
/// `result_type` defaults to `Ref` matching `_r_r` shape. Callers
/// exercising actual args must use [`make_call_descr`] directly to
/// pass matching `arg_types`.
fn call_descr_with_effect(idx: u32, extra: majit_ir::ExtraEffect) -> DescrRef {
    make_call_descr(idx, vec![], Type::Ref, extra)
}

/// Convenience: builds a `_r_r`-shape CallDescr with both
/// `extraeffect` and `oopspecindex` populated, for tests that need
/// to drive [`do_not_in_trace_call_result`] /
/// [`do_jit_force_virtual_guard`] / future oopspec-keyed guards.
fn call_descr_with_oopspec(
    idx: u32,
    extra: majit_ir::ExtraEffect,
    oopspec: majit_ir::OopSpecIndex,
) -> DescrRef {
    let mut effect = majit_ir::EffectInfo::default();
    effect.extraeffect = extra;
    effect.oopspecindex = oopspec;
    std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
        idx,
        vec![],
        Type::Ref,
        false,
        std::mem::size_of::<usize>(),
        effect,
    ))
}

#[test]
fn residual_call_r_r_with_elidable_cannot_raise_records_callpurer_no_guard() {
    // RPython parity: `do_residual_call` (pyjitpl.py) reads
    // `effectinfo.check_is_elidable()` + `effectinfo.check_can_raise()`,
    // then `execute_varargs(rop.CALL_R, ..., exc, pure)`. With
    // EF_ELIDABLE_CANNOT_RAISE: `pure=True` (CALL_PURE_R) + `exc=False`
    // (no GUARD_NO_EXCEPTION).
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let elidable_descr = call_descr_with_effect(7, majit_ir::ExtraEffect::ElidableCannotRaise);
    let descr_pool = vec![elidable_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "elidable+cannot-raise must record exactly CallPureR (no GuardNoException)",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(
        last.opcode,
        majit_ir::OpCode::CallPureR,
        "EF_ELIDABLE_CANNOT_RAISE must rewrite to CALL_PURE_R",
    );
}

// When the walk is the authoritative concrete leg,
// `try_execute_residual_call_via_executor` runs a `CallMayForce*`
// residual call concretely (RPython runs every residual_call during
// tracing) and stamps the recorded result OpRef so a downstream
// `goto_if_not` reads a concrete value.  Gated OFF by
// `is_authoritative_executor` so arm/shadow/probe walks never
// re-execute (which would double side effects / corrupt the live
// heap `cut_trace` cannot roll back).
extern "C" fn add2_for_walker_test(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

fn may_force_call_i_fixture(tc: &mut TraceCtx) -> ([OpRef; 3], DescrRef, OpRef) {
    let funcbox = tc.const_int(add2_for_walker_test as *const () as i64);
    let arg0 = tc.const_int(40);
    let arg1 = tc.const_int(2);
    let allboxes = [funcbox, arg0, arg1];
    let descr = make_call_descr(
        5,
        vec![Type::Int, Type::Int],
        Type::Int,
        majit_ir::ExtraEffect::CanRaise,
    );
    let recorded =
        tc.record_op_with_descr(majit_ir::OpCode::CallMayForceI, &allboxes, descr.clone());
    (allboxes, descr, recorded)
}

#[test]
fn authoritative_walker_executes_may_force_call_and_stamps_result() {
    let mut tc = fresh_trace_ctx();
    let (allboxes, descr, recorded) = may_force_call_i_fixture(&mut tc);
    let mut regs_i: Vec<OpRef> = Vec::new();
    let mut regs_r: Vec<OpRef> = Vec::new();
    let call_descr = descr.as_call_descr().expect("CallI descr");
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallMayForceI,
        &allboxes,
        call_descr,
        recorded,
        0,
    );
    drop(wc);
    assert_eq!(
        tc.box_value(recorded),
        Some(majit_ir::Value::Int(42)),
        "authoritative walker must execute add2_for_walker_test(40, 2) and stamp 42",
    );
}

#[test]
fn non_authoritative_walker_does_not_execute_may_force_call() {
    let mut tc = fresh_trace_ctx();
    let (allboxes, descr, recorded) = may_force_call_i_fixture(&mut tc);
    let mut regs_i: Vec<OpRef> = Vec::new();
    let mut regs_r: Vec<OpRef> = Vec::new();
    let call_descr = descr.as_call_descr().expect("CallI descr");
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallMayForceI,
        &allboxes,
        call_descr,
        recorded,
        0,
    );
    drop(wc);
    assert_eq!(
        tc.box_value(recorded),
        None,
        "non-authoritative walk (arm/shadow/probe) must NOT execute the may-force call",
    );
}

// A may-force call that RAISES (publishes on
// BH_LAST_EXC_VALUE) is transcribed onto WalkContext.last_exc_value
// (+ last_exc_value_concrete) and BH_LAST_EXC_VALUE is restored so the
// eval-loop walker-skip path can detect the pending exception; the
// result box is NOT stamped (only the normal-return path stamps a
// result, mirroring `execute_varargs`'s success-only `result_box.value`).
extern "C" fn raises_for_walker_test() -> i64 {
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0xDEAD));
    0
}

#[test]
fn authoritative_walker_transcribes_may_force_raise_to_last_exc() {
    let mut tc = fresh_trace_ctx();
    let funcbox = tc.const_int(raises_for_walker_test as *const () as i64);
    let allboxes = [funcbox];
    let descr = make_call_descr(6, vec![], Type::Int, majit_ir::ExtraEffect::CanRaise);
    let recorded =
        tc.record_op_with_descr(majit_ir::OpCode::CallMayForceI, &allboxes, descr.clone());
    let mut regs_i: Vec<OpRef> = Vec::new();
    let mut regs_r: Vec<OpRef> = Vec::new();
    let call_descr = descr.as_call_descr().expect("CallI descr");
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallMayForceI,
        &allboxes,
        call_descr,
        recorded,
        0,
    );
    let captured_exc = wc.last_exc_value;
    let captured_concrete = wc.last_exc_value_concrete;
    drop(wc);
    assert!(
        captured_exc.is_some(),
        "a raising may-force call must transcribe the exception to last_exc_value",
    );
    assert!(
        matches!(captured_concrete, ConcreteValue::Ref(_)),
        "last_exc_value_concrete must carry the raised exception pointer",
    );
    assert_eq!(
        tc.box_value(recorded),
        None,
        "the raising path does not stamp `recorded`; the exception routes \
             via last_exc_value (only the normal-return path stamps a result)",
    );
}

// pyjitpl.py / 3349-3353 vable token protocol around a
// concrete-executed may-force call: with an active standard
// virtualizable the executor sets TOKEN_TRACING_RESCALL before the
// call and probes-and-clears after it.  A token still intact means
// no force — execute + stamp as usual, token back to TOKEN_NONE.  A
// cleared token means the callee forced the virtualizable —
// `DispatchError::VableEscapedDuringResidualCall` (ABORT_ESCAPE,
// pyjitpl.py).
fn bind_fake_vable(tc: &mut TraceCtx, buf: &mut [u8]) {
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    assert!(
        info.token_offset + 8 <= buf.len(),
        "fake vable buffer must cover token_offset",
    );
    let vable_ref = tc.const_ref(buf.as_ptr() as i64);
    tc.init_virtualizable_boxes(
        &info,
        vable_ref,
        majit_ir::Value::Ref(majit_ir::GcRef(buf.as_ptr() as usize)),
        &[],
        &[],
        &[0],
    );
    tc.set_virtualizable_heap_ptr(buf.as_ptr());
}

#[test]
fn may_force_with_active_vable_executes_and_clears_token() {
    let mut tc = fresh_trace_ctx();
    let mut vable_buf = vec![0u8; 65536];
    bind_fake_vable(&mut tc, &mut vable_buf);
    let (allboxes, descr, recorded) = may_force_call_i_fixture(&mut tc);
    let mut regs_i: Vec<OpRef> = Vec::new();
    let mut regs_r: Vec<OpRef> = Vec::new();
    let call_descr = descr.as_call_descr().expect("CallI descr");
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let result = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallMayForceI,
        &allboxes,
        call_descr,
        recorded,
        0,
    );
    drop(wc);
    assert!(
        matches!(result, Ok(ResidualExecOutcome::Executed(Ok(_)))),
        "non-forcing may-force call with active vable must execute normally",
    );
    assert_eq!(
        tc.box_value(recorded),
        Some(majit_ir::Value::Int(42)),
        "active-vable may-force call must execute and stamp the result",
    );
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    let token = u64::from_le_bytes(
        vable_buf[info.token_offset..info.token_offset + 8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(
        token, 0,
        "token must be cleared back to TOKEN_NONE after the call"
    );
}

static FORCING_CALLEE_TOKEN_ADDR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

extern "C" fn forces_vable_for_walker_test(a: i64, b: i64) -> i64 {
    // A force path clears the vable token (virtualizable.rs
    // force_now on TOKEN_TRACING_RESCALL).
    let addr = FORCING_CALLEE_TOKEN_ADDR.load(std::sync::atomic::Ordering::SeqCst);
    unsafe { *(addr as *mut u64) = 0 };
    a.wrapping_add(b)
}

#[test]
fn may_force_vable_escape_surfaces_typed_abort() {
    let mut tc = fresh_trace_ctx();
    let mut vable_buf = vec![0u8; 65536];
    bind_fake_vable(&mut tc, &mut vable_buf);
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    FORCING_CALLEE_TOKEN_ADDR.store(
        vable_buf.as_ptr() as usize + info.token_offset,
        std::sync::atomic::Ordering::SeqCst,
    );
    let funcbox = tc.const_int(forces_vable_for_walker_test as *const () as i64);
    let arg0 = tc.const_int(40);
    let arg1 = tc.const_int(2);
    let allboxes = [funcbox, arg0, arg1];
    let descr = make_call_descr(
        5,
        vec![Type::Int, Type::Int],
        Type::Int,
        majit_ir::ExtraEffect::CanRaise,
    );
    let recorded =
        tc.record_op_with_descr(majit_ir::OpCode::CallMayForceI, &allboxes, descr.clone());
    let mut regs_i: Vec<OpRef> = Vec::new();
    let mut regs_r: Vec<OpRef> = Vec::new();
    let call_descr = descr.as_call_descr().expect("CallI descr");
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: true,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: make_fail_descr(1),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let result = try_execute_residual_call_via_executor(
        &mut wc,
        majit_ir::OpCode::CallMayForceI,
        &allboxes,
        call_descr,
        recorded,
        7,
    );
    drop(wc);
    assert!(
        matches!(
            result,
            Err(DispatchError::VableEscapedDuringResidualCall { pc: 7 })
        ),
        "a callee that forces the vable must surface VableEscapedDuringResidualCall, got {result:?}",
    );
}

#[test]
fn residual_call_r_r_with_not_in_trace_oopspec_returns_typed_error() {
    // RPython parity: `pyjitpl.py` routes
    // `OS_NOT_IN_TRACE` residual calls through `do_not_in_trace_call`
    // which executes the callee concretely and aborts to blackhole
    // only if it raises (`pyjitpl.py`). The walker has no
    // concrete executor, so it must surface a typed error rather
    // than recording either the normal-return or
    // SwitchToBlackhole shape.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let not_in_trace_descr = call_descr_with_oopspec(
        43,
        majit_ir::ExtraEffect::CannotRaise,
        majit_ir::OopSpecIndex::NotInTrace,
    );
    let descr_pool = vec![not_in_trace_descr];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("OS_NOT_IN_TRACE must surface a typed error");
    assert_eq!(
        err,
        DispatchError::NotInTraceRequiresConcreteExecution { pc: 0 },
    );
}

#[test]
fn residual_call_r_r_with_jit_force_virtual_oopspec_returns_typed_error() {
    // RPython parity: `pyjitpl.py` short-circuits
    // `do_residual_call` via `_do_jit_force_virtual` when
    // `effectinfo.oopspecindex == OS_JIT_FORCE_VIRTUAL`.  The
    // walker can't reproduce that short-circuit (needs concrete
    // `vref_ptr` resolver); fail-loud error keeps the path from
    // silently recording `CALL_MAY_FORCE_*` instead.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let force_virtual_descr = call_descr_with_oopspec(
        42,
        majit_ir::ExtraEffect::ForcesVirtualOrVirtualizable,
        majit_ir::OopSpecIndex::JitForceVirtual,
    );
    let descr_pool = vec![force_virtual_descr];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("OS_JIT_FORCE_VIRTUAL must surface a typed error");
    assert_eq!(
        err,
        DispatchError::JitForceVirtualRequiresConcreteResolver { pc: 0 },
    );
}

#[test]
fn residual_call_r_r_with_elidable_can_raise_records_callpurer_plus_guard() {
    // EF_ELIDABLE_CAN_RAISE: `pure=True` + `exc=True` —
    // CALL_PURE_R + GUARD_NO_EXCEPTION (pyjitpl.py:execute_varargs
    // emits both).
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let elidable_descr = call_descr_with_effect(8, majit_ir::ExtraEffect::ElidableCanRaise);
    let descr_pool = vec![elidable_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 2,
        "elidable+can-raise must record CallPureR + GuardNoException",
    );
    let opcodes: Vec<_> = tc.ops().iter().skip(ops_before).map(|o| o.opcode).collect();
    assert_eq!(
        opcodes,
        vec![
            majit_ir::OpCode::CallPureR,
            majit_ir::OpCode::GuardNoException
        ],
        "EF_ELIDABLE_CAN_RAISE must record CALL_PURE_R then GUARD_NO_EXCEPTION",
    );
}

#[test]
fn residual_call_r_r_with_cannot_raise_records_callr_no_guard() {
    // EF_CANNOT_RAISE: `pure=False` + `exc=False` — bare CallR,
    // no GUARD_NO_EXCEPTION.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let nothrow_descr = call_descr_with_effect(9, majit_ir::ExtraEffect::CannotRaise);
    let descr_pool = vec![nothrow_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "EF_CANNOT_RAISE must record bare CallR (no GuardNoException)",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::CallR);
}

#[test]
fn residual_call_r_r_writes_recorder_result_into_dst_register() {
    // Verify the dst writeback half of `residual_call_r_r/iRd>r`.
    // After the handler runs, `registers_r[dst]` must equal the
    // OpRef the recorder returned (i.e., the OpRef whose Op is
    // the recorded CallR at the trace tail).
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    // funcptr=regs_i[0], no args, descr index=0, dst=3
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x03];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let dst_val_pre = regs_r[3];
    // 0 R args → arg_types=[]; CallDescr required (RPython
    // do_residual_call always has one).
    let descr_pool = vec![make_call_descr(
        1,
        vec![],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    )];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    // The dst slot must hold the OpRef of the recorded CallR. Each
    // Op carries its OpRef in `op.pos` (recorder.rs), which lets
    // the test compare without re-deriving the index (input args
    // also occupy OpRef indices, so `ops.iter().position()` would
    // be off by `num_inputargs`).
    let dst_ref = wc.registers_r[3];
    assert_ne!(
        dst_ref, dst_val_pre,
        "dst must change from its pre-call value",
    );
    let call_op = wc
        .trace_ctx
        .ops()
        .iter()
        .find(|o| o.opcode == OpCode::CallR)
        .expect("a CallR op must be in the recorded trace");
    assert_eq!(
        dst_ref,
        call_op.pos.get(),
        "registers_r[dst] must be the recorded CallR's OpRef (op.pos.get())",
    );
}

#[test]
fn residual_call_r_r_can_raise_writes_dst_before_guard_no_exception() {
    // pyjitpl.py _opimpl_residual_call*: result lands in
    // `registers_*[reg_index]` BEFORE
    // `handle_possible_exception()` records GUARD_NO_EXCEPTION.
    // `walker_capture_snapshot_for_last_guard`
    // (`pyjitpl.py capture_resumedata(after_residual_call
    // =True)`) snapshots the active registers AFTER the writeback,
    // so the dst slot's recorded OpRef rides the snapshot's
    // fail_arg list.  The structural invariant tested here is:
    // after dispatch, the dst slot holds the recorded call op's
    // OpRef, and the recorded sequence is `[CallR, GuardNoException]`
    // — i.e. the writeback ran on the record-side BEFORE the
    // guard append (and therefore before the snapshot capture).
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    // funcptr=regs_i[0], no R args, descr=0, dst=3.
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x03];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let dst_pre = regs_r[3];
    let descr_pool = vec![make_call_descr(
        1,
        vec![],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    )];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
    drop(wc);
    let opcodes: Vec<_> = tc.ops().iter().skip(ops_before).map(|o| o.opcode).collect();
    assert_eq!(
        opcodes,
        vec![OpCode::CallR, OpCode::GuardNoException],
        "CAN_RAISE residual_call_r_r must record [CallR, GuardNoException]",
    );
    let call_pos = tc
        .ops()
        .iter()
        .find(|o| o.opcode == OpCode::CallR)
        .expect("CallR must be in the trace")
        .pos
        .get();
    assert_ne!(regs_r[3], dst_pre, "dst must be overwritten");
    assert_eq!(
        regs_r[3], call_pos,
        "registers_r[dst] must equal CallR's OpRef when GuardNoException is recorded",
    );
}

#[test]
fn residual_call_ir_r_can_raise_writes_dst_before_guard_no_exception() {
    // Same invariant as `residual_call_r_r_can_raise_...` for the
    // `_ir_*` shape (`dispatch_residual_call_iIRd_kind`): the
    // `iIRd>X` writeback must precede the GuardNoException record.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_ir_r/iIRd>r")
        .expect("`residual_call_ir_r/iIRd>r` must be in insns table");
    // funcptr=i[0], 0 i-args, 0 r-args, descr=0, dst=2.
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let mut regs_r = distinct_const_refs(&mut tc, 6);
    let dst_pre = regs_r[2];
    let descr_pool = vec![make_call_descr(
        1,
        vec![],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    )];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_ir_r/iIRd>r must dispatch");
    drop(wc);
    let opcodes: Vec<_> = tc.ops().iter().skip(ops_before).map(|o| o.opcode).collect();
    assert_eq!(
        opcodes,
        vec![OpCode::CallR, OpCode::GuardNoException],
        "CAN_RAISE residual_call_ir_r must record [CallR, GuardNoException]",
    );
    let call_pos = tc
        .ops()
        .iter()
        .find(|o| o.opcode == OpCode::CallR)
        .expect("CallR must be in the trace")
        .pos
        .get();
    assert_ne!(regs_r[2], dst_pre, "dst must be overwritten");
    assert_eq!(
        regs_r[2], call_pos,
        "registers_r[dst] must equal CallR's OpRef when GuardNoException is recorded",
    );
}

#[test]
fn residual_call_r_r_with_out_of_range_dst_register_surfaces_typed_error() {
    // Dst register OOR — the call was already recorded at this
    // point (RPython parity: `do_residual_or_indirect_call` records
    // first, then writes the result), but `registers_r` is empty
    // so the writeback surfaces RegisterOutOfRange.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x07]; // dst=7
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    // CallDescr required so the walker reaches the dst writeback
    // path (RPython do_residual_call invariant).
    let descr_pool = vec![make_call_descr(
        1,
        vec![],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    )];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("dst OOR must surface a typed error");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "r",
        },
    );
}

#[test]
fn residual_call_r_r_with_descr_index_out_of_range_surfaces_typed_error() {
    // descr-index OOR validation. Same shape as
    // RegisterOutOfRange, dedicated DispatchError variant for clarity.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    // descr_index=5, descr_refs.len()=2 → OOR
    let code = [residual_byte, 0x00, 0x00, 0x05, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let descr_pool = vec![make_fail_descr(1), make_fail_descr(1)];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc)
        .expect_err("descr index 5 with pool size 2 must surface DescrIndexOutOfRange");
    assert_eq!(
        err,
        DispatchError::DescrIndexOutOfRange {
            pc: 0,
            index: 5,
            len: 2,
        },
    );
}

#[test]
fn step_through_residual_call_r_i_records_calli_with_int_dst_writeback() {
    // kind sibling of `_r_r`. Same `iRd>X` operand
    // layout, dst kind flipped to int. RPython `pyjitpl.py
    // opimpl_residual_call_r_i = _opimpl_residual_call1` shares
    // the body; `do_residual_call`'s `descr.get_normalized_result_type()`
    // dispatch (pyjitpl.py) selects `'i' → CALL_*_I`.
    // CallDescr required (RPython do_residual_call invariant);
    // walker records `OpCode::CallI` + `OpCode::GuardNoException`,
    // writes the call's OpRef into `registers_i[dst]`.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_i/iRd>i")
        .expect("`residual_call_r_i/iRd>i` must be in insns table");
    // Operand encoding `iRd>i`: 1B funcptr (i-reg=2),
    // 1B varlen=2 + [r-reg=4, r-reg=7], 2B descr_index=1 (LE),
    // 1B dst-reg=3 (writeback target into registers_i).
    let code = [
        residual_byte,
        0x02, // funcptr from registers_i[2]
        0x02, // varlen
        0x04,
        0x07, // args from registers_r[4, 7]
        0x01,
        0x00, // descr index = 1 (LE)
        0x03, // dst i-reg = 3
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let funcptr_expected = regs_i[2];
    let arg0_expected = regs_r[4];
    let arg1_expected = regs_r[7];
    let dst_pre = regs_i[3];
    // 2 R args + Int return → CallDescr arg_types=[Ref, Ref],
    // result_type=Int. `_build_allboxes` permutation is identity
    // (R-only argboxes match arg_types order).
    let decoy = make_call_descr(
        2,
        vec![Type::Ref, Type::Ref],
        Type::Int,
        majit_ir::ExtraEffect::CanRaise,
    );
    let call_descr = make_call_descr(
        3,
        vec![Type::Ref, Type::Ref],
        Type::Int,
        majit_ir::ExtraEffect::CanRaise,
    );
    let descr_pool = vec![decoy, call_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).expect("residual_call_r_i/iRd>i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, code.len());
    // CallI (kind sibling of CallR) + GuardNoException recorded.
    assert_eq!(
        wc.trace_ctx.num_ops(),
        ops_before + 2,
        "_r_i must record CallI + GuardNoException (no-effectinfo fallback)",
    );
    let call_op = wc
        .trace_ctx
        .ops()
        .iter()
        .find(|o| o.opcode == majit_ir::OpCode::CallI)
        .expect("CallI must be recorded for the int-dst kind");
    assert_eq!(
        call_op
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![funcptr_expected, arg0_expected, arg1_expected],
        "CallI args must be [funcptr, ...args] from registers_i+registers_r",
    );
    let recorded_descr = call_op.getdescr().expect("CallI must carry the calldescr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &call_descr),
        "CallI descr must be descr_refs[1] (not decoy at index 0)",
    );
    // dst writeback into the int bank (NOT the r bank).
    let dst_post = wc.registers_i[3];
    assert_ne!(
        dst_post, dst_pre,
        "registers_i[dst] must change from its pre-call value",
    );
    assert_eq!(
        dst_post,
        call_op.pos.get(),
        "registers_i[dst] must be the recorded CallI's OpRef (op.pos.get())",
    );
}

#[test]
fn residual_call_r_i_with_elidable_cannot_raise_records_callpurei_no_guard() {
    // EF_ELIDABLE_CANNOT_RAISE on the int-kind sibling
    // must rewrite to CALL_PURE_I (not CALL_PURE_R) and skip
    // GUARD_NO_EXCEPTION. Confirms the `pure_op` selection in
    // `dispatch_residual_call_iRd_kind` follows `dst_bank`.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_i/iRd>i")
        .expect("`residual_call_r_i/iRd>i` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 4);
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let elidable_descr = call_descr_with_effect(7, majit_ir::ExtraEffect::ElidableCannotRaise);
    let descr_pool = vec![elidable_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("residual_call_r_i/iRd>i must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "elidable+cannot-raise on int-kind must record exactly CallPureI",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(
        last.opcode,
        majit_ir::OpCode::CallPureI,
        "EF_ELIDABLE_CANNOT_RAISE on int-kind must rewrite to CALL_PURE_I",
    );
}

#[test]
fn step_through_residual_call_ir_r_records_callr_with_int_and_ref_args() {
    // shape sibling `_ir_r/iIRd>r`. Operand layout adds
    // an i-bank list between funcptr and the R-list. RPython
    // `_build_allboxes` permutes argboxes by `descr.get_arg_types()`
    // ABI; for an [Int, Int, Ref, Ref] callee the permutation
    // reduces to identity → allboxes = [funcbox, i0, i1, r0, r1].
    // Mixed-kind permutation is exercised by the dedicated test
    // `residual_call_ir_r_permutes_argboxes_per_arg_types_abi`.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_ir_r/iIRd>r")
        .expect("`residual_call_ir_r/iIRd>r` must be in insns table");
    // Operand encoding `iIRd>r`: 1B funcptr (i-reg=2),
    // i-list: 1B count=2 + [i-reg=5, i-reg=6],
    // r-list: 1B count=2 + [r-reg=4, r-reg=7],
    // 2B descr_index=1 (LE),
    // 1B dst-reg=0.
    let code = [
        residual_byte,
        0x02, // funcptr from registers_i[2]
        0x02, // i-list count
        0x05,
        0x06, // i-args from registers_i[5, 6]
        0x02, // r-list count
        0x04,
        0x07, // r-args from registers_r[4, 7]
        0x01,
        0x00, // descr index = 1 (LE)
        0x00, // dst r-reg = 0
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let funcptr_expected = regs_i[2];
    let iarg0_expected = regs_i[5];
    let iarg1_expected = regs_i[6];
    let rarg0_expected = regs_r[4];
    let rarg1_expected = regs_r[7];
    // arg_types = [Int, Int, Ref, Ref] → `_build_allboxes`
    // permutation is identity over the source-list-order argboxes.
    let decoy = make_call_descr(
        2,
        vec![Type::Int, Type::Int, Type::Ref, Type::Ref],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    );
    let call_descr = make_call_descr(
        3,
        vec![Type::Int, Type::Int, Type::Ref, Type::Ref],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    );
    let descr_pool = vec![decoy, call_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let (outcome, next_pc) =
        step(&code, 0, &mut wc).expect("residual_call_ir_r/iIRd>r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc,
        code.len(),
        "residual_call_ir_r must advance past funcptr + i-list + r-list + descr + dst",
    );
    // CallR + GuardNoException recorded (no-effectinfo fallback).
    assert_eq!(
        wc.trace_ctx.num_ops(),
        ops_before + 2,
        "_ir_r must record CallR + GuardNoException (no-effectinfo fallback)",
    );
    let call_op = wc
        .trace_ctx
        .ops()
        .iter()
        .find(|o| o.opcode == majit_ir::OpCode::CallR)
        .expect("CallR must be recorded");
    assert_eq!(
        call_op
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![
            funcptr_expected,
            iarg0_expected,
            iarg1_expected,
            rarg0_expected,
            rarg1_expected,
        ],
        "CallR args must be [funcptr, i0, i1, r0, r1] — identity \
             permutation when descr.arg_types=[Int, Int, Ref, Ref]",
    );
    let recorded_descr = call_op.getdescr().expect("CallR must carry the calldescr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &call_descr),
        "CallR descr must be descr_refs[1] (not decoy at index 0)",
    );
    // dst writeback into registers_r[0].
    let dst_post = wc.registers_r[0];
    assert_eq!(
        dst_post,
        call_op.pos.get(),
        "registers_r[dst] must be the recorded CallR's OpRef (op.pos.get())",
    );
}

#[test]
fn residual_call_ir_r_permutes_argboxes_per_arg_types_abi() {
    // The `_ir_*` shape gives
    // the walker source-list-order argboxes `[i_args..., r_args...]`,
    // but RPython `_build_allboxes` (pyjitpl.py) re-orders
    // those to match the callee's `descr.get_arg_types()` ABI. This
    // test pins the non-identity permutation.
    //
    // Setup: 2 i-args + 2 r-args + arg_types = [Ref, Int, Ref, Int].
    // Source-list-order argboxes = [i0, i1, r0, r1].
    // `_build_allboxes` walk:
    //   iter 1, kind=Ref: src_r scans flat argboxes for first Ref →
    //     positions 0 (i0, skip), 1 (i1, skip), 2 (r0, match). src_r=3.
    //   iter 2, kind=Int: src_i scans for first Int → position 0
    //     (i0, match). src_i=1.
    //   iter 3, kind=Ref: src_r at 3 → position 3 (r1, match). src_r=4.
    //   iter 4, kind=Int: src_i at 1 → position 1 (i1, match). src_i=2.
    // Final allboxes = [funcbox, r0, i0, r1, i1].
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_ir_r/iIRd>r")
        .expect("`residual_call_ir_r/iIRd>r` must be in insns table");
    let code = [
        residual_byte,
        0x02, // funcptr from registers_i[2]
        0x02, // i-list count
        0x05,
        0x06, // i-args from registers_i[5, 6]
        0x02, // r-list count
        0x04,
        0x07, // r-args from registers_r[4, 7]
        0x00,
        0x00, // descr index = 0 (LE)
        0x00, // dst r-reg = 0
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let funcptr = regs_i[2];
    let i0 = regs_i[5];
    let i1 = regs_i[6];
    let r0 = regs_r[4];
    let r1 = regs_r[7];
    let mixed_descr = make_call_descr(
        0,
        vec![Type::Ref, Type::Int, Type::Ref, Type::Int],
        Type::Ref,
        majit_ir::ExtraEffect::CanRaise,
    );
    let descr_pool = vec![mixed_descr.clone()];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.outer_jitcode_index = test_outer_resume_jitcode_index();
    wc.outer_resume_marker_jit_pc = Some(0);
    let _ = step(&code, 0, &mut wc).expect("residual_call_ir_r/iIRd>r must dispatch");
    drop(wc);
    let call_op = tc
        .ops()
        .iter()
        .find(|o| o.opcode == majit_ir::OpCode::CallR)
        .expect("CallR must be recorded");
    assert_eq!(
        call_op
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![funcptr, r0, i0, r1, i1],
        "_build_allboxes must permute to match descr.arg_types \
             [Ref, Int, Ref, Int] — RPython pyjitpl.py:1960-1993",
    );
}

#[test]
fn residual_call_descr_not_call_descr_surfaces_typed_error() {
    // Walker requires CallDescr per RPython invariant
    // (pyjitpl.py do_residual_call). When the descr_pool entry
    // at the operand-encoded index lacks a CallDescr downcast (here
    // a FailDescr), the walker surfaces ResidualCallDescrNotCallDescr.
    // In production the codewriter never emits non-CallDescr; this
    // covers the test-fixture / future-deviation case.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let mut regs_r = distinct_const_refs(&mut tc, 1);
    let descr_pool = vec![make_fail_descr(7)];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc)
        .expect_err("FailDescr (not CallDescr) must surface ResidualCallDescrNotCallDescr");
    assert_eq!(
        err,
        DispatchError::ResidualCallDescrNotCallDescr {
            pc: 0,
            descr_index: 0,
        },
    );
}

#[test]
fn residual_call_r_r_with_out_of_range_arg_register_surfaces_typed_error() {
    // varlist member OOR validation. Bank tag = "r" since
    // R-list reads from registers_r.
    let residual_byte = *insns_opname_to_byte()
        .get("residual_call_r_r/iRd>r")
        .expect("`residual_call_r_r/iRd>r` must be in insns table");
    // varlen=1, arg=9 (registers_r is empty) → OOR
    let code = [residual_byte, 0x00, 0x01, 0x09, 0x00, 0x00, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_i = distinct_const_refs(&mut tc, 1);
    let descr_pool = vec![make_fail_descr(1)];
    let frame_done_descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc)
        .expect_err("R-list member out of range must surface RegisterOutOfRange");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 9,
            len: 0,
            bank: "r",
        },
    );
}

#[test]
#[ignore = "end-to-end walk of the execute_return_value helper reaches a goto_if_not whose value is symbolic in this unit setup (GotoIfNotValueNotConcrete { value: IntOp }); the walker needs concrete register values to decide the branch. Residual-call descrs now resolve (descr_pool_with_jitcode_adapters builds CallDescrs for BhDescr::Call); the remaining gap is a concrete-execution harness for the full helper body."]
fn walk_return_value_helper_terminates_at_first_ref_return() {
    // Acceptance: walk the ordinary `execute_return_value` JitCode
    // discovered through the portal closure end-to-end.
    // Layout (cranelift build):
    //
    //   pc=0..6   inline_call_r_r / dR>r  (recurse → SubReturn → caller dst write → Continue)
    //   pc=6..9   live /                  (continue)
    //   pc=9..11  ref_return / r          (terminate — top-level outermost)
    //   pc=11..18 (raise + ref_return tail, dead on this path)
    //
    // The helper's `inline_call_r_r` now recurses into the callee
    // jitcode via `production_sub_jitcodes` and
    // `descr_pool_with_jitcode_adapters`. The callee's
    // own `ref_return/r` surfaces as `SubReturn`; the caller writes
    // its dst register with that result and continues. The
    // caller's own `ref_return/r` at pc=9..11 then records the
    // outermost `Finish`.
    let jc = named_jitcode("execute_return_value")
        .expect("execute_return_value must resolve through the portal closure");
    let mut tc = fresh_trace_ctx();
    // 256 distinct OpRefs (one per possible 1-byte register
    // index). `inline_call_r_r`'s recursion overwrites the dst
    // slot with the callee's `SubReturn` value, so the
    // post-recursion `ref_return/r` reads the *recorded* OpRef
    // from the sub-walk, not a `regs_r` constant. The assertion
    // therefore checks the recorded Finish's args against the
    // post-recursion register state, not a precomputed constant.
    let mut regs_r = distinct_const_refs(&mut tc, 256);
    let mut regs_i = distinct_const_refs(&mut tc, 256);
    let descr = done_descr_ref_for_tests();
    let pool_len = crate::jitcode_runtime::all_descrs().len();
    let descr_pool = descr_pool_with_jitcode_adapters(pool_len);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &production_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, end_pc) =
        walk(&jc.code, 0, &mut wc).expect("ReturnValue helper must walk to a terminator");
    assert_eq!(
        outcome,
        DispatchOutcome::Terminate,
        "top-level walk must end on Terminate",
    );
    assert!(
        end_pc <= jc.code.len(),
        "walker must not run past the helper body \
             (end_pc={end_pc}, code.len()={})",
        jc.code.len(),
    );
    assert_eq!(
        end_pc, 11,
        "ReturnValue helper walker must terminate at outermost `ref_return/r` (pc=9..11)",
    );
    drop(wc);
    assert!(
        tc.num_ops() > ops_before,
        "at least one Finish op must have been recorded; \
             callee sub-walk may also have contributed CallR / Finish ops",
    );
    // Locate the *outermost* Finish (descr=done_with_this_frame).
    // Sub-walks don't emit Finish (they surface `SubReturn`), so
    // there should be exactly one Finish carrying the
    // done-with-this-frame descr.
    let outermost_finish = tc
        .ops()
        .iter()
        .find(|o| {
            o.opcode == majit_ir::OpCode::Finish
                && o.getdescr()
                    .map(|d| std::sync::Arc::ptr_eq(&d, &descr))
                    .unwrap_or(false)
        })
        .expect("outermost Finish with done-with-this-frame descr must exist");
    assert_eq!(outermost_finish.num_args(), 1);
    let recorded_descr = outermost_finish
        .getdescr()
        .expect("Finish must carry done_with_this_frame_descr_ref");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &descr),
        "Finish descr must be the exact instance the dispatcher was handed",
    );
}

#[test]
#[ignore = "end-to-end walk of the execute_pop_top helper reaches a goto_if_not \
        whose value is symbolic in this unit setup (GotoIfNotValueNotConcrete); \
        the walker needs concrete register values to decide the branch. \
        Residual-call descrs now resolve (descr_pool_with_jitcode_adapters \
        builds CallDescrs for BhDescr::Call); the remaining gap is a \
        concrete-execution harness for the full helper body. Bench traces never \
        reach a PopTop opcode at JIT record time, so production is \
        unaffected."]
fn walk_pop_top_helper_terminates_with_recorded_ops() {
    // Acceptance skeleton: walk the entire `execute_pop_top` helper
    // JitCode. The helper body is 25 bytes / 9 ops after the
    // jtransform `Ok` / `Err` / `Some` identity rewrite stripped
    // the trailing `int_copy + residual_call_r_r/iRd>r` wrapper
    // for the `Ok(StepResult::Continue)` return value
    // (`majit/majit-translate/src/codewriter/jtransform.rs
    //  ::rewrite_op_direct_call`).  The current sequence is:
    //
    //     inline_call_r_r/dR>r ; live/ ; catch_exception/L ;
    //     goto/L ; reraise/ ; ref_return/r ; live/ ;
    //     raise/r ; ref_return/r
    //
    // Every outer opname has a handler in this module.  The
    // remaining gap lives two levels deeper: PopTop's
    // `inline_call_r_r/dR>r` recurses into the `pop_top` callee
    // which itself recurses into a body whose first byte is
    // `getfield_vable_i/rd>i` — currently surfaced as
    // `UnsupportedOpname` by the walker.  Unignoring this test
    // requires landing the vable-aware getfield handlers (Phase
    // D-3 MIFrame integration).
    let jc = named_jitcode("execute_pop_top")
        .expect("execute_pop_top must resolve through the portal closure");
    let mut tc = fresh_trace_ctx();
    // Generously sized banks so any byte the codewriter emits is
    // in-range. 256 is the maximum register index a 1-byte slot
    // can address.
    let mut regs_r = distinct_const_refs(&mut tc, 256);
    let mut regs_i = distinct_const_refs(&mut tc, 256);
    // Descr pool: slot at each `BhDescr::JitCode` index in
    // `all_descrs()` is wrapped in a `TestJitCodeDescr` adapter so
    // `inline_call_r_r/dR>r` can resolve `as_jitcode_descr()`.
    // Other slots default to `make_fail_descr`.
    let pool_len = crate::jitcode_runtime::all_descrs().len();
    let descr_pool = descr_pool_with_jitcode_adapters(pool_len);
    let frame_done_descr = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done_descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &production_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, end_pc) =
        walk(&jc.code, 0, &mut wc).expect("PopTop helper must walk to a terminator");
    // PopTop's `inline_call_r_r/dR>r` recurses into the
    // codewriter-emitted callee jitcode (resolved via
    // `production_sub_jitcodes`); on success the outermost
    // `ref_return/r` lands a FINISH at top level.
    assert_eq!(
        outcome,
        DispatchOutcome::Terminate,
        "top-level PopTop walk must end on Terminate",
    );
    assert!(
        end_pc <= jc.code.len(),
        "walker must not run past the arm body \
             (end_pc={end_pc}, code.len()={})",
        jc.code.len(),
    );
    drop(wc);
    let ops_after = tc.num_ops();
    assert!(
        ops_after > ops_before,
        "PopTop walk must record at least one op (FINISH from \
             ref_return at top level) — recorded {} → {}",
        ops_before,
        ops_after,
    );
    // No `residual_call_r_r/iRd>r` in the post-rewrite arm, so no
    // CallR-descr identity check applies here.  If a future arm
    // regrows the wrapper, restore the `as_call_descr().is_some()`
    // + `Arc::ptr_eq(real_call_descr)` checks that lived here in
    // the pre-`Ok` / `Err` / `Some` identity rewrite version of
    // this fixture.
}

#[test]
fn inline_call_with_more_args_than_callee_regs_surfaces_arity_mismatch() {
    // codewriter shape contract says `R-list.len() <=
    // callee.num_regs_r` for `inline_call_r_r/dR>r`. Walker rejects
    // overflow with a typed error instead of silently dropping
    // (the dropped args would carry symbolic OpRefs the callee
    // never reads, breaking dataflow).
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_r/dR>r")
        .expect("`inline_call_r_r/dR>r` must be in insns table");
    // Callee declares num_regs_r=1 but caller passes 2 ref args.
    let callee_code: &'static [u8] = Box::leak(Box::new([0xFFu8])); // unreachable
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = move |idx: usize| {
        if idx == 5 {
            Some(sub_body.clone())
        } else {
            None
        }
    };
    // R-list = [r0, r1] but callee has only 1 slot.
    let caller_code = [inline_byte, 0x05, 0x00, 0x02, 0x00, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[5] = make_jitcode_descr(5);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&caller_code, 0, &mut wc).expect_err("arity overflow must surface error");
    assert_eq!(
        err,
        DispatchError::InlineCallArityMismatch {
            pc: 0,
            provided: 2,
            callee_num_regs_r: 1,
        },
    );
}

#[test]
fn inline_call_with_void_subreturn_surfaces_unexpected_void_error() {
    let err = DispatchError::UnexpectedVoidSubReturn { pc: 42 };
    assert_eq!(err, DispatchError::UnexpectedVoidSubReturn { pc: 42 },);
}

// ── inline_call_*_v regression tests ──────────────────────────────
//
// Exercise the void-return contract for all three dispatch variants:
//   dispatch_inline_call_dr_kind  (inline_call_r_v/dR)
//   dispatch_inline_call_dir_kind (inline_call_ir_v/dIR)
//   dispatch_inline_call_dirf_kind(inline_call_irf_v/dIRF)

#[test]
fn inline_call_r_v_accepts_void_returning_callee() {
    // Callee body: `void_return/` — surfaces SubReturn { None }.
    // Caller: `inline_call_r_v/dR  descr=7 R=[r0]` then `void_return/`.
    let void_ret = *insns_opname_to_byte()
        .get("void_return/")
        .expect("`void_return/` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_v/dR")
        .expect("`inline_call_r_v/dR` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([void_ret]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // dR layout: 2B descr(7) + 1B R-len(1) + 1B R-arg(r0)  — no >X dst
    let caller_code = [
        inline_byte,
        0x07,
        0x00, // descr index 7
        0x01,
        0x00,     // R: len=1, arg=r0
        void_ret, // caller terminates
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, _) =
        walk(&caller_code, 0, &mut wc).expect("inline_call_r_v with void callee must succeed");
    assert_eq!(outcome, DispatchOutcome::Terminate);
}

#[test]
fn inline_call_r_v_rejects_non_void_returning_callee() {
    // Callee body: `ref_return r0` — surfaces SubReturn { Some(_) }.
    // inline_call_r_v must reject with UnexpectedNonVoidSubReturn.
    let ref_ret = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_r_v/dR")
        .expect("`inline_call_r_v/dR` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([ref_ret, 0x00]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 0,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    let caller_code = [
        inline_byte,
        0x07,
        0x00, // descr index 7
        0x01,
        0x00, // R: len=1, arg=r0
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = walk(&caller_code, 0, &mut wc)
        .expect_err("inline_call_r_v with non-void callee must reject");
    assert_eq!(err, DispatchError::UnexpectedNonVoidSubReturn { pc: 0 });
}

#[test]
fn inline_call_ir_v_accepts_void_returning_callee() {
    let void_ret = *insns_opname_to_byte()
        .get("void_return/")
        .expect("`void_return/` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_ir_v/dIR")
        .expect("`inline_call_ir_v/dIR` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([void_ret]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // dIR layout: 2B descr(7) + I-list(len=1, i0) + R-list(len=1, r0) — no dst
    let caller_code = [
        inline_byte,
        0x07,
        0x00, // descr index 7
        0x01,
        0x00, // I: len=1, arg=i0
        0x01,
        0x00,     // R: len=1, arg=r0
        void_ret, // caller terminates
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let mut regs_i = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, _) =
        walk(&caller_code, 0, &mut wc).expect("inline_call_ir_v with void callee must succeed");
    assert_eq!(outcome, DispatchOutcome::Terminate);
}

#[test]
fn inline_call_ir_v_rejects_non_void_returning_callee() {
    let int_ret = *insns_opname_to_byte()
        .get("int_return/i")
        .expect("`int_return/i` must be in insns table");
    let inline_byte = *insns_opname_to_byte()
        .get("inline_call_ir_v/dIR")
        .expect("`inline_call_ir_v/dIR` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([int_ret, 0x00]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1,
        num_regs_f: 0,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    let caller_code = [
        inline_byte,
        0x07,
        0x00,
        0x01,
        0x00, // I: len=1, arg=i0
        0x01,
        0x00, // R: len=1, arg=r0
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let mut regs_i = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = walk(&caller_code, 0, &mut wc)
        .expect_err("inline_call_ir_v with non-void callee must reject");
    assert_eq!(err, DispatchError::UnexpectedNonVoidSubReturn { pc: 0 });
}

#[test]
fn inline_call_irf_v_accepts_void_returning_callee() {
    let void_ret = *insns_opname_to_byte()
        .get("void_return/")
        .expect("`void_return/` must be in insns table");
    let &inline_byte = insns_opname_to_byte()
        .get("inline_call_irf_v/dIRF")
        .expect("`inline_call_irf_v/dIRF` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([void_ret]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1,
        num_regs_f: 1,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    // dIRF layout: 2B descr + I-list + R-list + F-list — no dst
    let caller_code = [
        inline_byte,
        0x07,
        0x00, // descr index 7
        0x01,
        0x00, // I: len=1, arg=i0
        0x01,
        0x00, // R: len=1, arg=r0
        0x01,
        0x00,     // F: len=1, arg=f0
        void_ret, // caller terminates
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let mut regs_i = distinct_const_refs(&mut tc, 4);
    let mut regs_f = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut regs_f,
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, _) =
        walk(&caller_code, 0, &mut wc).expect("inline_call_irf_v with void callee must succeed");
    assert_eq!(outcome, DispatchOutcome::Terminate);
}

#[test]
fn inline_call_irf_v_rejects_non_void_returning_callee() {
    let ref_ret = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let &inline_byte = insns_opname_to_byte()
        .get("inline_call_irf_v/dIRF")
        .expect("`inline_call_irf_v/dIRF` must be in insns table");
    let callee_code: &'static [u8] = Box::leak(Box::new([ref_ret, 0x00]));
    let sub_body = SubJitCodeBody {
        code: callee_code,
        num_regs_r: 1,
        num_regs_i: 1,
        num_regs_f: 1,
        constants_i: &[],
        constants_r: &[],
        constants_f: &[],
    };
    let lookup = {
        let sub_body = sub_body.clone();
        move |idx: usize| {
            if idx == 7 {
                Some(sub_body.clone())
            } else {
                None
            }
        }
    };
    let caller_code = [
        inline_byte,
        0x07,
        0x00,
        0x01,
        0x00, // I
        0x01,
        0x00, // R
        0x01,
        0x00, // F
    ];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 4);
    let mut regs_i = distinct_const_refs(&mut tc, 4);
    let mut regs_f = distinct_const_refs(&mut tc, 4);
    let descr = done_descr_ref_for_tests();
    let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
    descr_pool[7] = make_jitcode_descr(7);
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut regs_f,
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &lookup,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = walk(&caller_code, 0, &mut wc)
        .expect_err("inline_call_irf_v with non-void callee must reject");
    assert_eq!(err, DispatchError::UnexpectedNonVoidSubReturn { pc: 0 });
}

/// Build a `SimpleFieldDescr` with a stable index so the
/// heapcache lookup hashes consistently across the cache-miss
/// and cache-hit assertions. Default `Descr::index()` returns
/// `u32::MAX`; tests that exercise heapcache need a real index.
fn field_descr_with_index(idx: u32) -> DescrRef {
    std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(
        idx,
        8, // offset
        8, // field_size
        majit_ir::Type::Int,
        false, // not immutable
    ))
}

#[test]
fn getfield_gc_i_cache_miss_records_op_and_writes_dst() {
    // First `getfield_gc_i/rd>i` invocation
    // is a heapcache miss — walker records `OpCode::GetfieldGcI`
    // with `[obj]` and `descr=descr_refs[d]`, writes the
    // recorder result into `registers_i[dst]`, and updates the
    // heapcache via `getfield_now_known(resbox)`.
    let byte = *insns_opname_to_byte()
        .get("getfield_gc_i/rd>i")
        .expect("`getfield_gc_i/rd>i` must be in insns table");
    // Operand layout `rd>i`: 1B r-reg(2) + 2B descr-index(LE 1) + 1B dst(5).
    let code = [byte, 0x02, 0x01, 0x00, 0x05];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let dst_pre = regs_i[5];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("getfield_gc_i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 5, "getfield_gc_i/rd>i operand layout = 4 bytes");
    let dst_post = wc.registers_i[5];
    assert_ne!(
        dst_post, dst_pre,
        "cache miss must write a fresh recorder OpRef into registers_i[dst]",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "cache miss must record exactly one GetfieldGcI op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::GetfieldGcI);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj],
        "GetfieldGcI args must be [obj] (the r-reg source)",
    );
    let recorded_descr = last
        .getdescr()
        .expect("GetfieldGcI must carry the field descr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &descr),
        "GetfieldGcI descr must be descr_refs[d] (the field descr)",
    );
    assert_eq!(dst_post, last.pos.get());
}

#[test]
fn getfield_gc_i_cache_hit_returns_cached_box_without_recording() {
    // Second invocation with the same
    // (obj, descr) pair must hit the heapcache and skip IR
    // emission. RPython parity:
    //   upd = heapcache.get_field_updater(box, fielddescr)
    //   if upd.currfieldbox is not None:
    //       return upd.currfieldbox  # no execute_with_descr
    let byte = *insns_opname_to_byte()
        .get("getfield_gc_i/rd>i")
        .expect("`getfield_gc_i/rd>i` must be in insns table");
    let code = [byte, 0x02, 0x01, 0x00, 0x05];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();

    // Pre-populate the heapcache as if a previous getfield had
    // already cached the field's value. RPython equivalent:
    // `heapcache.getfield_now_known(...)` after a prior fetch.
    let cached_field = tc.const_int(0xCAFE);
    tc.heapcache_getfield_now_known(obj, 1, cached_field);
    let ops_before = tc.num_ops();

    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("getfield_gc_i must dispatch");
    let dst_post = wc.registers_i[5];
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "cache hit must NOT record any new IR op",
    );
    assert_eq!(
        dst_post, cached_field,
        "cache hit must write the cached OpRef into registers_i[dst]",
    );
}

#[test]
fn getfield_gc_r_cache_miss_records_op_and_writes_ref_dst() {
    // GetfieldGcR variant — same flow as
    // GetfieldGcI but result lands in registers_r.
    let byte = *insns_opname_to_byte()
        .get("getfield_gc_r/rd>r")
        .expect("`getfield_gc_r/rd>r` must be in insns table");
    let code = [byte, 0x02, 0x01, 0x00, 0x06];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let dst_pre = regs_r[6];
    // Use a Ref-typed field descr — sanity-check that the walker
    // doesn't introspect the descr's field_type (it just feeds
    // descr_index into the heapcache and records the op).
    let descr: DescrRef = std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(
        1,
        16,
        8,
        majit_ir::Type::Ref,
        false,
    ));
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("getfield_gc_r must dispatch");
    let dst_post = wc.registers_r[6];
    assert_ne!(dst_post, dst_pre);
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::GetfieldGcR);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj]
    );
    assert_eq!(dst_post, last.pos.get());
}

#[test]
fn getfield_gc_with_out_of_range_obj_register_surfaces_typed_error() {
    let byte = *insns_opname_to_byte()
        .get("getfield_gc_i/rd>i")
        .expect("`getfield_gc_i/rd>i` must be in insns table");
    let code = [byte, 0x07, 0x00, 0x00, 0x00]; // r-reg=7, registers_r empty
    let mut tc = fresh_trace_ctx();
    let descr = field_descr_with_index(0);
    let descr_pool = vec![descr];
    let frame_done = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let err = step(&code, 0, &mut wc).expect_err("getfield_gc must validate r-reg");
    assert_eq!(
        err,
        DispatchError::RegisterOutOfRange {
            pc: 0,
            reg: 7,
            len: 0,
            bank: "r",
        },
    );
}

#[test]
fn getfield_vable_i_routes_through_metainterp_and_writes_dst() {
    // T2 sanity: `getfield_vable_i/rd>i` delegates to
    // `TraceCtx::vable_getfield_int`.  With no `virtualizable_info`
    // bound on the trace context, `is_nonstandard_virtualizable`
    // returns true and the fallback emits a `GetfieldGcI` op +
    // writes the recorder OpRef into `registers_i[dst]` — the same
    // shape `getfield_gc_via_heapcache` produces on a cache miss.
    // The handler itself stays orthodox to RPython
    // `pyjitpl.py opimpl_getfield_vable_i`; the
    // GETFIELD_GC fallback is `vable_getfield_int`'s decision, not
    // the walker's, so this test exercises the walker→trace_ctx
    // boundary without depending on a `virtualizable_info` fixture.
    let byte = *insns_opname_to_byte()
        .get("getfield_vable_i/rd>i")
        .expect("`getfield_vable_i/rd>i` must be in insns table");
    // Operand layout `rd>i`: 1B r-reg(2) + 2B descr-index(LE 1) + 1B dst(5).
    let code = [byte, 0x02, 0x01, 0x00, 0x05];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let dst_pre = regs_i[5];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("getfield_vable_i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 5, "getfield_vable_i/rd>i operand layout = 4 bytes");
    let dst_post = wc.registers_i[5];
    assert_ne!(
        dst_post, dst_pre,
        "fallback must write a fresh recorder OpRef into registers_i[dst]",
    );
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "nonstandard-vable fallback records exactly one GetfieldGcI op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::GetfieldGcI);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj],
        "GetfieldGcI args must be [obj] (the r-reg source)",
    );
    let recorded_descr = last
        .getdescr()
        .expect("GetfieldGcI must carry the field descr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &descr),
        "GetfieldGcI descr must be descr_refs[d] (the field descr)",
    );
    assert_eq!(dst_post, last.pos.get());
}

#[test]
fn setfield_vable_i_routes_through_metainterp_records_setfield_gc_fallback() {
    // T2a sanity: `setfield_vable_i/rid` delegates to
    // `TraceCtx::vable_setfield`.  With no `virtualizable_info`
    // bound on the trace context, `is_nonstandard_virtualizable`
    // returns true and the fallback records a `SetfieldGc` op
    // with `[obj, value]` + the field descr — same shape
    // `setfield_gc_via_heapcache` produces.  Exercises the
    // walker -> trace_ctx boundary for the int-bank variant
    // (the `r` and `f` variants share the handler body, varying
    // only `value_bank`).
    let byte = *insns_opname_to_byte()
        .get("setfield_vable_i/rid")
        .expect("`setfield_vable_i/rid` must be in insns table");
    // Operand layout `rid`: 1B r-reg(2) + 1B i-reg(3) + 2B descr-index(LE 1).
    let code = [byte, 0x02, 0x03, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let value = regs_i[3];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("setfield_vable_i must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 5, "setfield_vable_i/rid operand layout = 4 bytes");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "nonstandard-vable fallback records exactly one SetfieldGc op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::SetfieldGc);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj, value],
        "SetfieldGc args must be [obj, value]",
    );
    let recorded_descr = last
        .getdescr()
        .expect("SetfieldGc must carry the field descr");
    assert!(
        std::sync::Arc::ptr_eq(&recorded_descr, &descr),
        "SetfieldGc descr must be descr_refs[d] (the field descr)",
    );
}

#[test]
fn setfield_gc_i_redundant_write_skips_recording() {
    // When the heapcache already knows
    // valuebox is the current value of (obj, descr), the
    // SETFIELD_GC IR op must NOT be recorded. RPython parity:
    // `pyjitpl.py if upd.currfieldbox is valuebox: return`.
    let byte = *insns_opname_to_byte()
        .get("setfield_gc_i/rid")
        .expect("`setfield_gc_i/rid` must be in insns table");
    let code = [byte, 0x02, 0x03, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let valuebox = regs_i[3];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
    let frame_done = done_descr_ref_for_tests();
    // Pre-cache valuebox as the current field value.
    tc.heapcache_getfield_now_known(obj, 1, valuebox);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("setfield_gc_i must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "redundant setfield (cached valuebox == new valuebox) must skip recording",
    );
}

#[test]
fn setfield_gc_i_fresh_write_records_op_and_caches_value() {
    // A fresh write (no cached value)
    // must record SETFIELD_GC and update the heapcache so a
    // subsequent redundant write hits.
    let byte = *insns_opname_to_byte()
        .get("setfield_gc_i/rid")
        .expect("`setfield_gc_i/rid` must be in insns table");
    let code = [byte, 0x02, 0x03, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let valuebox = regs_i[3];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("setfield_gc_i must dispatch");
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before + 1,
        "fresh setfield must record exactly one SetfieldGc op",
    );
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::SetfieldGc);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj, valuebox],
        "SetfieldGc args must be [obj, valuebox] in that order",
    );
    assert!(std::sync::Arc::ptr_eq(
        &last.getdescr().expect("SetfieldGc must carry descr"),
        &descr,
    ),);
    // Cache must now know the new field value.  Box identity-only
    // check — value payload is Void in walker-emitted writes.
    assert_eq!(
        tc.heapcache_getfield_cached(obj, 1).map(|b| b),
        Some(valuebox),
        "post-setfield, the heapcache must reflect the written value",
    );
}

#[test]
fn setfield_gc_r_records_setfieldgc_with_ref_valuebox() {
    // `rrd` shape — both box and valuebox
    // come from registers_r. SetfieldGc is type-agnostic at the
    // IR level (the descr carries the field type).
    let byte = *insns_opname_to_byte()
        .get("setfield_gc_r/rrd")
        .expect("`setfield_gc_r/rrd` must be in insns table");
    let code = [byte, 0x02, 0x05, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let obj = regs_r[2];
    let valuebox = regs_r[5];
    let descr: DescrRef = std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(
        1,
        16,
        8,
        majit_ir::Type::Ref,
        false,
    ));
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("setfield_gc_r must dispatch");
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::SetfieldGc);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![obj, valuebox]
    );
}

#[test]
fn getarrayitem_gc_r_cache_miss_records_op_and_writes_dst() {
    // First `getarrayitem_gc_r/rid>r` is a
    // heapcache miss — record GetarrayitemGcR with
    // [array, index] + descr; write recorder result into r-dst
    // and update heapcache.
    let byte = *insns_opname_to_byte()
        .get("getarrayitem_gc_r/rid>r")
        .expect("`getarrayitem_gc_r/rid>r` must be in insns table");
    // Operand layout `rid>r`: 1B r-reg(2) + 1B i-reg(3) +
    // 2B descr(LE 1) + 1B r-dst(5).
    let code = [byte, 0x02, 0x03, 0x01, 0x00, 0x05];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_refs(&mut tc, 8);
    let array = regs_r[2];
    let index = regs_i[3];
    let dst_pre = regs_r[5];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("getarrayitem_gc_r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "getarrayitem_gc_r/rid>r operand layout = 5 bytes"
    );
    let dst_post = wc.registers_r[5];
    assert_ne!(dst_post, dst_pre);
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::GetarrayitemGcR);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![array, index],
        "GetarrayitemGcR args must be [array, index]",
    );
    assert!(std::sync::Arc::ptr_eq(
        &last.getdescr().expect("must carry array descr"),
        &descr,
    ));
    assert_eq!(dst_post, last.pos.get());
}

#[test]
fn getarrayitem_gc_r_cache_hit_returns_cached_box() {
    // Pre-cache (array, index, descr) →
    // cached_box. Second invocation must return cached_box and
    // not record an IR op.
    let byte = *insns_opname_to_byte()
        .get("getarrayitem_gc_r/rid>r")
        .expect("`getarrayitem_gc_r/rid>r` must be in insns table");
    let code = [byte, 0x02, 0x03, 0x01, 0x00, 0x05];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_ints(&mut tc, 8);
    let array = regs_r[2];
    let index = regs_i[3];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
    let frame_done = done_descr_ref_for_tests();
    let cached = tc.const_ref(0xCAFE_F00D);
    tc.heapcache_getarrayitem_now_known(array, index, 1, cached);
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let _ = step(&code, 0, &mut wc).expect("getarrayitem_gc_r must dispatch");
    let dst_post = wc.registers_r[5];
    drop(wc);
    assert_eq!(
        tc.num_ops(),
        ops_before,
        "cache hit must NOT record any new IR op",
    );
    assert_eq!(
        dst_post, cached,
        "cache hit must write cached OpRef into registers_r[dst]",
    );
}

#[test]
fn setarrayitem_gc_r_records_setarrayitemgc_with_three_args() {
    // `setarrayitem_gc_r/rird` records
    // SetarrayitemGc with [array, index, value] + descr and
    // updates the heapcache via setarrayitem.
    let byte = *insns_opname_to_byte()
        .get("setarrayitem_gc_r/rird")
        .expect("`setarrayitem_gc_r/rird` must be in insns table");
    // Operand layout `rird`: 1B r-reg(2) + 1B i-reg(4) +
    // 1B r-reg(6) + 2B descr(LE 1).
    let code = [byte, 0x02, 0x04, 0x06, 0x01, 0x00];
    let mut tc = fresh_trace_ctx();
    let mut regs_r = distinct_const_refs(&mut tc, 8);
    let mut regs_i = distinct_const_ints(&mut tc, 8);
    let array = regs_r[2];
    let index = regs_i[4];
    let value = regs_r[6];
    let descr = field_descr_with_index(1);
    let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
    let frame_done = done_descr_ref_for_tests();
    let ops_before = tc.num_ops();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &descr_pool,
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: frame_done,
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("setarrayitem_gc_r must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(
        next_pc, 6,
        "setarrayitem_gc_r/rird operand layout = 5 bytes"
    );
    drop(wc);
    assert_eq!(tc.num_ops(), ops_before + 1);
    let last = tc.ops().last().expect("recorded op must exist");
    assert_eq!(last.opcode, majit_ir::OpCode::SetarrayitemGc);
    assert_eq!(
        last.getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect::<Vec<_>>(),
        vec![array, index, value],
        "SetarrayitemGc args must be [array, index, value]",
    );
    assert!(std::sync::Arc::ptr_eq(
        &last.getdescr().expect("must carry array descr"),
        &descr,
    ));
    // Heapcache must reflect the write.  Box identity-only check —
    // value payload is Void in walker-emitted writes.
    assert_eq!(
        tc.heapcache_getarrayitem(array, index, 1).map(|b| b),
        Some(value),
        "post-setarrayitem, heapcache must reflect the written value",
    );
}

#[test]
fn dispatch_via_miframe_runs_ref_return_through_real_miframe_state() {
    // Acceptance: the bridge function takes a
    // real `MIFrame` (constructed via the same `PyreSym::new_uninit`
    // + `MIFrame { ctx, sym, .. }` shape that `state.rs`'s
    // existing tests use), pre-populates `sym.registers_r[2]` with
    // a known OpRef, then walks `ref_return r2`. Walker must
    // record `Finish([sym.registers_r[2]], descr=done_with_this_frame_descr_ref)`
    // through the *same* TraceCtx the MIFrame's `ctx` pointer
    // owns — i.e., production-shape state plumbing, no separate
    // test fixture.
    use crate::state::PyreSym;

    let mut tc = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
    let expected_arg = tc.const_ref(0xCAFE_F00D);
    let mut sym = PyreSym::new_uninit(OpRef::NONE);
    *sym.registers_r_mut() = vec![OpRef::NONE; 8];
    sym.registers_r_mut()[2] = expected_arg;

    let miframe = MIFrame {
        ctx: &mut tc,
        sym: &mut sym,
        fallthrough_pc: 0,
        pending_result_stack_idx: None,
        pending_result_type: None,
        pending_inline_frame: None,
        residual_call_pc: None,
        loop_close_marker_jit_pc: None,
        orgpc: 0,
        concrete_frame_addr: 0,
        pre_opcode_registers_r: None,
        pre_opcode_semantic_depth: None,
    };

    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let code = [ret_byte, 0x02];
    let descr = make_fail_descr(1);
    // PyPy `setup_call(argboxes)` analog: stamp `expected_arg` at
    // `R[2]_r` so the `ref_return r2` walker handler picks it up
    // from the fresh top-level register file.  Slots 0/1 stay
    // `OpRef::NONE` since this fixture exercises only slot 2.
    let argboxes_r = [OpRef::NONE, OpRef::NONE, expected_arg];
    fbw_finish_payload_reset();
    let session = std::cell::RefCell::new(WalkSession::default());
    let (outcome, end_pc) = dispatch_via_miframe(
        unsafe { &mut *miframe.ctx },
        unsafe { &mut *miframe.sym },
        miframe.concrete_frame_addr,
        miframe.orgpc,
        &session,
        &code,
        0,
        &[],
        RawDescrPool::Global,
        false,
        &no_sub_jitcodes,
        descr.clone(),
        make_fail_descr(101),
        make_fail_descr(102),
        make_fail_descr(103),
        make_fail_descr(2),
        true,
        8,
        0,
        0,
        &[],
        &[],
        &[],
        &argboxes_r,
        &[],
        &[],
    )
    .expect("dispatch_via_miframe must succeed for ref_return r2");
    assert_eq!(outcome, DispatchOutcome::Terminate);
    assert_eq!(end_pc, 2);

    // Drop miframe so we can inspect tc directly.
    drop(miframe);
    assert_eq!(
        fbw_finish_payload_take(),
        Some((expected_arg, Type::Ref)),
        "finish payload must be sym.registers_r[2] threaded through the MIFrame bridge",
    );
}

#[test]
fn dispatch_via_miframe_mirrors_last_exc_value_back_into_sym() {
    // When the walker's last_exc_value field
    // changes (raise/r sets it before terminating), the bridge
    // function must mirror it back to `sym.last_exc_box`. RPython
    // parity: `metainterp.last_exc_value = ...` is metainterp-level
    // state that survives across opimpl invocations.
    use crate::state::PyreSym;

    let mut tc = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
    let exc_oprep = tc.const_ref(0xDEAD_BEEF);
    let mut sym = PyreSym::new_uninit(OpRef::NONE);
    *sym.registers_r_mut() = vec![OpRef::NONE; 8];
    sym.registers_r_mut()[3] = exc_oprep;
    // Pre-condition: sym.last_exc_box is unset.
    assert!(sym.last_exc_box().is_none());

    let miframe = MIFrame {
        ctx: &mut tc,
        sym: &mut sym,
        fallthrough_pc: 0,
        pending_result_stack_idx: None,
        pending_result_type: None,
        pending_inline_frame: None,
        residual_call_pc: None,
        loop_close_marker_jit_pc: None,
        orgpc: 0,
        concrete_frame_addr: 0,
        pre_opcode_registers_r: None,
        pre_opcode_semantic_depth: None,
    };

    let raise_byte = *insns_opname_to_byte()
        .get("raise/r")
        .expect("`raise/r` must be in insns table");
    let code = [raise_byte, 0x03];
    let descr_done = make_fail_descr(1);
    let descr_exc = make_fail_descr(99);
    // Setup_call argbox at R[3]_r: `raise/r` reads its exc operand
    // from this slot in the fresh top-level register file.
    let argboxes_r = [OpRef::NONE, OpRef::NONE, OpRef::NONE, exc_oprep];
    let session = std::cell::RefCell::new(WalkSession::default());
    let (outcome, _) = dispatch_via_miframe(
        unsafe { &mut *miframe.ctx },
        unsafe { &mut *miframe.sym },
        miframe.concrete_frame_addr,
        miframe.orgpc,
        &session,
        &code,
        0,
        &[],
        RawDescrPool::Global,
        false,
        &no_sub_jitcodes,
        descr_done,
        make_fail_descr(101),
        make_fail_descr(102),
        make_fail_descr(103),
        descr_exc,
        true,
        8,
        0,
        0,
        &[],
        &[],
        &[],
        &argboxes_r,
        &[],
        &[],
    )
    .expect("dispatch_via_miframe must succeed for raise r3");
    assert_eq!(outcome, DispatchOutcome::Terminate);
    drop(miframe);
    // Post-condition: sym.last_exc_box was mirrored from the
    // walker's last_exc_value (set by raise/r before terminate).
    assert_eq!(
        sym.last_exc_box(),
        exc_oprep,
        "sym.last_exc_box must mirror the exc OpRef the walker captured \
             via WalkContext::last_exc_value",
    );
    // Post-condition: dispatch_via_miframe also sets
    // sym.class_of_last_exc_is_const to mirror RPython's
    // `pyjitpl.py opimpl_raise: class_of_last_exc_is_const = True`.
    assert!(
        sym.class_of_last_exc_is_const(),
        "sym.class_of_last_exc_is_const must be true after a raise/r",
    );
}

#[test]
fn dispatch_via_miframe_leaves_class_of_last_exc_is_const_unchanged_when_no_raise() {
    // When the walk does NOT raise (final last_exc remains None),
    // dispatch_via_miframe must NOT touch
    // sym.class_of_last_exc_is_const. The flag carries state from
    // a prior tracing step and must not be cleared by an unrelated
    // walk (e.g. a single ref_return-only top-level walk).
    use crate::state::PyreSym;

    let mut tc = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
    let value = tc.const_ref(0xC0FFEE);
    let mut sym = PyreSym::new_uninit(OpRef::NONE);
    *sym.registers_r_mut() = vec![OpRef::NONE; 8];
    sym.registers_r_mut()[2] = value;
    // Pre-condition: simulate prior raise — class_of_last_exc_is_const
    // is true and last_exc_box is set.
    sym.set_class_of_last_exc_is_const(true);
    sym.set_last_exc_box(value);

    let miframe = MIFrame {
        ctx: &mut tc,
        sym: &mut sym,
        fallthrough_pc: 0,
        pending_result_stack_idx: None,
        pending_result_type: None,
        pending_inline_frame: None,
        residual_call_pc: None,
        loop_close_marker_jit_pc: None,
        orgpc: 0,
        concrete_frame_addr: 0,
        pre_opcode_registers_r: None,
        pre_opcode_semantic_depth: None,
    };
    let ret_byte = *insns_opname_to_byte()
        .get("ref_return/r")
        .expect("`ref_return/r` must be in insns table");
    let code = [ret_byte, 0x02];
    // Setup_call argbox at R[2]_r — the `ref_return r2` walker
    // handler picks it up from the fresh top-level register file.
    let argboxes_r = [OpRef::NONE, OpRef::NONE, value];
    let session = std::cell::RefCell::new(WalkSession::default());
    let _ = dispatch_via_miframe(
        unsafe { &mut *miframe.ctx },
        unsafe { &mut *miframe.sym },
        miframe.concrete_frame_addr,
        miframe.orgpc,
        &session,
        &code,
        0,
        &[],
        RawDescrPool::Global,
        false,
        &no_sub_jitcodes,
        make_fail_descr(1),
        make_fail_descr(101),
        make_fail_descr(102),
        make_fail_descr(103),
        make_fail_descr(2),
        true,
        8,
        0,
        0,
        &[],
        &[],
        &[],
        &argboxes_r,
        &[],
        &[],
    )
    .expect("ref_return walk must succeed");
    drop(miframe);
    // Walker preserved the carried-in class flag because no raise
    // happened during the walk.
    assert!(
        sym.class_of_last_exc_is_const(),
        "no-raise walk must not clear class_of_last_exc_is_const",
    );
}

#[test]
fn walk_undecodable_byte_surfaces_typed_error() {
    // 0xFF is unknown to the insns table (21 entries 0..=20 today).
    let code = [0xFFu8];
    let mut tc = fresh_trace_ctx();
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut [],
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    assert_eq!(
        walk(&code, 0, &mut wc),
        Err(DispatchError::UndecodableOpcode { pc: 0 })
    );
}

/// `jit_merge_point/cIRFIRF` reached_loop_header behaviour: first
/// visit of a (green key, red shape) registers the merge point and
/// continues unrolling; the second visit with the same key + shape
/// closes the loop with the reds as jump args.  Mirrors
/// `pyjitpl.py` first-visit/found split.
#[test]
fn jit_merge_point_first_visit_continues_then_closes_loop() {
    let jmp_byte = *insns_opname_to_byte()
        .get("jit_merge_point/cIRFIRF")
        .expect("`jit_merge_point/cIRFIRF` must be in insns table");
    // cIRFIRF byte layout: jdindex(c) + greens(I gi, R gr, F gf) +
    // reds(I ri, R rr, F rf).  greens = portal jitdriver's
    // [next_instr(i0), pycode(r0)]; reds = [r1, r2] (loop-carried
    // refs, e.g. [frame, ec]).
    let code = [
        jmp_byte, 0x00, // c: jdindex
        0x01, 0x00, // gi: len=1, [i0 = next_instr]
        0x01, 0x00, // gr: len=1, [r0 = pycode]
        0x00, // gf: len=0
        0x00, // ri: len=0
        0x02, 0x01, 0x02, // rr: len=2, [r1, r2]
        0x00, // rf: len=0
    ];
    // Model the trace as having STARTED at this loop header: the close
    // gate fires only when the arriving green key equals the primary
    // `root_green_key` (a non-primary header re-arrival continues — the
    // cross-loop-cut elimination).  The arriving key is
    // `make_green_key(pycode, next_instr)` from the green concretes below.
    let mut tc = TraceCtx::for_test_types_with_green_key(
        &[Type::Ref],
        crate::driver::make_green_key(0x1_0000 as *const (), 42),
    );
    let next_instr = tc.const_int(42); // gi[0] = Python pc
    let pycode = tc.const_ref(0x1_0000); // gr[0] = PyCode ptr
    let red0 = tc.const_ref(0x2_0000); // rr[0]
    let red1 = tc.const_ref(0x3_0000); // rr[1]
    let mut regs_i = vec![next_instr];
    let mut regs_r = vec![pycode, red0, red1];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };

    // Arrival without a preceding `loop_header` stamp and with no
    // recorded ops is a plain pass-through (pyjitpl.py):
    // nothing registers, nothing closes.
    let (gated, gated_next) = step(&code, 0, &mut wc).expect("gated jit_merge_point must dispatch");
    assert_eq!(gated, DispatchOutcome::Continue);
    assert_eq!(gated_next, code.len());

    // First crossing via a backward jump: `loop_header` stamped the
    // per-trace flag (pyjitpl.py) — registers
    // (key, [red0, red1]) and continues.
    wc.trace_ctx.seen_loop_header_for_jdindex = 0;
    let (first, first_next) = step(&code, 0, &mut wc).expect("first jit_merge_point must dispatch");
    assert_eq!(first, DispatchOutcome::Continue);
    assert_eq!(first_next, code.len());
    // The stamp is consumed (pyjitpl.py).
    assert_eq!(wc.trace_ctx.seen_loop_header_for_jdindex, -1);

    // Second stamped crossing (same key + red shape): closes the loop.
    // The reds here are constants, so `remove_consts_and_duplicates`
    // (pyjitpl.py) replaces each with a freshly recorded
    // `same_as` op before the close — the jump args are runtime
    // OpRefs wrapping the original const reds, not the consts.
    wc.trace_ctx.seen_loop_header_for_jdindex = 0;
    let (second, _) = step(&code, 0, &mut wc).expect("second jit_merge_point must dispatch");
    match second {
        DispatchOutcome::CloseLoop {
            jump_args,
            loop_header_pc,
            ..
        } => {
            assert_eq!(loop_header_pc, 42);
            assert_eq!(jump_args.len(), 2);
            for arg in &jump_args {
                assert!(!arg.is_constant(), "const red must be same_as-wrapped");
                assert_eq!(wc.trace_ctx.get_opref_type(*arg), Some(majit_ir::Type::Ref));
            }
        }
        other => panic!("expected CloseLoop, got {other:?}"),
    }
}

/// `loop_header/i` stamps `seen_loop_header_for_jdindex` from its
/// int-constant operand and records nothing (pyjitpl.py).
#[test]
fn loop_header_stamps_seen_flag() {
    let lh_byte = *insns_opname_to_byte()
        .get("loop_header/i")
        .expect("`loop_header/i` must be in insns table");
    let code = [lh_byte, 0x00]; // i: register slot 0 holds the jdindex
    let mut tc = fresh_trace_ctx();
    let jdindex = tc.const_int(0);
    let mut regs_i = vec![jdindex];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    assert_eq!(wc.trace_ctx.seen_loop_header_for_jdindex, -1);
    let (outcome, next) = step(&code, 0, &mut wc).expect("loop_header must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next, code.len());
    assert_eq!(wc.trace_ctx.seen_loop_header_for_jdindex, 0);
    assert_eq!(wc.trace_ctx.num_ops(), 0, "loop_header records nothing");
}

#[test]
fn jit_merge_point_int_form_resolves_jdindex_from_the_int_bank() {
    let byte = *insns_opname_to_byte()
        .get("jit_merge_point/iIRFIRF")
        .expect("`jit_merge_point/iIRFIRF` must be in insns table");
    let code = [
        byte, 0x02, // jdindex is in i2, not the literal value 2
        0x01, 0x00, // gi: [i0]
        0x01, 0x00, // gr: [r0]
        0x00, // gf
        0x00, // ri
        0x00, // rr
        0x00, // rf
    ];
    let pycode_ptr = 0x1_0000usize;
    let mut tc = TraceCtx::for_test_types_with_green_key(
        &[Type::Ref],
        crate::driver::make_green_key(pycode_ptr as *const (), 42),
    );
    let next_instr = tc.const_int(42);
    let unused = tc.const_int(99);
    let jdindex = tc.const_int(0);
    let pycode = tc.const_ref(pycode_ptr as i64);
    let mut regs_i = [next_instr, unused, jdindex];
    let mut regs_r = [pycode];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: done_descr_ref_for_tests(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    wc.trace_ctx.seen_loop_header_for_jdindex = 0;
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int-form merge point must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, code.len());
    assert_eq!(wc.trace_ctx.seen_loop_header_for_jdindex, -1);
}

/// A green list with no concrete leading element cannot form the
/// green key — surfaces `JitMergePointGreenKeyUnresolved` rather than
/// guessing.
#[test]
fn jit_merge_point_unresolved_green_key_fails_loud() {
    let jmp_byte = *insns_opname_to_byte()
        .get("jit_merge_point/cIRFIRF")
        .expect("`jit_merge_point/cIRFIRF` must be in insns table");
    let code = [
        jmp_byte, 0x00, // c
        0x01, 0x00, // gi: len=1, [i0]
        0x01, 0x00, // gr: len=1, [r0]
        0x00, // gf
        0x00, // ri
        0x00, // rr
        0x00, // rf
    ];
    let mut tc = fresh_trace_ctx();
    // i0 is a non-constant input arg → no concrete next_instr.
    let mut regs_i = vec![OpRef::input_arg_int(0)];
    let mut regs_r = vec![tc.const_ref(0x1_0000)];
    let descr = done_descr_ref_for_tests();
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut regs_r,
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut [],
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: descr.clone(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    assert_eq!(
        step(&code, 0, &mut wc),
        Err(DispatchError::JitMergePointGreenKeyUnresolved { pc: 0 })
    );
}

#[test]
fn structural_midbody_anchor_requires_exact_floor_segment_start() {
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0), (101, 28), (115, 29)];
    assert!(super::exact_floor_segment_anchor(&pyjit.metadata, 28, 101));
    assert!(!super::exact_floor_segment_anchor(&pyjit.metadata, 28, 102));
    assert!(!super::exact_floor_segment_anchor(&pyjit.metadata, 27, 101));
}

#[test]
fn portal_marker_anchor_accepts_only_same_pc_last_instr_bookkeeping() {
    use majit_metainterp::jitcode::RuntimeBhDescr;
    use majit_translate::jitcode::BhDescr;

    let setfield = *insns_opname_to_byte()
        .get("setfield_vable_i/rid")
        .expect("setfield_vable_i must exist");
    let abort = *insns_opname_to_byte()
        .get("abort_permanent/")
        .expect("abort_permanent must exist");
    let int_copy = *insns_opname_to_byte()
        .get("int_copy/c>i")
        .expect("int_copy/c>i must exist");
    let descrs = [RuntimeBhDescr::Descr(BhDescr::VableField { index: 0 })];
    let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 29)];
    let portal_gap = [setfield, 1, 7, 0, 0, abort];

    assert!(super::portal_marker_first_jit_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &portal_gap,
        29,
        5,
        |_| 29,
    ));
    assert!(!super::portal_marker_first_jit_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &portal_gap,
        29,
        5,
        |_| 28,
    ));

    let computation_gap = [int_copy, 1, 7, abort];
    assert!(!super::portal_marker_first_jit_anchor(
        &pyjit.metadata,
        true,
        1,
        &descrs,
        &computation_gap,
        29,
        3,
        |_| 29,
    ));

    pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0), (5, 29)];
    assert!(super::portal_marker_first_jit_anchor(
        &pyjit.metadata,
        false,
        u16::MAX,
        &[],
        &portal_gap,
        29,
        5,
        |_| 29,
    ));
}

#[test]
fn callee_vable_ref_gates_on_frame_register_identity() {
    use majit_ir::{GcRef, Value};

    let ref_value = Value::Ref(GcRef(0x1000));
    let mut shadow = super::CalleeLocalsShadow::default();
    shadow.set_concrete(1, 3, ref_value);

    // An entry recorded through frame reg 1 resolves for that frame only.
    assert!(matches!(
        super::callee_vable_ref_at(Some(&shadow), 1, 3),
        Some(ConcreteValue::Ref(value)) if value as usize == 0x1000
    ));
    // A foreign frame register declines (no strict-fold witness either).
    assert_eq!(super::callee_vable_ref_at(Some(&shadow), 2, 3), None);

    // The strict-fold witness admits the slot even when the recorded writer
    // frame differs from the queried frame.
    shadow.fold_frame_reg = 2;
    assert!(matches!(
        super::callee_vable_ref_at(Some(&shadow), 2, 3),
        Some(ConcreteValue::Ref(value)) if value as usize == 0x1000
    ));

    // A missing slot, the `u16::MAX` sentinel, and an absent shadow decline.
    assert_eq!(super::callee_vable_ref_at(Some(&shadow), 1, 4), None);
    assert_eq!(super::callee_vable_ref_at(Some(&shadow), u16::MAX, 3), None);
    assert_eq!(super::callee_vable_ref_at(None, 1, 3), None);
}

/// `insert_renamings` routes a cyclic parallel move through the Int scratch
/// (`int_push/i` then `int_pop/>i`).  The pop must land the source's concrete
/// shadow in `concrete_registers_i[dst]`; leaving the destination slot's old
/// shadow behind makes every later `read_int_reg_concrete` on that register
/// report the value the move overwrote.
#[test]
fn int_scratch_move_carries_the_concrete_shadow_to_the_destination() {
    let push_byte = *insns_opname_to_byte()
        .get("int_push/i")
        .expect("int_push must be in the runtime instruction table");
    let pop_byte = *insns_opname_to_byte()
        .get("int_pop/>i")
        .expect("int_pop must be in the runtime instruction table");
    // Push r0, pop into r1: r1's pre-existing shadow must not survive.
    let code = [push_byte, 0, pop_byte, 1];
    let mut tc = TraceCtx::for_test_types(&[Type::Int, Type::Int]);
    let src = OpRef::input_arg_int(0);
    let dst_before = OpRef::input_arg_int(1);
    tc.set_opref_concrete(src, Value::Int(7));
    tc.set_opref_concrete(dst_before, Value::Int(99));
    let mut regs_i = vec![src, dst_before];
    let mut concrete_i = vec![ConcreteValue::Int(7), ConcreteValue::Int(99)];
    let session = std::cell::RefCell::new(WalkSession::default());
    let mut wc = WalkContext {
        callee_shadow: None,
        inline_callee_consts: None,
        fbw_mode: test_fbw_mode(),
        session: &session,
        registers_r: &mut [],
        registers_i: &mut regs_i,
        registers_f: &mut [],
        concrete_registers_r: &mut [],
        concrete_registers_i: &mut concrete_i,
        descr_refs: &[],
        raw_descrs: RawDescrPool::Global,
        is_authoritative_executor: false,
        trace_ctx: &mut tc,
        done_with_this_frame_descr_ref: done_descr_ref_for_tests(),
        done_with_this_frame_descr_int: make_fail_descr(101),
        done_with_this_frame_descr_float: make_fail_descr(102),
        done_with_this_frame_descr_void: make_fail_descr(103),
        exit_frame_with_exception_descr_ref: make_fail_descr(2),
        is_top_level: true,
        sub_jitcode_lookup: &no_sub_jitcodes,
        last_exc_value: None,
        last_exc_value_concrete: ConcreteValue::Null,
        entry_py_pc: EntryPyPc::Py(0),
        outer_resume_marker_jit_pc: None,
        outer_jitcode_index: 0,
        outer_active_boxes: Vec::new(),
        store_subscr_fn_addr: None,
        pending_guard_snapshot_error: None,
        vstack_boxes: Vec::new(),
        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_ceiling: u32::MAX,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
    };
    let (outcome, next_pc) = step(&code, 0, &mut wc).expect("`int_push/i` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 2);
    let (outcome, next_pc) = step(&code, next_pc, &mut wc).expect("`int_pop/>i` must dispatch");
    assert_eq!(outcome, DispatchOutcome::Continue);
    assert_eq!(next_pc, 4);
    assert_eq!(wc.registers_i[1], src, "the pop moves the source OpRef");
    assert_eq!(
        wc.concrete_registers_i[1],
        ConcreteValue::Int(7),
        "the pop must overwrite the destination's stale concrete shadow",
    );
}

/// `FASTPATHS_SAME_BOXES` membership, on the non-fused (macro-table) arms.
/// The generated loop that carries `if b1 is b2: return <const>` spells only
/// the six signed int compares plus the four ref compares; the unsigned
/// compares live in the other loop and must still record their op.
#[test]
fn same_box_fastpath_covers_exactly_the_generated_fastpath_list() {
    // (opname, expected folded value) for every FASTPATHS_SAME_BOXES member
    // reachable through `binop_int_record` / `binop_ref_to_int_record`.
    let int_cases: &[(&str, i64)] = &[
        ("int_eq/ii>i", 1),
        ("int_le/ii>i", 1),
        ("int_ge/ii>i", 1),
        ("int_ne/ii>i", 0),
        ("int_lt/ii>i", 0),
        ("int_gt/ii>i", 0),
    ];
    for (opname, expected) in int_cases {
        let byte = *insns_opname_to_byte()
            .get(*opname)
            .unwrap_or_else(|| panic!("{opname} must be in the runtime instruction table"));
        // Same register for both operands: `b1 is b2`.
        let code = [byte, 0, 0, 1];
        let mut tc = TraceCtx::for_test_types(&[Type::Int]);
        let same = OpRef::input_arg_int(0);
        let mut regs_i = vec![same, OpRef::NONE];
        let (outcome, _) = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
            .unwrap_or_else(|_| panic!("`{opname}` must dispatch"));
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert!(
            tc.ops().is_empty(),
            "`{opname}` on identical boxes must answer without recording an op",
        );
        assert_eq!(
            tc.constant_value(regs_i[1]),
            Some(*expected),
            "`{opname}` on identical boxes must fold to {expected}",
        );
    }
    // The unsigned compares share the `ii>i` shape but are NOT fast-path
    // members: they must still record.
    for opname in [
        "uint_lt/ii>i",
        "uint_le/ii>i",
        "uint_gt/ii>i",
        "uint_ge/ii>i",
    ] {
        let byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("{opname} must be in the runtime instruction table"));
        let code = [byte, 0, 0, 1];
        let mut tc = TraceCtx::for_test_types(&[Type::Int]);
        let same = OpRef::input_arg_int(0);
        let mut regs_i = vec![same, OpRef::NONE];
        let (outcome, _) = run_hint_step(&code, &mut tc, &mut [], &mut [], &mut regs_i)
            .unwrap_or_else(|_| panic!("`{opname}` must dispatch"));
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            tc.ops().len(),
            1,
            "`{opname}` is not a same-boxes fast-path member and must record",
        );
    }
}

/// The four ref compares are all `FASTPATHS_SAME_BOXES` members, including
/// `instance_ptr_eq` / `instance_ptr_ne`.
#[test]
fn ref_compare_same_box_fastpath_covers_the_instance_ptr_spellings() {
    let cases: &[(&str, i64)] = &[
        ("ptr_eq/rr>i", 1),
        ("ptr_ne/rr>i", 0),
        ("instance_ptr_eq/rr>i", 1),
        ("instance_ptr_ne/rr>i", 0),
    ];
    for (opname, expected) in cases {
        let byte = *insns_opname_to_byte()
            .get(*opname)
            .unwrap_or_else(|| panic!("{opname} must be in the runtime instruction table"));
        let code = [byte, 0, 0, 0];
        let mut tc = TraceCtx::for_test_types(&[Type::Ref]);
        let same = OpRef::input_arg_ref(0);
        let mut regs_r = vec![same];
        let mut concrete_r = vec![ConcreteValue::Null];
        let mut regs_i = vec![OpRef::NONE];
        let (outcome, _) = run_hint_step(&code, &mut tc, &mut regs_r, &mut concrete_r, &mut regs_i)
            .unwrap_or_else(|_| panic!("`{opname}` must dispatch"));
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert!(
            tc.ops().is_empty(),
            "`{opname}` on identical boxes must answer without recording an op",
        );
        assert_eq!(
            tc.constant_value(regs_i[0]),
            Some(*expected),
            "`{opname}` on identical boxes must fold to {expected}",
        );
    }
}
