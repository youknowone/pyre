//! Trace-side jitcode walker.
//!
//! RPython parity: this is the trace-side counterpart of
//! `BlackholeInterpBuilder.dispatch_loop` (`blackhole.py:65-100`). The
//! blackhole loop *executes* each `bhimpl_*` in turn; the tracing-side
//! analogue lives in `pyjitpl.py:opimpl_*` where each opcode becomes
//! a `MetaInterp.execute_and_record` call (RPython
//! `pyjitpl.py:1640-1660`). This module is the sole production tracer:
//! it consumes the codewriter-emitted jitcode bytes directly, executing
//! as it records (`is_authoritative_executor`). The trait-driven
//! `MIFrame::execute_opcode_step` interpret loop is retired.
//!
//! Module layout vs. RPython: because this FBW walker has no single
//! `rpython/jit/metainterp/` file counterpart (the file-for-file parity
//! mirror is the `majit-metainterp` crate — `pyjitpl.rs`, `executor.rs`,
//! `heapcache.rs`, `resume.rs`, `virtualizable.rs`, `blackhole.rs`, ...),
//! the submodules here are a pyre-local navigability split. Each submodule
//! header names the PyPy concept it is the trace-side counterpart of, or
//! declares itself pyre-specific where there is no upstream analogue.
//! Behavioural parity is enforced per opcode arm, not per file.
//!
//! Opcode coverage:
//!
//! | opname              | parity status | behaviour |
//! |---------------------|---------------|-----------|
//! | `live/`             | PARITY        | skip OFFSET_SIZE, continue (RPython tracing does not record `live/` either) |
//! | `goto/L`            | PARITY        | jump to 2-byte LE target, continue |
//! | `catch_exception/L` | PARITY        | skip 2-byte target on normal flow (`pyjitpl.py:497-504` records nothing); the target is consumed by `inline_call`'s SubRaise arm via `try_catch_exception_at` (`pyjitpl.py:2517-2522`) |
//! | `switch/id`         | STRUCTURAL ADAPTATION | RPython `opimpl_switch` shape: read int box, lookup `SwitchDictDescr.dict`, emit `GUARD_VALUE` on hit or `INT_EQ` + `GUARD_FALSE` chain on miss. Concrete branch value comes from `TraceCtx::concrete_of_opref`; non-concrete symbolic OpRefs surface `SwitchValueNotConcrete` instead of guessing a branch. |
//! | `ref_return/r`      | PARITY        | top-level: record `Finish(reg) descr=done_with_this_frame_descr_ref` + terminate (`pyjitpl.py:opimpl_ref_return → compile_done_with_this_frame`); sub-walk: surface `SubReturn{Some(value)}` to caller (`pyjitpl.py:1688-1698 finishframe`) |
//! | `int_return/i`      | PARITY        | int-bank counterpart of `ref_return/r` — top-level records `Finish(reg) descr=done_with_this_frame_descr_int` (`pyjitpl.py:3206-3208`), sub-walk surfaces `SubReturn{Some(value)}`. RPython `pyjitpl.py:463 opimpl_int_return = _opimpl_any_return`. |
//! | `float_return/f`    | PARITY        | float-bank counterpart — top-level records `Finish(reg) descr=done_with_this_frame_descr_float` (`pyjitpl.py:3212-3214`), sub-walk surfaces `SubReturn{Some(value)}`. RPython `pyjitpl.py:465 opimpl_float_return = _opimpl_any_return`. |
//! | `void_return/`      | PARITY        | void return — top-level records `Finish([]) descr=done_with_this_frame_descr_void` (`pyjitpl.py:3202-3205`, `exits = []` branch), sub-walk surfaces `SubReturn{None}`. RPython `pyjitpl.py:467-469 opimpl_void_return → finishframe(None)`. |
//! | `inline_call_r_r/dR>r` | PARITY (per-frame catch) | recurses into sub-jitcode via `JitCodeDescr::jitcode_index()`, populates callee `registers_r` (`setup_call_r`, OOR surfaces `InlineCallArityMismatch`), writes `SubReturn{value}` into caller dst (Ref bank), scans caller's `op.next_pc` for `live/` + `catch_exception/L` on `SubRaise` (`pyjitpl.py:2506-2522 finishframe_exception`). Sub-walk reaching `Terminate` is unexpected (top-level should never fire from a sub-walk); `SubReturn{None}` into a `_r_*` slot surfaces `UnexpectedVoidSubReturn`. |
//! | `inline_call_r_i/dR>i` | PARITY        | int-result sibling of `inline_call_r_r/dR>r`. Same recursion + arglist + raise routing; only the dst bank changes (`registers_i[dst] = subreturn_value`). RPython `pyjitpl.py:1266-1324 _opimpl_inline_call*` is generated through `_opimpl_any_inline_call` decorator that varies on the result type — pyre's walker shares the body via `dispatch_inline_call_dr_kind(dst_bank)`. |
//! | `inline_call_ir_r/dIR>r`, `inline_call_ir_i/dIR>i` | PARITY | extended-arglist siblings — descr + I-list + R-list + dst. RPython `setup_call(argboxes_i, argboxes_r, argboxes_f)` (pyjitpl.py:230-260) populates the callee's int + ref banks from the two lists. Walker uses `dispatch_inline_call_dir_kind(dst_bank)` which reads `read_int_var_list` then `read_ref_var_list` and surfaces per-bank arity overflow as `InlineCallIntArityMismatch` / `InlineCallArityMismatch`. |
//! | `inline_call_irf_r/dIRF>r`, `inline_call_irf_f/dIRF>f` | PARITY | full-arglist variants — descr + I-list + R-list + F-list + dst. RPython same `setup_call` distribution; walker uses `dispatch_inline_call_dirf_kind(dst_bank)` extending the dIR helper with `read_float_var_list` + float-bank arg setup. Float arity overflow surfaces `InlineCallFloatArityMismatch`. |
//! | `int_copy/i>i`      | PARITY        | `registers_i[dst] = registers_i[src]` SSA rename, no IR op emitted (`pyjitpl.py:471-477 _opimpl_any_copy + >i` decorator) |
//! | `ref_copy/r>r`      | PARITY        | Ref-bank sibling — `registers_r[dst] = registers_r[src]` SSA rename, no IR op. Const-source variants (codewriter `emit_ref_copy!` with `ConstRef`) resolve through the constants window of `registers_r` (pre-populated by `setposition` in [`num_regs_r, num_regs_and_consts_r)`). |
//! | `int_<binop>/ii>i`  | PARITY        | int_add/int_sub/int_mul/int_and/int_or/int_xor/int_lshift/int_rshift + comparisons int_eq/int_ne/int_lt/int_le/int_gt/int_ge (14 ops). Reads two `i`-coded regs, records `OpCode::Int<Binop>` with `[a, b]`, writes recorder result into dst (`pyjitpl.py:279-336`). Mixed shapes such as `int_lshift/ri>i` stay unwired: those are kind-flow kind-flow bugs and must stay unsupported. |
//! | `float_<binop>/ff>f` + `float_neg/f>f` | PARITY | float_add/float_sub/float_truediv binops + float_neg unary (4 ops total — float_mul, float comparisons, float_abs all absent from codewriter today, would land mechanically when emitted). Read on `registers_f` bank, record `OpCode::Float<Binop>`, write dst (`pyjitpl.py:284-292`). |
//! | `int_neg/i>i`, `int_invert/i>i` | PARITY | unary i→i ops via `unop_int_record`. RPython `pyjitpl.py:356-368` exec-generated unary opimpls. `int_same_as/i>i` has a dormant walker arm for forward-prep, but the generated table should not contain it because RPython `jtransform.py:246 rewrite_op_same_as` removes `same_as` before assembly. |
//! | `cast_int_to_float/i>f` | PARITY | i-bank read, record `CastIntToFloat`, f-bank write. RPython `pyjitpl.py:357 cast_int_to_float` (same exec-generated unary opimpl loop). |
//! | `ptr_eq/rr>i`, `ptr_ne/rr>i` | PARITY | r-bank pair → record PtrEq/PtrNe → i-bank dst via `binop_ref_to_int_record`. RPython `pyjitpl.py:326-336` exec-generated comparison opimpls (b1 is b2 fast path omitted, same rationale as int comparisons). |
//! | `getfield_gc_i/rd>i`, `getfield_gc_r/rd>r` | PARITY (heapcache-aware) | r-bank obj + descr → heapcache lookup. Cache hit returns cached OpRef without recording; cache miss records `OpCode::GetfieldGc<I,R>` + `getfield_now_known` writeback. RPython `pyjitpl.py:855-882 + 929-950 _opimpl_getfield_gc_any_pureornot`. ConstPtr fast-path (`pyjitpl.py:856-860`) deferred — pyre walker doesn't track ConstPtr identity (optimizer's job post-trace). The pyre-specific `id>X` shape (int source — kind-flow kind-flow) stays unsupported. |
//! | `setfield_gc_i/rid`, `setfield_gc_r/rrd` | PARITY (heapcache-aware, alias-clearing) | r-bank box + (i\|r)-bank valuebox + descr. If `getfield_cached(obj,descr) == Some(valuebox)` skip recording (RPython `if upd.currfieldbox is valuebox: return`); otherwise record `OpCode::SetfieldGc(obj, valuebox)` + `setfield_cached` write-through. Aliasing semantics: `CacheEntry.do_write_with_aliasing` (heapcache.py:90-94) routes through `_clear_cache_on_write(seen_alloc)` — always wipes `cache_anything`, additionally wipes `cache_seen_allocation` when the write target itself isn't seen-allocated. RPython `pyjitpl.py:973-988 _opimpl_setfield_gc_any`. The disabled is_unescaped branch (`pyjitpl.py:981-988`) is intentionally not ported — RPython itself has it commented out. `iid` / `ird` (int box) shapes stay unsupported (kind-flow territory). |
//! | `getarrayitem_gc_r/rid>r` | PARITY (heapcache-aware) | r-bank array + i-bank index + descr → heapcache `getarrayitem` lookup. Cache hit returns cached OpRef without IR; cache miss records `OpCode::GetarrayitemGcR(array, index)` + `getarrayitem_now_known` writeback. RPython `pyjitpl.py:639-688 _do_getarrayitem_gc_any`. All three `_i` / `_r` / `_f` result shapes are wired to this same heapcache body (kind-keyed dst bank, dispatch arms below) and registered in `wellknown_bh_insns()` (`insns.rs:865-866`) for blackhole execution + codewriter emission. |
//! | `setarrayitem_gc_r/rird`, `setarrayitem_gc_r/rcrd` | PARITY (heapcache-aware) | r-bank array + i-bank index + r-bank value + descr. Always records `OpCode::SetarrayitemGc(array, index, value)` + `heapcache.setarrayitem(...)` write. RPython `pyjitpl.py:736-744 _opimpl_setarrayitem_gc_any` — no skip-on-redundant short-circuit because `setarrayitem` does aliasing-aware invalidation. The `rcrd` `c`-argcode form (USE_C_FORM `assembler.py:99-107/312`) decodes the index as one inline signed byte → ConstInt; same recording body otherwise. `rrid` / `rrrd` / `rrfd` (Ref index) shapes stay unsupported (kind-flow). |
//! | `residual_call_r_r/iRd>r` | TODO (`direct_assembler_call` + `capture_resumedata` not yet wired) | classifies the call by `EffectInfo`. Wired sub-cases: (1) release-gil via [`direct_call_release_gil`] — `CallReleaseGilI` + arglist `[savebox, funcbox] + argboxes[1:]` reshape per `pyjitpl.py:3675-3681`, plus the outer forces-branch `GUARD_NOT_FORCED` (`:2079`) + `GUARD_NO_EXCEPTION` (`:2082`); (2) loop-invariant heapcache via [`loopinvariant_lookup`] / [`loopinvariant_now_known`] per `pyjitpl.py:2088 + 2109`; (3) vable IR bookkeeping (`pyjitpl.py:2055-2080`) via [`maybe_walker_vable_and_vrefs_before_residual_call`] — emits FORCE_TOKEN + SETFIELD_GC only; the runtime heap halves of the token protocol (`vinfo.tracing_before_residual_call` / `vrefinfo.tracing_before_residual_call` and the after-call `vinfo.tracing_after_residual_call`, `pyjitpl.py`) are bracketed around the concrete callee execution by [`try_execute_residual_call_via_executor`], which arms TOKEN_TRACING_RESCALL before the call and probe-and-clears it after, surfacing [`DispatchError::VableEscapedDuringResidualCall`] on a detected force (`pyjitpl.py` ABORT_ESCAPE parity). The vref halves of the bracket are ported on `TraceCtx` but uncalled by the walker — see the module preamble. The remaining branches go through [`select_residual_call_opcode`]: `CallMayForce*` + `GuardNotForced` on the rest of the forces-virtual path (`pyjitpl.py:2017-2082`), `CallLoopinvariant*` on `EF_LOOPINVARIANT` (`pyjitpl.py:2087-2110`), `CallPure*` on elidable, otherwise `Call*`. `GuardNoException` follows whenever `effectinfo.check_can_raise(False)` is true (`pyjitpl.py:2082 handle_possible_exception`). `heapcache.invalidate_caches_varargs(call_opcode, ei, allboxes)` (`pyjitpl.py:2042 + 2659`) is wired around every recorded call op. `OS_NOT_IN_TRACE` is fail-loud-guarded up front via [`do_not_in_trace_call_result`] — `effect_info_for_call_flavor` stub never sets the index today (`flatten.rs:431`), making it dead until producers land. Same fail-loud treatment via [`do_jit_force_virtual_guard`] for `OS_JIT_FORCE_VIRTUAL` (stricter-than-PyPy — needs OpRef→concrete-pointer resolver). Still deferred (each blocked on infrastructure absent from pyre-jit-trace): `direct_libffi_call` / `direct_assembler_call` specialization (`pyjitpl.py:1908-1990` — assembler_call paths route through `inline_call_*/dR>X` instead), KEEPALIVE for vablebox (only fires when `direct_assembler_call` returns a vablebox), and `num_live`-aware `capture_resumedata(after_residual_call=True)` on the guards (`pyjitpl.py:2078-2082 → 2586`). |
//! | `residual_call_r_i/iRd>i` | PARITY (kind sibling of `_r_r`) | same EffectInfo classification + guard emission as `_r_r` — `select_residual_call_opcode('i', ...)` returns the int-typed `Call*` family (`CallReleaseGilI` / `CallMayForceI` / `CallLoopinvariantI` / `CallPureI` / `CallI`); only the dst writeback bank (`registers_i`) differs. RPython parity: `pyjitpl.py:1346 opimpl_residual_call_r_i = _opimpl_residual_call1`; `do_residual_call`'s `descr.get_normalized_result_type()` dispatch (pyjitpl.py:2022-2044) selects the int-result CALL op. Argboxes pass through [`build_allboxes`] same as `_r_r` (R-list-only argboxes → identity permutation when arg_types is ref-only). |
//! | `residual_call_ir_r/iIRd>r` | PARITY (shape sibling of `_r_r`) | adds an i-bank list between funcptr and the R-list. RPython parity: `pyjitpl.py:1349 opimpl_residual_call_ir_r = _opimpl_residual_call2`; `boxes2` argcode (`pyjitpl.py:3750-3760`) decodes the two count-prefixed lists into `argboxes = [i_args..., r_args...]`. Walker passes that flat list through [`build_allboxes`] (line-by-line port of `pyjitpl.py:1960-1993 _build_allboxes`) which permutes argboxes by `descr.get_arg_types()` so the recorded `Call*` arglist matches the callee's actual ABI even for mixed orderings like `[REF, INT, REF, INT]`. Same EffectInfo classification + guard emission as `_r_r` via [`select_residual_call_opcode`]. |
//! | `raise/r`           | PARITY (`GUARD_CLASS`) | sets `ctx.last_exc_value` (`pyjitpl.py:1695`); top-level records `Finish(exc) descr=exit_frame_with_exception_descr_ref` (`pyjitpl.py:3238-3242 compile_exit_frame_with_exception`); sub-walk surfaces `SubRaise{exc}`. Caller-side handler scan (`finishframe_exception`) lives on `inline_call`'s SubRaise arm (above). RPython `pyjitpl.py:1690-1693` also emits `GUARD_CLASS(exc, cls_of_box(exc))` when `heapcache.is_class_known(exc) == false`; the retired trait-side path read `concrete_exc.ob_header.ob_type` from the concrete frame snapshot and emitted the orthodox `GuardClass(exc_box, cls_const)` per the heapcache `is_class_known` gate. |
//! | `reraise/`          | PARITY        | reads `ctx.last_exc_value` (asserts via `ReraiseWithoutLastExcValue` matching `pyjitpl.py:1702 assert`); same dual top-level/sub-walk routing as `raise/r` (`pyjitpl.py:1700-1704 popframe + finishframe_exception`). |
//! | `last_exc_value/>r` | PARITY        | reads `ctx.last_exc_value`, writes the OpRef into `registers_r[dst]` — pure SSA rename, no IR op recorded. RPython `pyjitpl.py:1716-1719 opimpl_last_exc_value` returns `self.metainterp.last_exc_box` after asserting `last_exc_value` is non-null; missing slot surfaces `LastExcValueWithoutActiveException` (codewriter invariant: only emits inside `catch_exception/L` body). |
//!
//! Covers: decode walker, `WalkContext { registers_r, trace_ctx }` +
//! `ref_return/r` recording, `goto/L`, `catch_exception/L`,
//! `reraise/`, `int_copy/i>i`, `residual_call_r_r/iRd>r`,
//! `inline_call_r_r/dR>r` recursion, caller-frame `catch_exception`
//! scan, `last_exc_value` field, `reraise` finishframe routing, typed
//! arity / shape / no-active-exception errors, production
//! `PyreJitCodeDescr` adapter.
//!
//! This module replaced the trait dispatch in
//! `MIFrame::execute_opcode_step` (now retired). The entry point is
//! [`walk`], driven from `trace_bytecode` (`trace.rs`) with the
//! appropriate context.
//!
//! Production fidelity gaps (ranked by priority for follow-on work):
//!
//! 1. `residual_call_r_r/iRd>r` `do_residual_call` port
//!    (`pyjitpl.py:1995-2127`). Walker selects the IR opcode via
//!    [`select_residual_call_opcode`] (`CallReleaseGil*` /
//!    `CallMayForce*` / `CallLoopinvariant*` / `CallPure*` / `Call*`),
//!    unconditionally emits `GuardNotForced` on the forces and
//!    release-gil branches, and emits `GuardNoException` whenever
//!    `effectinfo.check_can_raise(False)` is true. Items now wired (was
//!    deferred in earlier audits):
//!    - `vable_and_vrefs_before_residual_call` IR portion (FORCE_TOKEN +
//!      SETFIELD_GC `vable_token_descr`) via
//!      [`walker_vable_and_vrefs_before_residual_call`].
//!    - `direct_call_release_gil` (`pyjitpl.py:3675-3681`) via
//!      [`direct_call_release_gil`].
//!    - `loopinvariant_lookup` / `loopinvariant_now_known`
//!      (`pyjitpl.py:2088 + 2109`).
//!    - `heapcache.invalidate_caches_varargs(call_opcode, ei, allboxes)`
//!      (`pyjitpl.py:2042 + 2072`) wired around every recorded call op.
//!    - `OS_NOT_IN_TRACE` fail-loud guard via
//!      [`do_not_in_trace_call_result`] (`pyjitpl.py:2003-2005`) —
//!      `effect_info_for_call_flavor` stub never sets the index today
//!      (`flatten.rs::effect_info_for_call_flavor` audit table), making
//!      it dead until the codewriter analyzer trio (annotator/rtyper/
//!      translator) lands.
//!    Items still deferred (each on infrastructure outside walker
//!    scope):
//!    a. **Now walker-bracketed (formerly trait-leg-only)**:
//!       `vable_after_residual_call` (`pyjitpl.py`) observes a runtime
//!       force via a heap-token read. The authoritative walk executes
//!       the callee concretely in
//!       [`try_execute_residual_call_via_executor`], which arms the
//!       token before the call and probe-and-clears it after, surfacing
//!       [`DispatchError::VableEscapedDuringResidualCall`] on a force
//!       (`pyjitpl.py` ABORT_ESCAPE parity).  The vref halves
//!       (`vrefs_before_residual_call` / `vrefs_after_residual_call` →
//!       `stop_tracking_virtualref`) are bracketed at the same two
//!       points, over the vrefs an inlined call's `enter` pushed
//!       (`inline_call.rs::walker_ec_enter`).  They still iterate zero
//!       times for a callee the walker inlines without seeding a frame:
//!       such a level has no frame object to take a `jit.virtual_ref`
//!       of, where upstream's `perform_call` builds one for every
//!       inlined call (`pyjitpl.py:2445-2476, 1862-1874`).
//!    b. **Codewriter-side**: `direct_assembler_call` + KEEPALIVE on
//!       vablebox (`pyjitpl.py:3589-3609 + 2080-2081`). Walker's
//!       residual_call dispatchers never receive `assembler_call=True`
//!       — the parallel `inline_call_*/dR>X` opcode family
//!       ([`dispatch_inline_call_dr_kind`]) routes that case.
//!    c. **Cross-leg, not yet implemented**: `_do_jit_force_virtual`
//!       PTR_EQ + GUARD_VALUE prelude (`pyjitpl.py:2011-2014 → 2153-2172`).
//!       Walker fail-louds via [`do_jit_force_virtual_guard`]
//!       (stricter-than-PyPy: typed error rather than divergent IR);
//!       full body needs an OpRef → concrete-pointer resolver (Task
//!       #45) before it can return `Some(vref_opref)` /
//!       `Some(standard_opref)` / `None`. Production reach today: 0 —
//!       `OopSpecIndex::JitForceVirtual` is set only by
//!       `jtransform.rs:1903 jit.force_virtual` lowering, which our
//!       benchmarks don't reach. Metainterp orthodox port at
//!       `majit-metainterp/src/pyjitpl.rs:11828` is tests-only.
//!    d. **Not yet implemented**: `direct_libffi_call`
//!       (`pyjitpl.py:3622-3667`) needs `CIF_DESCRIPTION_P` parser +
//!       dynamic calldescr builder; live tracer also returns None
//!       universally (`pyjitpl.rs:11487-11491`). Production reach
//!       0 — pyre interpreter doesn't expose libffi calls.
//!    e. Guard recording uses `ctx.trace_ctx.record_guard(..., 0)`
//!       followed by `walker_capture_snapshot_for_last_guard`
//!       (`jitcode_dispatch.rs:walker_capture_snapshot_for_last_guard`)
//!       — the walker-side port of `capture_resumedata(
//!       after_residual_call=True)` (`pyjitpl.py:2599-2603`).  Each
//!       residual_call guard (`GuardNotForced`, `GuardNoException`)
//!       carries a single-frame snapshot keyed by
//!       `ctx.outer_jitcode_index` so the optimizer's
//!       `store_final_boxes_in_guard` finds populated resume data.
//!       Active-box narrowing via per-PC liveness is a future
//!       follow-up; today's helper conservatively snapshots every
//!       non-`OpRef::NONE` register across all three banks.
//!    (`_build_allboxes` ABI re-ordering is wired — see
//!    [`build_allboxes`].)
//! 2. `raise/r`'s `GUARD_CLASS` (`pyjitpl.py:1690-1693 opimpl_raise`)
//!    is emitted by reading `concrete_exc.ob_header.ob_type`, checking
//!    `heap_cache.is_class_known(exc_box)`, and emits
//!    `GuardClass(exc_box, cls_const)` when needed — the orthodox
//!    `pyjitpl.py:1690-1696` flow.
//! 3. End-to-end portal-closure helper tests (`walk_return_value_helper_*`,
//!    `walk_pop_top_helper_*`) stay `#[ignore]` until handlers exist for
//!    every opname the codewriter-emitted callee bodies use (e.g.
//!    `getfield_vable_i/rd>i`).
//! 4. (External) `build_default_bh_builder_with_unwired_report` is a
//!    transitional helper for kind-flow (6 unwired opnames:
//!    `int_ge/ir>i`, `int_mul/ir>i`, `int_ne/fr>i`, `int_xor/ri>i`,
//!    `setarrayitem_gc_f/rrfd`, `setarrayitem_gc_i/rrid` — kind-flow
//!    bug in assembler emitting mixed-kind operand types). RPython
//!    upstream has no non-strict builder. Removed when kind-flow
//!    closes; not blocking dispatcher work.
//! 5. Concrete-truth-dependent branch opnames (`goto_if_not/iL`,
//!    `goto_if_exception_mismatch/iL`) and the non-constant side of
//!    `switch/id`. RPython
//!    `pyjitpl.py:511-526 opimpl_goto_if_not`: `switchcase = box.getint()`
//!    branches on the runtime concrete value — `if switchcase: opnum =
//!    GUARD_TRUE; promoted_box = CONST_1` else `opnum = GUARD_FALSE`,
//!    then `metainterp.generate_guard(opnum, box, resumepc=orgpc)`.
//!    `WalkContext` carries `OpRef`s rather than concrete
//!    `box.getint()` values, so these opnames are resolved against the
//!    per-step concrete frame snapshot: the walker reads the runtime
//!    truth and records the corresponding branch guard directly.
//!    Same handling for `goto_if_exception_mismatch/iL`
//!    (`pyjitpl.py:484-496` — `last_exc_value`/llexitcase comparison).
//! 6. Class-introspection opname `last_exception/>i`. RPython
//!    `pyjitpl.py:1707-1713 opimpl_last_exception`: returns
//!    `ConstInt(ptr2int(rclass.ll_cast_to_object(exc_value).typeptr))` —
//!    the class pointer of the standing exception. Resolving the class
//!    needs `concrete_exc.ob_header.ob_type`, read from the concrete frame
//!    snapshot, the same source used for items 2 and 5.

use crate::jitcode_runtime::{DecodedOp, decode_op_at};
use crate::state::{ConcreteValue, MIFrame, WalkSym};
use majit_ir::{DescrRef, OopSpecIndex, OpCode, OpRef, Type, Value};
use majit_metainterp::{TraceCtx, default_effect_info};

// jitcode_dispatch submodules (extracted from this file). Their `pub`
// items are re-exported so `crate::jitcode_dispatch::` paths stay stable.
// __SUBMODULES__
mod specialize;
pub use specialize::*;
mod inline_call;
pub use inline_call::*;
mod residual_call;
pub use residual_call::*;
mod fbw_state;
pub use fbw_state::*;
mod resume_snapshot;
pub use resume_snapshot::*;
mod branch;
pub use branch::*;
mod vstack_mirror;
pub use vstack_mirror::*;
mod vable_ops;
pub use vable_ops::*;
mod heapcache_ops;
pub use heapcache_ops::*;
mod bridge_subwalk;
pub use bridge_subwalk::*;
mod arith;
pub use arith::*;
mod diag;
pub use diag::*;

/// Body of a callee jitcode that the walker needs to recurse into.
/// RPython parity: when `inline_call_r_r/dR>r` fires, the metainterp
/// reads the descr's `JitCode` body (`pyjitpl.py
/// _opimpl_inline_call*`). Walker consumes the same minimal subset:
/// the bytecode bytes + register-bank sizes for the fresh callee
/// frame.
///
/// Body is always `'static` — production wires the lookup to
/// `crate::jitcode_runtime::all_jitcodes()` whose `Arc<JitCode>`
/// entries live inside a `LazyLock<Vec<...>>` (`'static`); tests
/// either use static byte arrays or `Box::leak` to surface
/// `'static`. Constraining the body's lifetime simplifies
/// `WalkContext`'s lifetime parameters — otherwise the closure's
/// covariance would force register-bank borrows to extend to the
/// lookup's lifetime.
#[derive(Debug, Clone)]
pub struct SubJitCodeBody {
    /// Callee's jitcode bytes (RPython `JitCode.code`).
    pub code: &'static [u8],
    /// Number of Ref-bank registers the callee declares
    /// (`JitCode.num_regs_r`). The walker allocates a fresh
    /// `Vec<OpRef>` of this size for the recursive frame.
    pub num_regs_r: usize,
    /// Number of Int-bank registers (`JitCode.num_regs_i`).
    pub num_regs_i: usize,
    /// Number of Float-bank registers (`JitCode.num_regs_f`).
    pub num_regs_f: usize,
    /// Callee's Int-bank constant pool (`JitCode.constants_i`).
    /// The callee bytecode references constant slots via register
    /// indices `[num_regs_i, num_regs_i + constants_i.len())`;
    /// `setposition` (RPython `pyjitpl.py copy_constants`)
    /// pre-populates those slots with `ConstClass(constants_i[i])`.
    pub constants_i: &'static [i64],
    /// Callee's Ref-bank constant pool (`JitCode.constants_r`). Each
    /// `i64` is the erased `PyObjectRef` of a const object resolved
    /// at codewriter time.
    pub constants_r: &'static [i64],
    /// Callee's Float-bank constant pool (`JitCode.constants_f`).
    pub constants_f: &'static [i64],
}

/// Caller-provided sub-jitcode lookup. RPython equivalent: descr
/// resolution within the metainterp loop reads `BhDescr::JitCode {
/// jitcode_index, .. }` and looks up `ALL_JITCODES[idx]`. Walker
/// inverts the dependency: the caller supplies the lookup so the
/// walker stays decoupled from the runtime's all-jitcodes table
/// (production passes a closure over `crate::jitcode_runtime::all_jitcodes()`,
/// tests pass synthetic closures over a local fixture map).
pub type SubJitCodeLookup = dyn Fn(usize) -> Option<SubJitCodeBody>;

/// Build a [`SubJitCodeBody`] view over the build-time `ALL_JITCODES[idx]`
/// entry (`crate::jitcode_runtime::all_jitcodes`). Returns `None` for an
/// out-of-range index.
///
/// The all-jitcodes table is `Box::leak`'d at load
/// (`jitcode_runtime::load_all_jitcodes`), so the borrowed `code` /
/// `constants_*` slices are `'static` as [`SubJitCodeBody`] requires.
///
/// This is the production sub-jitcode lookup shape — the shadow walker,
/// the per-opcode arm entry, and trace-time list-helper specializations
/// that descend into a charon body (e.g. `w_list_append`) all resolve a
/// callee body through it. RPython parity: a `BhDescr::JitCode {
/// jitcode_index }` operand resolves to `ALL_JITCODES[jitcode_index]`.
pub fn sub_jitcode_body_by_index(idx: usize) -> Option<SubJitCodeBody> {
    crate::jitcode_runtime::all_jitcodes()
        .get(idx)
        .map(|jc| SubJitCodeBody {
            code: jc.code.as_slice(),
            num_regs_r: jc.num_regs_r(),
            num_regs_i: jc.num_regs_i(),
            num_regs_f: jc.num_regs_f(),
            constants_i: jc.constants_i.as_slice(),
            constants_r: jc.constants_r.as_slice(),
            constants_f: jc.constants_f.as_slice(),
        })
}

/// State the walker reads from / writes to while stepping. RPython
/// equivalent: `MetaInterp` itself — the trace recorder, the symbolic
/// register banks (`registers_i`, `registers_r`, `registers_f`), and
/// the metainterp static data are all reachable from `self` in
/// `pyjitpl.py:opimpl_*`. Pyre passes them via this struct so the
/// walker can be tested without standing up a full `MIFrame`.
///
/// Field roster:
///
/// * `registers_r`: Ref bank for `r`-coded operands.
/// * `registers_i`: Int bank for `i`-coded operands.
///   `registers_f` (Float bank) lands when float opnames join the
///   handler table.
/// * `descr_refs`: descr pool for `d`-coded operands.
///   Mirrors RPython `Assembler.descrs` (`assembler.py`); each
///   2-byte LE descr index in the jitcode bytes resolves through this
///   table.
/// * `trace_ctx`: live trace recorder.
///
/// Register banks are *mutable* — `int_copy/i>i` and
/// `residual_call_r_r/iRd>r` write their dst slot inline (RPython parity:
/// `pyjitpl.py _opimpl_any_copy` returns the box, the
/// `@arguments("box")` + `>X` decorator pair writes it into the result
/// slot; `pyjitpl.py _opimpl_residual_call*` returns the
/// recorder OpRef which the `>X` slot consumes). `inline_call_r_r/dR>r`
/// also *would* write dst (after sub-jitcode recursion) but stays
/// deferred — see the per-handler comments + module-level "Production
/// fidelity gaps" below.
///
/// Raw `BhDescr` pool selector for the `(VableArray, Array)` recognition
/// in vable-array ops (`vable_array_descrs_from_jitcode`).
///
/// RPython `MIFrame.vable_array_index_pair_at` (`blackhole.rs`) reads
/// `self.descrs[idx]` and asserts `isinstance(BhDescr_VableArray)` to
/// recover the array `index`.  Pyre's single per-walk descr table is
/// either the shared global pool (build-time canonical jitcodes
/// resolve through `ALL_DESCRS`) or the
/// per-`CodeObject` body JitCode's own `exec.descrs`
/// (`jitcode/mod.rs`) — runtime per-frame jitcodes have no global
/// allocation index, so they carry their own pool.  This selector keeps
/// the recognition logic byte-identical and only switches the pool
/// source; it is NOT a swap of the adapted [`WalkContext::descr_refs`]
/// (those still resolve `d`-coded operands index-parallel to whichever
/// raw pool is selected).
#[derive(Clone, Copy)]
pub enum RawDescrPool<'a> {
    /// Shared global `ALL_DESCRS` (`jitcode_runtime::all_descrs`).
    /// Build-time canonical jitcodes (e.g. `w_list_append`, inlined by
    /// the full-body walker's specialization sub-walks) and tests use
    /// this — their `d`/`j` operands index the global pool.
    Global,
    /// Per-`CodeObject` body pool (`JitCode.exec.descrs`).  Full-body
    /// walks (the walker-as-tracer path) resolve `d`/`j` operands through
    /// the body JitCode's own pool.
    PerFn(&'a [majit_metainterp::jitcode::RuntimeBhDescr]),
}

impl<'a> RawDescrPool<'a> {
    /// Resolve the raw `BhDescr` at `idx`, mirroring RPython
    /// `self.descrs[idx]`.  `None` for an out-of-range index or a
    /// per-fn slot whose `RuntimeBhDescr` is not an ordinary `Descr`
    /// (a `JitCode` / `Call` / `AssemblerToken` slot — never read as a
    /// vable-array descr operand).
    fn bh_descr_at(self, idx: usize) -> Option<&'a majit_translate::jitcode::BhDescr> {
        match self {
            Self::Global => crate::jitcode_runtime::all_descrs().get(idx),
            Self::PerFn(descrs) => descrs.get(idx).and_then(|d| d.as_bh_descr()),
        }
    }

    fn len(self) -> usize {
        match self {
            Self::Global => crate::jitcode_runtime::all_descrs().len(),
            Self::PerFn(descrs) => descrs.len(),
        }
    }
}

/// A callee local slot's recording-time concrete, tagged with the frame
/// register it was written through. Own-frame reads (`getarrayitem_vable`) and
/// the mid-body live-value recovery only honor an entry whose `frame_reg`
/// matches the frame they resolve against, so a value stored to another frame's
/// same-indexed slot never leaks across a frame identity.
#[derive(Clone, Copy)]
pub struct CalleeLocalConcrete {
    pub frame_reg: u16,
    pub value: majit_ir::Value,
}

/// Per-inline-level callee locals shadow owned by the walking frame
/// (`MIFrame.registers_i/r/f`, `pyjitpl.py`).
///
/// The maps are keyed by `localsplus` slot rather than register color because
/// pyre lowers callee `LOAD_FAST`/`STORE_FAST` to
/// `getarrayitem_vable`/`setarrayitem_vable(frame, slot)`, while the
/// `WalkContext` register banks are post-regalloc color-indexed. At top level
/// the slot concrete comes from the seeded `virtualizable_boxes`; a fresh
/// inlined callee has no seeded frame, so it owns this slot-indexed shadow.
/// The concrete values survive may-force heapcache clears and only specialize
/// recording; runtime guards still re-check them.
pub struct CalleeLocalsShadow {
    /// SSA value held by each fresh-frame local slot. Own-frame vable accesses
    /// fold through this map without emitting GC ops: `fresh_virtualizable`
    /// makes `is_virtualizable_getset` return false
    /// (`rpython/jit/codewriter/jtransform.py`; the frame is marked by
    /// `pypy/interpreter/pycode.py`).
    pub opref: std::collections::HashMap<i64, OpRef>,
    /// Recording-time concrete held by each local slot, including across
    /// may-force operations that clear the heapcache. Each entry records the
    /// frame register it was written through so a slot read resolves against
    /// its own frame only.
    pub concrete: std::collections::HashMap<i64, CalleeLocalConcrete>,
    /// Portal frame register used to resolve own-frame vable operations.
    /// `u16::MAX` means that the strict fresh-frame fold is inactive.
    pub fold_frame_reg: u16,
}

impl Default for CalleeLocalsShadow {
    fn default() -> Self {
        Self {
            opref: Default::default(),
            concrete: Default::default(),
            fold_frame_reg: u16::MAX,
        }
    }
}

impl CalleeLocalsShadow {
    /// A `NONE` value clears the slot.
    fn set_opref(&mut self, slot: i64, value: OpRef) {
        if value.is_none() {
            self.opref.remove(&slot);
        } else {
            self.opref.insert(slot, value);
        }
    }

    /// A `Void` value clears the slot so a later read does not resurrect a
    /// stale concrete. `frame_reg` is the frame register the write targeted.
    fn set_concrete(&mut self, frame_reg: u16, slot: i64, value: majit_ir::Value) {
        if matches!(value, majit_ir::Value::Void) {
            self.concrete.remove(&slot);
        } else {
            self.concrete
                .insert(slot, CalleeLocalConcrete { frame_reg, value });
        }
    }
}

/// One inlined-callee level of the walk's framestack.
pub struct InlineFrame {
    /// Callee `w_code`, used by the recursion-depth scan. Once the same code
    /// reaches [`FBW_MAX_INLINE_RECURSION`], the call folds to a residual
    /// instead of unrolling its call tree (`pyjitpl.py`).
    pub w_code: usize,
    /// Paused caller snapshot for the multi-frame path. `None` preserves the
    /// straight-line single-frame collapse while retaining the callee level.
    pub parent: Option<InlineParentFrame>,
    /// [`FBW_EXECUTED_EFFECT_COUNT`] when this level was entered, i.e. at the
    /// CALL its `parent` pauses on.  An abort-point flush that resumes the
    /// caller AT that CALL re-executes the call, discarding everything this
    /// level and its descendants did, so it is only sound while the odometer
    /// has not moved since.
    pub entry_executed_effects: usize,
}

/// Per-trace-attempt walk session, owned by the walk driver and threaded
/// through [`WalkContext`] — `MetaInterp.framestack` (`pyjitpl.py`,
/// `:2487`; depth scan `:1390`). Innermost level last.
pub struct WalkSession {
    /// Inlined callee levels. Parent snapshots are outermost-first, matching
    /// `Snapshot.frames`; a caller is pushed at its inline CALL and popped
    /// when the callee sub-walk returns.
    pub framestack: Vec<InlineFrame>,
    /// Whether the terminating permanent abort fired inside an inline
    /// sub-walk. Its `op.pc` is then a callee coordinate with no meaning in
    /// the outer snapshot root's py_pc→jitcode translation, so abort-point
    /// flushing must decline after the sub-walk unwinds.
    pub abort_in_subwalk: bool,
    /// Blackhole `tmpreg_r`/`tmpreg_i`/`tmpreg_f` (`blackhole.py`):
    /// the single-slot scratch that `insert_renamings` (`flatten.py`)
    /// routes a cyclic parallel move through via `*_push`/`*_pop` pairs.
    /// `ref_push/r` writes the source Ref (+ its concrete shadow) here;
    /// `ref_pop/>r` reads it back into a dst register.  Persisted on the
    /// per-walk session because a push and its matching pop straddle
    /// intervening `*_copy` ops within one trampoline.
    ///
    /// The Ref and Int banks carry their concrete shadow alongside the OpRef,
    /// because those shadows live in the `concrete_registers_{r,i}` side
    /// tables: moving the OpRef alone would leave the destination slot's
    /// shadow describing whatever value the move overwrote.  The Float bank
    /// resolves concretes from the OpRef itself, so it needs no companion.
    pub tmpreg_r: OpRef,
    pub tmpreg_r_concrete: ConcreteValue,
    pub tmpreg_i: OpRef,
    pub tmpreg_i_concrete: ConcreteValue,
    pub tmpreg_f: OpRef,
    /// Concrete root frame and jitcode identity for application traceback
    /// recording. Inlined callees carry their own identity separately in
    /// `InlineCalleeConsts`.
    pub recording_frame_ptr: usize,
    pub recording_jitcode_index: i32,
    /// Last opcode executed by the root recording frame.
    pub recording_opcode_position: usize,
}

impl Default for WalkSession {
    fn default() -> Self {
        Self {
            framestack: Vec::new(),
            abort_in_subwalk: false,
            tmpreg_r: OpRef::NONE,
            tmpreg_r_concrete: ConcreteValue::Null,
            tmpreg_i: OpRef::NONE,
            tmpreg_i_concrete: ConcreteValue::Null,
            tmpreg_f: OpRef::NONE,
            recording_frame_ptr: 0,
            recording_jitcode_index: -1,
            recording_opcode_position: 0,
        }
    }
}

fn record_top_level_application_traceback<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: ConcreteValue,
    opcode_position: usize,
    execute_concrete: bool,
    emit_runtime: bool,
) {
    if !ctx.is_top_level {
        return;
    }
    let ConcreteValue::Ref(exc_ptr) = exc_concrete else {
        return;
    };
    if exc_ptr.is_null() {
        return;
    }
    let (frame_ptr, jitcode_index) = {
        let session = ctx.session.borrow();
        (session.recording_frame_ptr, session.recording_jitcode_index)
    };
    if execute_concrete {
        majit_metainterp::record_application_traceback_for_recording(
            exc_ptr as usize as i64,
            frame_ptr as i64,
            jitcode_index,
            opcode_position as i32,
        );
    }
    let hook = majit_metainterp::record_application_traceback_hook_address();
    let frame = crate::state::pyjitcode_for_jitcode_index(jitcode_index)
        .and_then(|jitcode| {
            ctx.registers_r
                .get(jitcode.metadata.portal_frame_reg as usize)
                .copied()
        })
        .unwrap_or(OpRef::NONE);
    if emit_runtime && !hook.is_null() && !exc.is_none() && !frame.is_none() {
        let jitcode = ctx.trace_ctx.const_int(i64::from(jitcode_index));
        let opcode = ctx.trace_ctx.const_int(opcode_position as i64);
        ctx.trace_ctx.call_void_typed_with_effect(
            hook,
            &[exc, frame, jitcode, opcode],
            &[Type::Ref, Type::Ref, Type::Int, Type::Int],
            default_effect_info(),
        );
    }
}

fn record_inline_application_traceback<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: ConcreteValue,
    opcode_position: usize,
    execute_concrete: bool,
    emit_runtime: bool,
) {
    if ctx.is_top_level {
        return;
    }
    let Some(consts) = ctx.inline_callee_consts else {
        return;
    };
    // A non-standard virtualizable frame in a bridge sub-walk carries the
    // `GcRef(usize::MAX)` sentinel (or null) as `w_code` instead of a real
    // `PyCode` — a synthetic frame with no Python code to anchor a node on.
    // The host adapter (`record_inline_traceback_for_recording`) dereferences
    // `w_code` through `createframe_obj`, so a sentinel / garbage pointer would
    // SIGSEGV.  Skip null / sentinel / non-code; the null + sentinel checks run
    // before `is_code`, whose `py_type_check` would deref the raw sentinel
    // (`CAN_BE_TAGGED` is off).
    if consts.w_code == 0 || consts.w_code == usize::MAX {
        return;
    }
    if !unsafe { pyre_interpreter::pycode::is_code(consts.w_code as pyre_object::PyObjectRef) } {
        return;
    }
    let ConcreteValue::Ref(exc_ptr) = exc_concrete else {
        return;
    };
    if exc_ptr.is_null() {
        return;
    }
    if execute_concrete {
        majit_metainterp::record_inline_application_traceback_for_recording(
            exc_ptr as usize as i64,
            consts.w_code as i64,
            consts.w_globals as i64,
            consts.jitcode_index,
            opcode_position as i32,
        );
    }
    let hook = majit_metainterp::record_inline_application_traceback_hook_address();
    if emit_runtime && !hook.is_null() && !exc.is_none() {
        let w_code = ctx.trace_ctx.const_ref(consts.w_code as i64);
        let w_globals = ctx.trace_ctx.const_ref(consts.w_globals as i64);
        let jitcode = ctx.trace_ctx.const_int(i64::from(consts.jitcode_index));
        let opcode = ctx.trace_ctx.const_int(opcode_position as i64);
        ctx.trace_ctx.call_void_typed_with_effect(
            hook,
            &[exc, w_code, w_globals, jitcode, opcode],
            &[Type::Ref, Type::Ref, Type::Ref, Type::Int, Type::Int],
            default_effect_info(),
        );
    }
}

fn recording_instruction_is_bare_reraise<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    opcode_position: usize,
) -> bool {
    let jitcode_index = if ctx.is_top_level {
        ctx.session.borrow().recording_jitcode_index
    } else {
        ctx.inline_callee_consts
            .map_or(-1, |consts| consts.jitcode_index)
    };
    crate::state::jitcode_pc_is_bare_reraise(jitcode_index, opcode_position as i32)
}

/// The frame identity, code object and instruction coordinate one
/// `PyTraceback` node needs, resolved from the walk's current frame.
struct TracebackNodeSite {
    /// `PyTraceback.frame`.  `OpRef::NONE` when the walk has no materialized
    /// frame for this level — a branchless leaf inlined without one leaves
    /// the color unseeded.  Recording NONE as a `SetfieldGc` operand would
    /// put a bogus box in the trace, so every caller has to DECLINE on it and
    /// leave the node to the frame-fabricating hook.
    frame: OpRef,
    w_code: usize,
    last_instruction: i32,
    lineno: i64,
}

/// Resolve the `PyTraceback` fields of the frame the walk is currently in.
///
/// `None` means no node can be built here: `w_code` is missing, is the
/// `GcRef(usize::MAX)` sentinel a non-standard virtualizable sub-walk
/// carries, is not a code object, or is `hidden_applevel`
/// (`pytraceback.py record_application_traceback`'s first line — such a
/// frame contributes no node at all).  The null / sentinel tests run before
/// `is_code`, whose `py_type_check` would dereference the raw sentinel.
fn traceback_node_site<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode_position: usize,
) -> Option<TracebackNodeSite> {
    let (jitcode_index, w_code) = if ctx.is_top_level {
        let session = ctx.session.borrow();
        let frame = session.recording_frame_ptr as *const pyre_interpreter::PyFrame;
        let w_code = if frame.is_null() {
            0
        } else {
            unsafe { (*frame).pycode as usize }
        };
        (session.recording_jitcode_index, w_code)
    } else {
        ctx.inline_callee_consts
            .map_or((-1, 0), |consts| (consts.jitcode_index, consts.w_code))
    };
    if w_code == 0 || w_code == usize::MAX {
        return None;
    }
    if !unsafe { pyre_interpreter::pycode::is_code(w_code as pyre_object::PyObjectRef) } {
        return None;
    }
    if unsafe {
        pyre_interpreter::pycode::w_code_hidden_applevel(w_code as pyre_object::PyObjectRef)
    } {
        return None;
    }
    let jitcode = crate::state::pyjitcode_for_jitcode_index(jitcode_index)?;
    let last_instruction =
        crate::state::python_pc_for_jitcode_pc_public(jitcode_index, opcode_position as i32)?;
    let raw_code = crate::state::raw_code_for_jitcode_index(jitcode_index)?;
    let lineno =
        unsafe { pyre_interpreter::pyframe::offset2lineno(&*raw_code, last_instruction as isize) }
            as i64;
    let frame = ctx
        .registers_r
        .get(jitcode.metadata.portal_frame_reg as usize)
        .copied()
        .unwrap_or(OpRef::NONE);
    Some(TracebackNodeSite {
        frame,
        w_code,
        last_instruction,
        lineno,
    })
}

/// Emit the `NewWithVtable` + field stores that build one `PyTraceback`
/// node, and stamp it as `exc`'s traceback.  `w_next` is the node's tail.
fn emit_traceback_node<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    kind: pyre_object::interp_exceptions::ExcKind,
    site: &TracebackNodeSite,
    w_next: OpRef,
) {
    let traceback = ctx.trace_ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::pytraceback_size_descr(),
    );
    ctx.trace_ctx.heap_cache_mut().new_object(traceback);
    let fields = [
        (
            ctx.trace_ctx
                .const_ref(pyre_object::pyobject::get_instantiate(
                    &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE,
                ) as i64),
            0,
        ),
        (site.frame, 1),
        (ctx.trace_ctx.const_int(i64::from(site.last_instruction)), 2),
        (w_next, 3),
        (ctx.trace_ctx.const_int(site.lineno), 4),
        (ctx.trace_ctx.const_ref(site.w_code as i64), 5),
    ];
    for (value, index) in fields {
        let descr = crate::descr::pytraceback_field_descr(index);
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[traceback, value], descr.clone());
        ctx.trace_ctx
            .heapcache_setfield_cached(traceback, descr.index(), value);
    }

    // `f_lineno` resolves through `offset2lineno(pycode, last_instr)` on every
    // read, so the frame itself has to carry the coordinate — the node's own
    // `tb_lasti` answers a different question and is frozen. The interpreter
    // gets this for free from `pyopcode.py`'s per-opcode `last_instr` store;
    // compiled code does not run it, and `fbw_publish_exit_last_instr` only
    // reaches the virtualizable, so an inlined callee frame would keep the `-1`
    // initialization sentinel and report its `def` line. A frame that goes on
    // running has this overwritten by its own later publish, exactly as the
    // per-opcode store would.
    let last_instr_value = ctx.trace_ctx.const_int(i64::from(site.last_instruction));
    let last_instr_descr = crate::descr::pyframe_next_instr_descr();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[site.frame, last_instr_value],
        last_instr_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(site.frame, last_instr_descr.index(), last_instr_value);

    let traceback_descr = crate::descr::w_exception_traceback_descr(kind);
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[exc, traceback],
        traceback_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(exc, traceback_descr.index(), traceback);
}

/// IR-virtual PREPEND of one `PyTraceback` node — the general-case sibling
/// of [`record_fresh_application_traceback`].
///
/// `pytraceback.py record_application_traceback`:
///
/// ```text
///     tb = operror.get_traceback()
///     tb = PyTraceback(space, frame, last_instruction, tb)
///     operror.set_traceback(tb)
/// ```
///
/// The fresh variant hard-codes `w_next = NULL`, sound only for an exception
/// the walk just built.  Here `get_traceback()` is emitted as a real
/// `GETFIELD_GC_R` on the per-kind `w_traceback` descr and threaded into
/// `w_next`, so the node prepends instead of truncating the chain a callee
/// already contributed to.
///
/// Every op is optimizer-visible, so escape analysis deletes the node when
/// the exception never escapes.  The opaque
/// `record_{,inline_,top_level_}application_traceback` hooks cannot be: they
/// carry `default_effect_info()` — `EffectInfo::MOST_GENERAL`, i.e. random
/// effects — so the optimizer escapes every argument, forces all lazy sets
/// and drops the whole heapcache, and the top-level one additionally passes
/// the virtualizable frame box, forcing the vable.
///
/// Returns `false` when nothing was emitted; the caller then keeps the
/// opaque hook.
fn record_prepend_application_traceback<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: ConcreteValue,
    opcode_position: usize,
) -> bool {
    if exc.is_none() || exc.is_constant() {
        // A `Const` exception box freezes the RECORDING iteration's address
        // (`walker_record_guard_exception` pins every raise after the first
        // one in a walk), so reading and writing `w_traceback` through it
        // would chain onto a stale object on every later iteration.  The
        // opaque hook takes the live exception as an argument instead.
        return false;
    }
    let ConcreteValue::Ref(exc_ptr) = exc_concrete else {
        return false;
    };
    if exc_ptr.is_null() || unsafe { !pyre_object::is_exception(exc_ptr) } {
        return false;
    }
    let Some(site) = traceback_node_site(ctx, opcode_position) else {
        return false;
    };
    if site.frame.is_none() {
        // No materialized frame for this level.  The opaque inline hook
        // fabricates one from the promoted callee metadata
        // (`record_inline_traceback_for_recording`), so `tb_frame` keeps
        // answering a real frame there; a null here would lose it.
        //
        // Top level has no such hook fallback: `record_top_level_application_
        // traceback` resolves its frame operand from the same unseeded color
        // and declines too, so that level contributes no node at all.  A
        // shorter traceback than `record_application_traceback` would build is
        // the deliberate trade — the frame box a node needs is exactly what
        // this walk lacks, and the vable box is not a substitute (a traceback
        // outlives the frame, so storing it demands the escape marking this
        // path does not perform).
        return false;
    }
    let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc_ptr) };
    // `tb = operror.get_traceback()`.  MUST precede the node's own store, or
    // the heapcache answers with the node being built and `w_next` self-links.
    let traceback_descr = crate::descr::w_exception_traceback_descr(kind);
    let w_next = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, exc, traceback_descr);
    emit_traceback_node(ctx, exc, kind, &site, w_next);
    true
}

/// IR-virtual traceback record for an exception the walk itself built: the
/// node starts a FRESH chain, so `w_next` is NULL rather than the
/// exception's current traceback.  Sound only where the caller has proven
/// the exception carries no prior node (`fbw_built_exc_take`); the general
/// case is [`record_prepend_application_traceback`].
///
/// Returns `false` when nothing was emitted; the caller then keeps the
/// opaque hook.
fn record_fresh_application_traceback<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: ConcreteValue,
    opcode_position: usize,
) -> bool {
    let ConcreteValue::Ref(exc_ptr) = exc_concrete else {
        return false;
    };
    if exc_ptr.is_null() || unsafe { !pyre_object::is_exception(exc_ptr) } {
        return false;
    }
    let Some(site) = traceback_node_site(ctx, opcode_position) else {
        return false;
    };
    if site.frame.is_none() {
        // Same disposition as the prepend sibling, for the same reason: the
        // opaque inline hook fabricates a frame from the promoted callee
        // metadata, while a null one answers None and breaks every consumer
        // that follows `tb_frame.f_code` — `traceback.print_exc` among them.
        return false;
    }
    let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc_ptr) };
    let w_next = ctx.trace_ctx.const_ref(0);
    emit_traceback_node(ctx, exc, kind, &site, w_next);
    true
}

/// Compile-time-constant frame fields of an inlined callee.
#[derive(Clone, Copy)]
pub struct InlineCalleeConsts {
    /// `frame.w_globals` object (`VABLE_NAMESPACE_FIELD_IDX` = 5): the
    /// callee function's `__globals__` as a `PyObjectRef`.
    w_globals: usize,
    /// `frame.pycode` (`VABLE_CODE_FIELD_IDX` = 1): the callee's `W_Code`
    /// pointer.
    w_code: usize,
    /// Jitcode identity used to translate a callee opcode to its Python
    /// instruction coordinate when constructing `PyTraceback` metadata.
    jitcode_index: i32,
}

/// Walk-scoped FBW modes inherited parent-to-child across sub-walk
/// constructions. Formerly three thread-locals.
pub struct FbwWalkMode<Sym: WalkSym> {
    /// Full-body snapshot root used to map a guard's jitcode `op_pc` through
    /// the outer jitcode and read the live walk banks. Null for per-opcode
    /// walks, where `walker_capture_snapshot_for_last_guard` keeps its legacy
    /// single-coordinate behavior.
    pub snapshot_sym: *const Sym,
    /// Guards emitted in an inline sub-walk resume at the caller's CALL
    /// boundary rather than mapping the callee `op_pc` through the outer
    /// jitcode in `walker_capture_snapshot_for_last_guard`.
    pub inline_subwalk: bool,
    /// The current sub-walk is a translated builtin gateway helper, which has
    /// no Python blackhole frame of its own. Guard snapshots therefore encode
    /// only the precomputed Python caller chain on `framestack`, omitting the
    /// helper JitCode itself.
    pub transparent_helper_subwalk: bool,
    /// A bridge-carrier resume folds nested self-recursive calls directly to
    /// `CALL_ASSEMBLER` (`opimpl_recursive_call_assembler`) rather than
    /// re-unrolling the call tree to the multi-frame depth cap.
    pub carrier_resume: bool,
    /// Bridge-entry view of `ExecutionContext.sys_exc_value` reconstructed
    /// from the failing guard's pending setfield.  The walk is temporally
    /// displaced from the guard failure, so live TLS is not a valid source
    /// until a walked SETFIELD replaces this seed.
    pub current_exception_seed: Option<OpRef>,
    /// Concrete shadow paired with [`FbwWalkMode::current_exception_seed`].
    pub current_exception_seed_concrete: pyre_object::PyObjectRef,
    /// Walker-carried mirror of
    /// `MetaInterp.class_of_last_exc_is_const` (`pyjitpl.py`).
    /// This is shared logically across recursive MIFrame walks; catch routing
    /// writes the proven-class state back into the caller's copy.
    pub class_of_last_exc_is_const: bool,
    /// Python-pc of a guard-failure bridge walk's own resume coordinate.
    /// `Some` only on the top-level walk of a bridge trace, `None` otherwise
    /// (loop compiles and sub-walks).
    ///
    /// `generate_guard(resumepc=orgpc)` (`pyjitpl.py:2610-2626`) places a
    /// guard's resume coordinate INSIDE the guarded opcode's implementation,
    /// strictly past the dispatch-top `jit_merge_point`, so an RPython MIFrame
    /// resumed from a guard never re-crosses the loop-header merge point at
    /// position zero. Pyre's bridge walker instead resumes at the opcode
    /// BOUNDARY and would re-cross the header immediately with an empty body,
    /// closing a 0-progress no-op bridge. The merge-point arm consumes this
    /// (via `take()`) to skip exactly the first crossing that lands on the
    /// bridge's own resume coordinate, restoring the RPython positional
    /// semantics.
    pub bridge_entry_merge_pc: Option<usize>,
}

impl<Sym: WalkSym> Clone for FbwWalkMode<Sym> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Sym: WalkSym> Copy for FbwWalkMode<Sym> {}

/// The outer snapshot's Python-PC coordinate. Non-root producers preserve the
/// raw JitCode offset that produced the Python word, postponing the exact
/// `backxlat_py_pc` inversion until a consumer needs it. Root entries and test
/// fixtures have no such native coordinate and retain their Python value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EntryPyPc {
    Py(u32),
    Jit(usize),
}

impl EntryPyPc {
    fn audit_variant(self) -> &'static str {
        match self {
            Self::Py(_) => "py",
            Self::Jit(_) => "jit",
        }
    }
}

impl<Sym: WalkSym> Default for FbwWalkMode<Sym> {
    fn default() -> Self {
        Self {
            snapshot_sym: std::ptr::null(),
            inline_subwalk: false,
            transparent_helper_subwalk: false,
            carrier_resume: false,
            current_exception_seed: None,
            current_exception_seed_concrete: pyre_object::PY_NULL,
            class_of_last_exc_is_const: false,
            bridge_entry_merge_pc: None,
        }
    }
}

/// `WalkContext` carries two lifetimes:
/// * `'frame` — the inner-frame lifetime: register banks + trace
///   recorder. Sub-walk recursion (`inline_call_r_r/dR>r`) allocates
///   fresh register banks scoped to the sub-walk's block, so
///   `'frame` must be allowed to *shrink* on recursion.
/// * `'static_a` — the outer lifetime: descr pool + sub-jitcode
///   lookup. These flow unchanged from caller into callee, so they
///   keep their original (longer) lifetime.
pub struct WalkContext<'frame, 'static_a: 'frame, Sym: WalkSym> {
    /// Present only for an inlined-callee sub-walk. Top-level and other walks
    /// have no callee shadow, preserving the former empty-stack no-op behavior.
    pub callee_shadow: Option<CalleeLocalsShadow>,
    /// Compile-time-constant frame fields of this inlined callee's own
    /// unseeded portal frame. This is the walk-time equivalent of the
    /// codewriter's non-portal branch (`codewriter.rs`),
    /// where `pycode` and `w_globals` are constants rather than
    /// reads that alias the caller's frame. `None` when this is not an
    /// inlined-callee sub-walk.
    pub inline_callee_consts: Option<InlineCalleeConsts>,
    /// FBW walk modes inherited by nested sub-walk contexts.
    pub fbw_mode: FbwWalkMode<Sym>,
    /// Caller-owned state shared by every frame in this walk attempt.
    pub session: &'static_a std::cell::RefCell<WalkSession>,
    /// Symbolic Ref-bank register file. Indexing matches RPython
    /// `MIFrame.registers_r` (`pyjitpl.py`); the byte after a
    /// `r`-coded operand opcode indexes directly into this slice.
    /// Mutable so handlers writing `>r` results (currently
    /// `residual_call_r_r/iRd>r`) can land their dst.
    pub registers_r: &'frame mut [OpRef],
    /// Symbolic Int-bank register file. Indexing matches RPython
    /// `MIFrame.registers_i` (`pyjitpl.py`). Pyre's PyreSym is
    /// mid-migration to a 3-bank typed model — production callers may
    /// pass an empty slice today (the assembler only emits `i`-coded
    /// operands once the codewriter wires Int kind). Mutable so
    /// `int_copy/i>i` can land its dst.
    pub registers_i: &'frame mut [OpRef],
    /// Symbolic Float-bank register file. Indexing matches RPython
    /// `MIFrame.registers_f` (`pyjitpl.py`). Mutable so
    /// `float_<binop>/ff>f` and `float_neg/f>f` can land their dst.
    pub registers_f: &'frame mut [OpRef],
    /// Concrete shadow mirror for `registers_r`.
    ///
    /// Semantic-slot indexed, length equals `registers_r.len()`. At
    /// `dispatch_via_miframe` entry, populated by concatenating
    /// `PyreSym.concrete_locals` + `PyreSym.concrete_stack`; sub-walks
    /// allocate a fresh `Vec<ConcreteValue>` sized to the callee's
    /// `num_regs_r` and fill arg slots from the parent's slice at the
    /// arg byte indices.
    ///
    /// **Mutable invariant**: every walker handler that
    /// writes `registers_r[dst]` MUST also write `concrete_registers_r
    /// [dst]` in lock-step.  Use the [`write_ref_reg`] helper which
    /// enforces this contract.  Sites that don't know the result's
    /// concrete pass `ConcreteValue::Null` — downstream consumers
    /// (e.g. `raise/r` GUARD_CLASS gate) treat `Null` as "no info,
    /// skip the guard", same as slots the snapshot never populated.
    /// Copy-style handlers (`ref_copy/r>r`,
    /// `last_exc_value/>r`) propagate the source's concrete.
    ///
    /// The slice is mutable so the concrete shadow tracks the symbolic
    /// register in lock-step. If it were immutable, sibling handlers
    /// like `last_exc_value/>r` could rewrite the symbolic register
    /// without touching the concrete snapshot, so a follow-on `raise/r`
    /// would read a stale concrete and silently skip the GUARD_CLASS
    /// gate; the lock-step contract keeps walker-side GUARD_CLASS sound.
    ///
    /// **Companion bank** `concrete_registers_i` below now exists as a
    /// skeleton field (Concrete shadow skeleton) so handlers can plumb concrete int
    /// shadow without changing the WalkContext signature again.  Seed
    /// wiring (real concrete int values at trace entry + per-handler
    /// writes for `int_*` arithmetic) is deferred to Concrete shadow seeding and
    /// later slices.  Until then every callsite passes `&mut []`.
    ///
    /// `goto_if_not/iL` and `switch/id` (which consume concrete Int)
    /// continue to fall back to the strict-mode fail-loud path until
    /// `concrete_registers_i` is populated.
    pub concrete_registers_r: &'frame mut [ConcreteValue],
    /// **Skeleton — Concrete shadow skeleton.**  Concrete shadow mirror for
    /// `registers_i`.  Color-indexed (not semantic-slot indexed like
    /// `concrete_registers_r`) because pyre's Int bank has no
    /// "semantic-slot" abstraction — Int registers are post-regalloc
    /// colors directly.  Length equals `registers_i.len()` when
    /// populated, or `0` when callsites pass `&mut []`.
    ///
    /// Future invariant (Concrete shadow seeding+): every walker handler that writes
    /// `registers_i[dst]` MUST also write `concrete_registers_i[dst]`
    /// in lock-step, mirroring the Ref-bank contract above.  A future
    /// `write_int_reg` helper will enforce this once seeding lands.
    ///
    /// Consumers (`dispatch_goto_if_not/iL`, `switch/id`) currently
    /// fall back to the strict-mode fail-loud path; they will switch
    /// to reading `concrete_registers_i[src]` for the test-direction
    /// fold once Concrete shadow seeding populates seeds.
    pub concrete_registers_i: &'frame mut [ConcreteValue],
    /// Descr pool for `d`-coded operands. Each `d` argcode in the
    /// jitcode bytes resolves to `descr_refs[2-byte LE index]`.
    /// RPython `Assembler.descrs` (`assembler.py`) +
    /// `BlackholeInterpBuilder.setup_descrs` (`blackhole.py`)
    /// — production callers pass the codewriter-emitted descr table.
    pub descr_refs: &'static_a [DescrRef],
    /// Raw `BhDescr` pool source for vable-array `(VableArray, Array)`
    /// recognition (`vable_array_descrs_from_jitcode`).  [`RawDescrPool::
    /// Global`] for build-time canonical jitcodes + tests (their operands
    /// index the shared `ALL_DESCRS`); [`RawDescrPool::PerFn`] for full-body
    /// walks (the per-`CodeObject` body resolves through its own
    /// `exec.descrs`).  Index-parallel to [`Self::descr_refs`].
    pub raw_descrs: RawDescrPool<'static_a>,
    /// Whether this walk is the SOLE concrete-execution leg.
    ///
    /// `false` (shadow validation, the diagnostic full-body probe,
    /// tests): a separate concrete interpreter (the Python interpreter,
    /// or none for the discard-the-trace probe) is
    /// authoritative, so the walker must NOT re-execute may-force
    /// residual calls — doing so would double their side effects /
    /// corrupt the live heap (`cut_trace` rolls back only the IR
    /// recorder, not heap/iterator state).
    ///
    /// `true` (the production full-body walk and its inline
    /// sub-walks): the walker is the only thing executing the
    /// JitCode body — `eval_loop_jit` skips `execute_opcode_step` for
    /// walker-handled opcodes — so
    /// [`try_execute_residual_call_via_executor`] runs residual calls
    /// concretely.  RPython parity: the metainterp executes EVERY
    /// residual_call during tracing (`do_residual_call` →
    /// `executor.execute_varargs`, pyjitpl.py), pure or not, so a
    /// downstream `goto_if_not` reads a concrete result.
    pub is_authoritative_executor: bool,
    /// Live trace recorder. `record_finish` / `record_op` /
    /// `record_op_with_descr` go through this.
    pub trace_ctx: &'frame mut TraceCtx,
    /// Whether this `WalkContext` is the outermost trace frame
    /// (`true`) or a nested sub-jitcode frame entered through
    /// `inline_call_r_r/dR>r` recursion (`false`). The flag
    /// disambiguates dual-behaviour terminators:
    ///
    /// * `ref_return/r` at top-level stashes the finish payload +
    ///   Terminate; inside a sub-walk it returns `SubReturn { result }`
    ///   so the caller's `inline_call_*` handler can write the dst
    ///   register.
    /// * `raise/r` propagates `SubRaise { exc }` either way; only at
    ///   top-level, with no handler anywhere in the framestack, does the
    ///   post-op scan turn it into an `is_exception` finish payload.  In
    ///   a sub-walk the caller's `inline_call_*` handler may catch via
    ///   `catch_exception` metadata or bubble up further.
    ///
    /// RPython parity: pyre flattens the framestack-driven
    /// `metainterp.popframe()` + `finishframe[_exception]` flow
    /// (`pyjitpl.py`) into this Rust-level outcome.
    pub is_top_level: bool,
    /// Caller-provided callback resolving a `jitcode_index` to a
    /// `SubJitCodeBody`. Invoked when `inline_call_r_r/dR>r` fires
    /// and needs to recurse into the callee's bytecode body.
    pub sub_jitcode_lookup: &'static_a SubJitCodeLookup,
    /// Per-frame mirror of RPython `metainterp.last_exc_value`
    /// (`pyjitpl.py`). Set by `raise/r` (caller-frame side, before
    /// `SubRaise` propagates) and by the `inline_call` SubRaise arm
    /// when it catches at a `catch_exception/L` handler (the handler's
    /// own opcodes — `last_exception`, `last_exc_value`, `reraise/` —
    /// read this field). RPython keeps this on the metainterp object
    /// (one shared slot); the walker carries one per WalkContext
    /// because each recursive frame has its own context. The flow
    /// (callee raise → caller catch → caller handler reads) only
    /// touches the caller's slot, so per-frame storage is equivalent
    /// to RPython's metainterp-level slot for the catch path.
    pub last_exc_value: Option<OpRef>,
    /// Concrete shadow companion to [`last_exc_value`].
    /// Holds the live `PyObjectRef` of the standing
    /// exception so `last_exc_value/>r` can propagate the concrete
    /// into the destination's `concrete_registers_r` slot, and so a
    /// follow-on `raise/r` reading that destination finds a non-Null
    /// concrete and emits the correct GUARD_CLASS.
    ///
    /// Set by `raise/r` (walker side) alongside `last_exc_value`, by
    /// the `inline_call` SubRaise arm when it catches at
    /// `catch_exception/L`, and by `dispatch_via_miframe`'s entry
    /// from `sym.last_exc_value` when an adapter caller seeded the exception.
    ///
    /// `ConcreteValue::Null` means "no active exception concrete
    /// known" — matches `last_exc_value == None` for the common case,
    /// or means an adapter caller seeded only the symbolic OpRef without
    /// a concrete (e.g. a synthetic test fixture).
    pub last_exc_value_concrete: ConcreteValue,
    /// Outer snapshot coordinate. Root entries keep `MIFrame.orgpc`; every
    /// coordinate-native producer stores its raw JitCode offset and resolves
    /// the matching Python PC lazily at the consumer boundary.
    ///
    /// Read by [`walker_capture_snapshot_for_last_guard`] to stamp the
    /// snapshot frame's Python PC.
    pub entry_py_pc: EntryPyPc,
    /// Codewrite-time resume-marker twin for the outer snapshot coordinate
    /// (`entry_py_pc`), carried by the arm-path snapshot word. `None` when the
    /// creation site has no
    /// jitcode-native outer coordinate.
    pub outer_resume_marker_jit_pc: Option<usize>,
    /// JitCode index of the **outer** `PyJitCode.jitcode` — the Python
    /// bytecode jitcode whose Python opcode is currently being
    /// dispatched.  Pyre's blackhole resume only re-enters Python-
    /// bytecode jitcodes, so guard snapshots must reference the outer
    /// pyjitcode regardless of how deep the walker's sub-walk nesting
    /// is.
    ///
    /// The retired per-opcode arm entry read this from
    /// `(*sym.jitcode).index()`; inline sub-walks seed it from the
    /// CALL-site capture (sub-walks don't change the outer Python
    /// opcode).  Test fixtures + [`dispatch_via_miframe`] default to
    /// `0` — the full-body guard capture reads `sym.jitcode` directly
    /// instead of this field.
    pub outer_jitcode_index: u32,
    /// Frozen `PyFrame` state at the outer Python opcode boundary —
    /// `sym.registers_r ∪ sym.registers_i.opref ∪ sym.registers_f.opref`
    /// captured at walk entry (the retired per-opcode arm entry did
    /// this; inline sub-walks seed it from the CALL-site capture; the
    /// full-body root leaves it empty and collects at guard capture),
    /// filtered by `OpRef::is_none()`.  This is what
    /// [`walker_capture_snapshot_for_last_guard`] passes as the
    /// snapshot frame's active boxes on the arm path.
    ///
    /// Sub-walks clone the parent's Vec — outer active-box count is
    /// small (a Python frame's live locals + stack tail) and walker
    /// nesting depth is shallow (2–3 levels), so the per-sub-walk
    /// clone cost is negligible.
    pub outer_active_boxes: Vec<OpRef>,
    /// Runtime address of `bh_store_subscr_fn` (pyre-jit's
    /// `cpu.store_subscr_fn` binding) used by
    /// `try_walker_store_subscr_specialization` to recognise the
    /// 3-arg `residual_call_r_v(store_subscr_fn, obj, key, value)`
    /// emitted by `codewriter.rs
    /// build_store_subscr_fn_residual_call_r_v_insn`.
    ///
    /// Every root entry currently passes `None` (the retired
    /// per-opcode arm entry was the caller that plumbed the address
    /// through pyre-jit's `cpu.store_subscr_fn`).  Sub-walks inherit
    /// the parent's value.  `None` disables the field-based
    /// specialization gate.
    ///
    /// `PYRE_WALKER_STORE_SUBSCR_FNADDR` is read as the fallback when this
    /// field is `None`, keeping test fixtures and runtime overrides from
    /// needing a full production `MIFrame` entry.
    pub store_subscr_fn_addr: Option<usize>,
    /// Snapshot-capture failure latched by the `WalkerFrameOps`
    /// `generate_guard` impl, whose `()` trait signature (shared with
    /// `MIFrame`) has no error channel.  The STORE_SUBSCR
    /// specialization drives the `majit-translate` codegen helpers over
    /// this context; its dispatcher call site drains the latch and
    /// surfaces the `DispatchError` so a guard recorded without a resume
    /// snapshot aborts the walk instead of compiling.
    pub pending_guard_snapshot_error: Option<DispatchError>,
    /// PyPy-faithful kept-operand-stack snapshot: the walk-level
    /// symbolic operand stack, indexed by ABSOLUTE operand-stack depth
    /// (slot `s`, `s in 0..vstack_depth`).  The Python operand stack is
    /// all-Ref (`W_Root`), so a single `Vec<OpRef>` (Ref bank) suffices.
    /// This is the walker analog of PyPy's `MIFrame.registers_r`
    /// valuestack array snapshotted by `get_list_of_active_boxes`
    /// (`pyjitpl.py`) — the authoritative per-slot box source the
    /// `stack_sync` vable overlay reads at a branch guard instead of the
    /// unreliable `registers_r[stack_slot_color_map[s]]` static-color read.
    ///
    /// Maintained ONLY when `sym.owns_virtualizable_shadow()`; on any
    /// unmodeled stack effect the maintenance sets `vstack_valid = false`
    /// and `stack_sync` omits every operand slot, which resume
    /// re-materializes (zero regression).
    ///
    /// The mirror is the SOLE kept-stack source at a branch guard: the
    /// flat `stack_slot_color_map` static-color read it once fell back to
    /// is retired, so a slot the mirror does not cover is omitted rather
    /// than read from the flat map.  `PYRE_VSTACK_DIAG` logs the per-op
    /// reconcile trace.
    pub vstack_boxes: Vec<OpRef>,
    /// #73: the absolute operand-stack depth `vstack_boxes` currently
    /// reflects — the depth ON ENTRY to the Python opcode at
    /// `vstack_cur_pypc` (i.e. AFTER the previous opcode's stack effect
    /// was reconciled).
    pub vstack_depth: usize,
    /// #73: the Python pc of the opcode currently being walked.  A change
    /// in `python_pc_for_jitcode_pc(jit_pc)` from this value marks a
    /// Python-opcode boundary, where the previous opcode's stack effect is
    /// reconciled into `vstack_boxes` (see [`reconcile_vstack_at_boundary`]).
    pub vstack_cur_pypc: u32,
    /// #73: whether `vstack_boxes` is a trustworthy mirror of the live
    /// operand stack.  Set `false` at walk entry until seeded, and latched
    /// `false` permanently on the first unmodeled stack effect so the
    /// `stack_sync` overlay declines to use it.
    pub vstack_valid: bool,
    /// #73: the last Ref box written via [`write_ref_reg`] during the
    /// CURRENT Python opcode — the box a value-producing opcode lands on
    /// the operand-stack TOS.  Reset to `OpRef::NONE` at every opcode
    /// boundary; read by [`reconcile_vstack_at_boundary`] for the
    /// RESULT-TO-TOS class.
    pub vstack_last_ref: OpRef,
    /// #389(b): the py_pc the walk backed off FROM when it entered the
    /// codewriter's out-of-order FOR_ITER-entry permutation lowering
    /// (`SWAP`/`BUILD_LIST`/`SWAP` before a `FOR_ITER`, lowered non-monotonically
    /// in jitcode).  `u32::MAX` when not inside such a region.  While set, the
    /// per-op reconcile is invalid (the walk re-visits already-passed py_pcs out
    /// of order), so `reconcile_vstack_at_boundary` reseeds the whole mirror from
    /// the virtualizable shadow instead of replaying stack effects.  Cleared once
    /// the walk advances past this ceiling (py-pc order is monotonic again).
    pub vstack_reorder_ceiling: u32,
    /// #73: the jitcode offset of the `-live-` byte that
    /// precedes the CURRENT opcode's guard resume point — the `-live-`
    /// BEFORE (`pyjitpl.py`, normal guard resume reads at
    /// `self.pc - SIZE_LIVE_OP`).  Maintained purely from `DecodedOp` by the
    /// walk; `usize::MAX` until the first `-live-` is seen.  Side-data only.
    pub live_before_jit_pc: usize,
    /// #73: the jitcode offset of the `-live-` byte that
    /// trails a residual-call opcode — the `-live-` AFTER (`pyjitpl.py`,
    /// residual-call guard resume reads at `self.pc`).  Maintained purely
    /// from `DecodedOp` by the walk; `usize::MAX` until set.  Side-data only.
    pub live_after_jit_pc: usize,
}

impl<Sym: WalkSym> WalkContext<'_, '_, Sym> {
    /// Resolve the outer snapshot coordinate at the exact consumer boundary.
    fn entry_py_pc(&self) -> u32 {
        match self.entry_py_pc {
            EntryPyPc::Py(py_pc) => py_pc,
            EntryPyPc::Jit(jitcode_pc) => crate::state::forward_py_pc_or_backxlat(
                self.outer_jitcode_index as i32,
                jitcode_pc as i32,
            ) as u32,
        }
    }
}

/// Outcome of dispatching one opcode. The walker uses this to decide
/// whether to continue stepping or terminate.
///
/// RPython parity: `pyjitpl.py:opimpl_*` returns through Python's
/// generator/exception flow — opcodes that end a trace raise
/// `DoneWithThisFrameRef`/`SwitchToBlackhole`/`ChangeFrame`. Pyre
/// flattens that into an explicit enum because Rust has no analogous
/// non-local exit and we want the walker to stay in plain Result form.
///
/// Not `Copy`: the `CloseLoop` variant carries a `Vec<OpRef>` of merge-point
/// jump args, mirroring `TraceAction::CloseLoopWithArgs` (`lib.rs`)
/// which is likewise non-`Copy`.
#[derive(Debug, Clone)]
pub enum DispatchOutcome {
    /// Step succeeded, continue with the next opcode at the returned pc.
    Continue,
    /// Trace ends here. The arm produced a final `ref_return`/`raise`
    /// equivalent at the top-level frame and no further bytes should
    /// be walked.
    Terminate,
    /// Sub-walk frame returned with a result OpRef (Some) or void
    /// (None — no `>X` slot in the callee's `*_return` op). Surfaced
    /// only when `WalkContext::is_top_level == false`. The caller's
    /// `inline_call_r_r/dR>r` handler consumes this to write the dst
    /// register and continue stepping its own jitcode.
    ///
    /// RPython parity: `metainterp.popframe()` after an `opimpl_*_return`
    /// (`pyjitpl.py`) — the callee frame ends, control returns
    /// to the caller's metainterp loop with the resbox in hand.
    SubReturn { result: Option<OpRef> },
    /// Sub-walk frame raised. RPython
    /// `metainterp.popframe() + finishframe_exception()` walks up the
    /// framestack scanning each parent's exceptiontable; pyre's walker
    /// surfaces the outcome to the caller's `inline_call_*` handler,
    /// which today bubbles it up further (no per-handler
    /// exceptiontable scan yet — that lives behind the
    /// `catch_exception/L` metadata pipe and is deferred until the
    /// per-PC exceptiontable plumb-through lands).
    ///
    /// `exc_concrete` carries the callee's `last_exc_value_concrete`
    /// across the frame boundary so the caller's `inline_call` SubRaise
    /// arm can seed its own `last_exc_value_concrete` and a downstream
    /// `raise/r` / `reraise/` reads the right concrete for GUARD_CLASS
    /// emission. Empty when the callee itself didn't track a concrete
    /// (e.g. shadow gap or `Null`-seeded raise).
    SubRaise {
        exc: OpRef,
        exc_concrete: ConcreteValue,
    },
    /// Trace recording must abort and resume in blackhole mode.
    ///
    /// RPython parity: `pyjitpl.py` routes
    /// `OS_NOT_IN_TRACE` residual calls through
    /// `do_not_in_trace_call`; `pyjitpl.py` raises
    /// `SwitchToBlackhole(ABORT_ESCAPE)` if the concrete call raises.
    /// The trace walker cannot execute the callee concretely yet, so a
    /// reached `OS_NOT_IN_TRACE` site is surfaced as this non-local
    /// outcome instead of recording a call that upstream would omit.
    SwitchToBlackhole {
        reason: i32,
        raising_exception: bool,
    },
    /// `jit_merge_point/cIRFIRF` reached a loop header that was already
    /// visited with a matching red-bank shape — close the trace as a
    /// loop. `jump_args` are the merge point's red boxes (the live loop
    /// args, in `[int.., ref.., float..]` order); `loop_header_pc` is the
    /// Python pc of the merge point (decoded from the op's `next_instr`
    /// green).
    ///
    /// RPython parity: `reached_loop_header` "found"
    /// branch → `compile_trace`/close-loop. The production driver maps
    /// this to `TraceAction::CloseLoopWithArgs { jump_args, loop_header_pc }`
    /// (`lib.rs`); the retired trait-path counterpart was
    /// `close_loop_args`'s `Ok(Some(live_args))`.
    CloseLoop {
        jump_args: Vec<OpRef>,
        loop_header_pc: usize,
        /// Codewrite-time resume-marker twin at the merge point op's jitcode
        /// offset, carried into the loop-close guards' snapshot words.
        loop_header_marker_jit_pc: Option<usize>,
    },
    /// `jit_merge_point/cIRFIRF` reached a loop header that already has
    /// compiled targets, and the in-walk `compile_trace` attempt
    /// succeeded: the trace-so-far was compiled as a bridge / entry
    /// bridge ending in a JUMP into the existing loop
    /// (`reached_loop_header` →
    /// `self.compile_trace(live_arg_boxes, ptoken)`; "raises in case it
    /// works"). The tracing session was consumed by `compile_trace`;
    /// the walk must stop without compiling or aborting again. The
    /// driver maps this to `TraceAction::CompileTrace`.
    ///
    /// `loop_header_pc` is the merge point's python pc (the
    /// `jit_merge_point` `next_instr`), carried so the walk driver can
    /// flush the walk-end frame state and resume the interpreter AT the
    /// header instead of replaying the walked region
    /// (`flush_walk_end_state_to_frame`).
    CompileTracePending { loop_header_pc: usize },
    /// A multi-frame inlined callee sub-walk reached the callee's OWN
    /// `jit_merge_point` (its loop back-edge / header) while a compiled
    /// loop token for that green key already exists. Instead of declining
    /// the whole enclosing trace (the old `JitMergePointGreenKeyUnresolved`
    /// path → trait leg), the sub-walk surfaces this so the caller's
    /// `try_walker_inline_user_call` return site emits a
    /// `CALL_ASSEMBLER` into the callee loop token — the walker mirror of
    /// `opimpl_recursive_call_assembler` (`metainterp.rs`). The callee
    /// frame is the seeded virtual `PyFrame` the inline already populated
    /// via `setarrayitem_vable` during the prologue walk; the caller sets
    /// `last_instr = target_pc - 1` on it and passes it as the
    /// CALL_ASSEMBLER `[frame, ec]` red arg (forcing the virtual
    /// materializes the locals); surfaced only from an inlined sub-walk.
    SubLoopCalleeCallAssembler {
        token: std::sync::Arc<majit_backend::JitCellToken>,
        target_pc: usize,
    },
}

impl PartialEq for DispatchOutcome {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Continue, Self::Continue) => true,
            (Self::Terminate, Self::Terminate) => true,
            (Self::SubReturn { result: a }, Self::SubReturn { result: b }) => a == b,
            (
                Self::SubRaise {
                    exc: a_exc,
                    exc_concrete: a_concrete,
                },
                Self::SubRaise {
                    exc: b_exc,
                    exc_concrete: b_concrete,
                },
            ) => a_exc == b_exc && a_concrete == b_concrete,
            (
                Self::SwitchToBlackhole {
                    reason: a_reason,
                    raising_exception: a_raising,
                },
                Self::SwitchToBlackhole {
                    reason: b_reason,
                    raising_exception: b_raising,
                },
            ) => a_reason == b_reason && a_raising == b_raising,
            (
                Self::CloseLoop {
                    jump_args: a_args,
                    loop_header_pc: a_pc,
                    loop_header_marker_jit_pc: a_marker,
                },
                Self::CloseLoop {
                    jump_args: b_args,
                    loop_header_pc: b_pc,
                    loop_header_marker_jit_pc: b_marker,
                },
            ) => a_args == b_args && a_pc == b_pc && a_marker == b_marker,
            (
                Self::CompileTracePending {
                    loop_header_pc: a_pc,
                },
                Self::CompileTracePending {
                    loop_header_pc: b_pc,
                },
            ) => a_pc == b_pc,
            (
                Self::SubLoopCalleeCallAssembler {
                    token: a_token,
                    target_pc: a_pc,
                },
                Self::SubLoopCalleeCallAssembler {
                    token: b_token,
                    target_pc: b_pc,
                },
            ) => a_token.number == b_token.number && a_pc == b_pc,
            _ => false,
        }
    }
}

fn trace_too_long_blackhole_snapshot_safe(outcome: &DispatchOutcome) -> bool {
    matches!(outcome, DispatchOutcome::Continue)
}

fn trace_too_long_abort_safe(
    outcome: &DispatchOutcome,
    blackhole_latched: bool,
    executed_effects: usize,
) -> bool {
    trace_too_long_blackhole_snapshot_safe(outcome) && (blackhole_latched || executed_effects == 0)
}

/// Errors surfaced by the trace-side walker.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DispatchError {
    /// The opcode byte at `pc` is not present in the `insns` table or
    /// the instruction's operand bytes overflowed the code slice. This
    /// is the same `decode_op_at -> None` path surfaced as a typed
    /// error.
    UndecodableOpcode { pc: usize },
    /// The opcode is decodable but the dispatcher has no handler for
    /// it yet. Carries the `opname/argcodes` key so callers can
    /// identify what blocked the walk.
    UnsupportedOpname { pc: usize, key: &'static str },
    /// A register operand byte indexed past the symbolic register file.
    /// `len` is the slice length the walker was handed in the
    /// `WalkContext`; `reg` is the byte the codewriter emitted. Surfaces
    /// either an assembler-pass bug (out-of-range register) or a
    /// caller mismatch between the symbolic register layout and the
    /// arm's expected number of registers.
    RegisterOutOfRange {
        pc: usize,
        reg: usize,
        len: usize,
        bank: &'static str,
    },
    /// A `d`-coded descr index resolved past the descr pool. Surfaces
    /// either an assembler-pass bug (descr index out of range) or a
    /// caller mismatch between the codewriter's descr table size and
    /// the table the walker was handed in `WalkContext::descr_refs`.
    DescrIndexOutOfRange { pc: usize, index: usize, len: usize },
    /// `inline_call_*` resolved a descr that does not implement
    /// `JitCodeDescr`. Surfaces either a codewriter bug (an
    /// `inline_call_*` opnum emitted with a non-jitcode descr index)
    /// or a caller mismatch (the descr pool wasn't built from the
    /// codewriter's descr table). `descr_index` is the 2-byte LE
    /// index the walker decoded.
    ExpectedJitCodeDescr { pc: usize, descr_index: usize },
    /// `inline_call_*`'s descr resolved to a `jitcode_index`, but the
    /// caller's `sub_jitcode_lookup` returned `None`. Production wires
    /// the lookup to `crate::jitcode_runtime::all_jitcodes()`; tests
    /// build synthetic maps. A `None` return means the codewriter
    /// emitted an index past the runtime's jitcode table.
    SubJitCodeNotFound { pc: usize, jitcode_index: usize },
    /// `inline_call_*` provided more Ref args in its R-list than the
    /// callee declared `num_regs_r` slots. RPython parity: `pyjitpl.py
    /// MIFrame.setup_call(argboxes)` distributes argboxes into the
    /// callee's typed register banks; the JitCode-level shape contract
    /// (`assembler.py:write_call`) requires `len(argboxes) <=
    /// num_regs_r` for the `_r_r` variant. Excess args are a
    /// codewriter-emitted shape mismatch.
    InlineCallArityMismatch {
        pc: usize,
        provided: usize,
        callee_num_regs_r: usize,
    },
    /// `inline_call_*` provided more Int args in its I-list than the
    /// callee declared `num_regs_i` slots. Same shape contract as the
    /// Ref variant — `pyjitpl.py setup_call` populates each
    /// kind-bank from its respective list and asserts capacity.
    InlineCallIntArityMismatch {
        pc: usize,
        provided: usize,
        callee_num_regs_i: usize,
    },
    /// `inline_call_*` provided more Float args in its F-list than the
    /// callee declared `num_regs_f` slots.
    InlineCallFloatArityMismatch {
        pc: usize,
        provided: usize,
        callee_num_regs_f: usize,
    },
    /// `inline_call_r_r/dR>r`'s callee surfaced
    /// `SubReturn { result: None }`. RPython parity: the `_r_r` variant
    /// is wired (in `assembler.py:gen_inline_call`) to a callee whose
    /// `*_return` op carries a Ref; reaching it without a result means
    /// the callee body executed `void_return/` (or an analogue) instead
    /// of `ref_return/r`, which is a codewriter shape mismatch — the
    /// caller has nowhere to land the missing value.
    UnexpectedVoidSubReturn { pc: usize },
    /// `inline_call_*_v/d{R,IR,IRF}`'s callee surfaced
    /// `SubReturn { result: Some(_) }`. RPython parity: the `_v` variant
    /// (`bhimpl_inline_call_*_v`, `blackhole.py`) is
    /// wired to a callee whose `*_return` op is `void_return/`; reaching
    /// it with a typed-return result means the callee body executed
    /// `int_return/i` / `ref_return/r` / `float_return/f` instead of
    /// `void_return/`, which is a codewriter shape mismatch — the
    /// caller has no `>X` slot to land the surplus value.
    UnexpectedNonVoidSubReturn { pc: usize },
    /// `reraise/` fired but `WalkContext::last_exc_value` was `None`.
    /// RPython parity: `pyjitpl.py
    /// opimpl_reraise: assert self.metainterp.last_exc_value` —
    /// reaching `reraise` without an active exception is a codewriter
    /// invariant violation (`raise` or a catch-handler entry must have
    /// set `last_exc_value` first).
    ReraiseWithoutLastExcValue { pc: usize },
    /// `last_exc_value/>r` fired but `WalkContext::last_exc_value` was
    /// `None`. RPython parity: `pyjitpl.py opimpl_last_exc_value`:
    ///
    ///   exc_value = self.metainterp.last_exc_value
    ///   assert exc_value
    ///   return self.metainterp.last_exc_box
    ///
    /// Same codewriter invariant as `reraise/`: this opname only emits
    /// inside a `catch_exception` body where the unwinder has already
    /// stored the in-flight exception. Reaching it without an active
    /// exception is a flatten/codewriter shape mismatch.
    LastExcValueWithoutActiveException { pc: usize },
    /// `catch_exception/L` was reached on the normal fall-through path
    /// (no `SubRaise` routing) but `WalkContext::last_exc_value` was
    /// non-`None`. RPython parity: `pyjitpl.py opimpl_catch_exception`:
    ///
    ///   assert not self.metainterp.last_exc_value
    ///
    /// On the normal path the previous instruction did NOT raise — if
    /// it had, `finishframe_exception` would have routed control past
    /// the catch_exception/L (or to its target if matched), never
    /// running the catch_exception/L instruction itself. Reaching it
    /// with an active exception means the codewriter mis-emitted a
    /// catch_exception/L outside an exception-table position, OR a
    /// previous handler forgot to clear `last_exc_value` after handling.
    CatchExceptionWithActiveException { pc: usize },

    /// `residual_call_*` decoded a descr that does not implement
    /// `CallDescr`. RPython parity: `pyjitpl.py
    /// do_residual_call` always receives a `calldescr` from the
    /// codewriter — there is no fallback path. The walker mirrors that
    /// invariant by surfacing a typed error when the descr_pool entry
    /// at the operand-encoded index lacks a CallDescr downcast. In
    /// production the codewriter never emits a non-CallDescr; this
    /// variant fires only when test fixtures (or future deviations)
    /// route a non-CallDescr into a residual_call slot.
    ResidualCallDescrNotCallDescr { pc: usize, descr_index: usize },
    /// A residual_call argument resolved to `OpRef::NONE` — an unbound
    /// register.  RPython's `do_residual_call` reads every argbox out of
    /// `env[box]`, so a missing binding is a `KeyError` there; the
    /// cranelift/dynasm backends mirror that as a hard `resolve_opref`
    /// assert when `OpRef::NONE` reaches arg-resolution.  A per-opcode
    /// arm whose body forwards an unseeded dispatcher parameter (e.g. a
    /// seam wrapper that residualizes `execute_<op>(executor, code,
    /// instruction, op_arg)` — only `executor`=r0 is seeded at the arm
    /// entry, so `code`/`instruction`/`op_arg` stay `OpRef::NONE`) would
    /// record such a call.  Surfacing it here turns the would-be backend
    /// crash into a graceful trace abort, matching the pre-seam inline
    /// arm whose payload read aborted with `GotoIfNotValueNotConcrete`.
    ResidualCallArgUnbound { pc: usize, arg_index: usize },
    /// A `getfield_gc_*` cache-miss would record an op whose `FieldDescr`
    /// carries no `get_parent_descr()` backreference.  The optimizer's
    /// `ensure_ptr_info_arg0` (`optimizer.py`) panics rather than
    /// install a malformed PtrInfo for such a descr, so — like
    /// `ResidualCallArgUnbound` — surface it here to turn the would-be
    /// optimizer crash into a graceful trace abort.  Production sub-walks
    /// never reach this (they would already panic); it fires only on the
    /// orthodox `w_list_append` descent over a descr (e.g.
    /// `W_ListObject.strategy`) not yet resolved to its parent descr group.
    FieldDescrMissingParentDescr { pc: usize },
    /// The orthodox `w_list_append` body sub-walk descended the full append
    /// (past the strategy switch + `is_plain_int1`, descr-pool resolved) and
    /// the trace compiles, but the guards it records resume through side-exit
    /// bridges whose reconstruction is wrong — executing the compiled loop
    /// jumps to garbage.  This is the sub-walk guard resume/faillocs epic
    /// (#62/#73/#34) the hand-rolled fold sidesteps with explicit
    /// `walker_capture_snapshot_for_last_guard` snapshots.  Surfaced here to
    /// abort the trace gracefully (interpreter fallback) instead of committing
    /// a trace that SIGSEGVs on its first side exit.  Default OFF.
    OrthodoxSubWalkTraceUnsupported { pc: usize },
    /// The LIST_APPEND opcode's void `jit_list_append` residual
    /// (`ListAppendValue`) reached the authoritative full-body walker but the
    /// orthodox `w_list_append` fold declined (the list needs a resize, or the
    /// strategy/value is unsupported — `orthodox_list_append_recognize`
    /// returned `None`) or the fold is disabled.  Unlike the fold's
    /// `orthodox_list_append_commit`, the generic residual dispatcher
    /// concrete-executes `jit_list_append` WITHOUT `fbw_append_journal_push`,
    /// so `fbw_store_journal_rollback` cannot rewind it; a later trace abort +
    /// interpreter replay would then apply the SAME append twice. Decline the
    /// trace to interpretation instead of falling through, mirroring the
    /// pre-#171 `emit_abort_permanent` LIST_APPEND lowering — the common
    /// int/float/object append still folds; only the decline shapes abort, and
    /// they aborted before #171 too.
    UnfoldableListAppendResidualUnsupported { pc: usize },
    /// `switch/id` decoded a descr that does not implement
    /// `SwitchDescr`. RPython parity: `pyjitpl.py` asserts
    /// `isinstance(switchdict, SwitchDictDescr)`.
    ExpectedSwitchDescr { pc: usize },
    /// `switch/id` needs RPython's `valuebox.getint()` at trace time.
    /// The symbolic walker can obtain that only when `TraceCtx` can
    /// reconstruct an Int concrete for the OpRef today;
    /// choosing a branch without a concrete value would record the wrong
    /// guard chain, so surface the missing concrete value explicitly.
    SwitchValueNotConcrete { pc: usize, value: OpRef },
    /// `goto_if_not/iL` needs RPython's `box.getint()` at trace time
    /// (`pyjitpl.py opimpl_goto_if_not`).  Without the
    /// concrete value the walker can't pick GUARD_TRUE vs GUARD_FALSE
    /// or decide whether to jump to the label target, so surface the
    /// missing concrete explicitly instead of guessing.
    GotoIfNotValueNotConcrete { pc: usize, value: OpRef },
    /// An overflow-checking integer jump needs both operands' runtime values
    /// to choose the overflow arm. Decline when either value is unavailable
    /// instead of crashing the tracer.
    IntOvfOperandNotConcrete { pc: usize, value: OpRef },
    /// `OS_NOT_IN_TRACE` must run the callee concretely and record no
    /// IR on the normal path (`pyjitpl.py`). The standalone
    /// symbolic walker has no concrete executor, so it must stop here
    /// instead of faking either the normal return or
    /// `SwitchToBlackhole`.
    NotInTraceRequiresConcreteExecution { pc: usize },
    /// `OS_JIT_FORCE_VIRTUAL` would short-circuit `do_residual_call`
    /// before recording `CALL_MAY_FORCE_*` (`pyjitpl.py →
    /// 2153-2172 _do_jit_force_virtual`). The short-circuit needs a
    /// concrete `vref_ptr` for arbitrary Ref OpRefs to determine
    /// `isstandard_int` at trace time — walker only knows
    /// `concrete_vable_ptr`, not the concrete value behind every Ref
    /// OpRef. Surfacing this as an error prevents silently recording
    /// `CALL_MAY_FORCE_*` for an op the live tracer would have folded
    /// to `vref_box` / `standard_box` / None. Production reach today:
    /// `OopSpecIndex::JitForceVirtual` is set only by
    /// `jtransform.rs jit.force_virtual` lowering, which our
    /// benchmarks don't trigger; this guard is fail-loud against future
    /// silent TODOs.
    JitForceVirtualRequiresConcreteResolver { pc: usize },
    /// `{get,set}field_vable_{i,r,f}` read its box operand from a Ref
    /// register holding `OpRef::None`. RPython parity: `pyjitpl.py
    /// opimpl_getfield_vable_{i,r,f}` / `_opimpl_setfield_vable(box, ...,
    /// pc)` always receive a live `box` from the register file — the
    /// virtualizable frame box. Pyre's walker initializes Ref register
    /// slots to `OpRef::None`; an inlined callee frame may leave the vable
    /// register unseeded (documented walker arg-seeding gap). Both vable
    /// accessors route through `is_nonstandard_virtualizable` →
    /// `heapcache.nonstandard_virtualizables_now_known(box)`, which would
    /// feed `OpRef::None` (`raw() == u32::MAX`) into the dense heapcache
    /// flag `Vec<u32>`, resizing it to `u32::MAX + 1` (16 GiB). Surface as
    /// a trace abort so production resumes in the interpreter rather than
    /// allocating; no compiled trace is produced.
    VableBoxNotSeeded { pc: usize },
    /// `{get,set}arrayitem_vable_*` / `arraylen_vable` decoded its
    /// `(VableArray, Array)` descr-pool pair but one of the two slots
    /// was missing or held the wrong `BhDescr` variant. RPython parity:
    /// `MIFrame.vable_array_index_pair_at` (`blackhole.rs`)
    /// asserts the exact `(VableArray, Array)` pair and panics otherwise
    /// — a malformed pair is a flatten/codewriter shape bug. The walker
    /// surfaces it as a fail-loud abort instead of crashing the tracer.
    VableArrayDescrMalformed {
        pc: usize,
        field_idx: usize,
        array_idx: usize,
    },
    /// `{get,set}arrayitem_vable_*` / `arraylen_vable` reached the vable
    /// resolution path but `TraceCtx::virtualizable_info()` was `None`.
    /// RPython parity: these opnames only emit for the jitdriver's
    /// virtualizable (`pyjitpl.py`); a missing
    /// `virtualizable_info` means the descr pair pointed at an array
    /// field of a struct that is not the registered virtualizable, which
    /// is a codewriter shape mismatch.
    VableArrayMissingVirtualizableInfo { pc: usize },
    /// The `VableArray.index` decoded from the descr pair indexed past
    /// `VirtualizableInfo::array_field_descrs` / `array_descrs`. Surfaces
    /// a codewriter/virtualizable-layout mismatch between the emitted
    /// array-field index and the registered virtualizable's array fields.
    VableArrayIndexOutOfRange { pc: usize, index: usize },
    /// `{get,set}arrayitem_vable_*` needs RPython's `indexbox.getint()`
    /// at trace time (`pyjitpl.py _get_arrayitem_vable_index`
    /// calls `implement_guard_value(indexbox, pc)` then `indexbox.getint()`).
    /// The symbolic walker can resolve that only when the index OpRef has
    /// a concrete Int; without it the array slot can't be chosen, so
    /// surface the missing concrete explicitly instead of guessing slot 0.
    VableArrayIndexNotConcrete { pc: usize, value: OpRef },
    /// `abort/` (BC_ABORT) reached. The front-end emits this marker for a
    /// graph node with no dedicated `OpKind`; reaching it during a body
    /// walk means the trace contains an untranslatable op and cannot be
    /// recorded. Blackhole counterpart `handler_abort_marker_pyre`
    /// (`blackhole.rs`) sets `aborted = true` + `LeaveFrame`; the
    /// walker surfaces it as a typed abort that the production driver
    /// maps to `TraceAction::Abort` (recoverable — may retry later).
    AbortMarkerReached { pc: usize },
    /// Record-time construction of a concrete shadow object failed after the
    /// specialization had emitted IR. Falling through to the generic residual
    /// would retain that partial prologue, so abort the whole walk.
    ConcreteShadowAllocationFailed { pc: usize },
    /// `abort_permanent/` (BC_ABORT_PERMANENT) reached. pyre's codegen
    /// emits this for fail-paths that must always terminate the frame
    /// (e.g. BigInt-overflow / unported-op fallbacks). Blackhole
    /// counterpart `bhimpl_abort_permanent` (`blackhole.rs`) routes a
    /// TLS-stashed exception or sets `aborted`; during a trace walk no
    /// blackhole TLS exists, so the walker surfaces a typed permanent
    /// abort that the production driver maps to `TraceAction::AbortPermanent`
    /// (never trace this location again).
    AbortPermanentMarkerReached { pc: usize },
    /// A result-bearing may-force CALL (`CallMayForce{R,I,F}`, a Python
    /// function-entry call) was recorded with a concrete-NULL Ref
    /// argument. This is the specialized direct-call shape: the callee
    /// was folded to its entry address and the `PUSH_NULL` self-slot is
    /// baked as `ptr(0x0)`. `try_execute_residual_call_via_executor` skips
    /// such a call (its NULL-receiver SEGV guard), so the result stays
    /// unresolved, and the baked NULL arg makes the compiled call pass
    /// NULL where the entry needs the callee's globals/closure — yielding
    /// a NULL result at runtime (closures / functions bound to a local
    /// and called in a loop). The walker surfaces a typed abort so the
    /// driver maps it to `TraceAction::Abort` and the loop falls back to
    /// the interpreter instead of compiling a wrong trace.
    MayForceNullRefArgUnsupported { pc: usize },
    /// The virtualizable escaped during a concrete-executed may-force
    /// residual call: `vinfo.tracing_after_residual_call(virtualizable)`
    /// found the token cleared by the callee's force path
    /// (pyjitpl.py `vable_after_residual_call` →
    /// `SwitchToBlackhole(Counters.ABORT_ESCAPE, raising_exception=True)`).
    /// The trace can no longer treat the frame as a virtualizable, so the
    /// walk aborts and the interpreter resumes from the (now heap-
    /// authoritative) frame. Soft abort — escape is data-dependent, the
    /// same location may trace cleanly later.
    VableEscapedDuringResidualCall { pc: usize },
    /// A walker-emitted guard needs a resume snapshot, but the live
    /// virtualizable box list carries an untyped entry (typically the
    /// identity box `[-1]` of a deeper inlined / recursive frame). Building
    /// the snapshot would trip the `build_vable_snapshot_boxes`
    /// `OpRef::ty()` invariant. The full-body walk surfaces a typed abort so
    /// the driver maps it to `TraceAction::Abort` and interpretation instead
    /// of panicking the tracer. Resuming such a guard needs the multi-frame
    /// vable snapshot machinery.
    GuardSnapshotVableUntyped { pc: usize },
    /// A guard capture had no decodable carried JitCode coordinate. Publishing
    /// a Python pc here would make a later side exit resume at a guessed
    /// block head, so the walker declines before the trace is installed.
    GuardResumeCoordinateUnavailable { pc: usize },
    /// `last_exception/>i` fired but no concrete standing exception was
    /// available. RPython parity: `pyjitpl.py opimpl_last_exception`:
    ///
    ///   exc_value = self.metainterp.last_exc_value
    ///   assert exc_value
    ///   assert self.metainterp.class_of_last_exc_is_const
    ///
    /// The class pointer is read from the concrete exception's
    /// `ob_header.ob_type`, so a missing `last_exc_value` OpRef or a
    /// `Null`/non-Ref `last_exc_value_concrete` violates the same
    /// codewriter invariant as [`LastExcValueWithoutActiveException`]:
    /// this opname only emits inside a `catch_exception` body where the
    /// unwinder has already stored the in-flight exception.
    LastExceptionWithoutActiveException { pc: usize },
    /// `jit_merge_point/cIRFIRF` could not derive its green key from the
    /// op's green operands. pyre's portal jitdriver greens =
    /// `[next_instr, is_being_profiled, pycode]` (`eval.rs`), so the
    /// green key is `make_green_key(concrete(gr[0]=pycode),
    /// concrete(gi[0]=next_instr))`. This fires when the int/ref green
    /// list is empty or its leading element has no concrete Int/Ref —
    /// either a codewriter shape mismatch or (pre-Phase-5) the greens
    /// weren't seeded with concretes. RPython parity:
    /// `pyjitpl.py get_procedure_token(greenboxes)` always receives
    /// concrete greens at a reached merge point.
    JitMergePointGreenKeyUnresolved { pc: usize },
    /// `loop_header/i` or `jit_merge_point/iIRFIRF` could not resolve its
    /// jdindex operand to a concrete Int. The assembler encodes the jdindex as a
    /// populated int-constant-pool slot (`assembler.rs loop_header`:
    /// `add_const_i` + patch at `finish()`), so an unresolved slot is a
    /// structural encoding bug, mirroring the `expect` on
    /// `frame.int_values[slot]` in majit's `BC_LOOP_HEADER` arm
    /// (`pyjitpl/dispatch.rs`).
    LoopHeaderJdIndexUnresolved { pc: usize },
    /// An `inline_call_*` sub-walk surfaced `DispatchOutcome::CloseLoop`.
    /// `jit_merge_point` is the portal loop header; an inlined (non-portal)
    /// callee body should never reach one and close a loop. RPython parity:
    /// `_opimpl_inline_call*` (`pyjitpl.py`) inlines the callee
    /// into the SAME trace — recursive portal re-entry takes the
    /// `recursive_call` path, not `inline_call`. Reaching a loop-close
    /// inside an inline sub-walk is therefore a codewriter/flatten shape
    /// mismatch.
    SubWalkClosedLoop { pc: usize },
    /// A `goto_if_not` branch guard resumes at a target that still carries
    /// two or more live operand-stack temps (resume-target stack depth > 1)
    /// and the not-taken edge's kept-value recovery is incomplete.
    /// This is the multi-kept-temp short-circuit shape — `(x and y) or z`,
    /// chained comparison `a < b < c` — where CPython keeps more than one
    /// tested value on the value stack across the branch (`COPY` / `TO_BOOL`
    /// / `POP_JUMP_IF_*`). The full-body walk's single-frame guard snapshot
    /// rebuilds locals + the post-opcode operand stack from the live register
    /// banks; a single depth-1 kept temp is recovered from the walk-level box
    /// mirror (`vstack`) in `collect_outer_active_boxes`.  Depth > 1 is
    /// supported only when the not-taken edge's decoded `ref_copy` moves
    /// (`#420`) cover every distinct kept resume color AND each move's source
    /// resolves to a live register value; that resolved set then drives the
    /// snapshot encoder.  This abort fires only when that recovery is
    /// incomplete — a kept slot the edge does not rename ("live-across"), a
    /// cyclic `*_push`/`*_pop` move, a truncated/under-sized color map, or a
    /// source that resolves to `OpRef::NONE` — i.e. when the deopt re-entry
    /// would otherwise restore a wrong value into a loop-carried slot
    /// (per-PC resume-value precision).  Plain `while` / `if`
    /// branches resume at depth 0 and are unaffected.  The walker surfaces a
    /// typed abort so the driver maps it to `TraceAction::Abort` →
    /// interpreter fallback (correct, untraced) instead of compiling a trace
    /// whose guard-failure path corrupts the frame.
    BranchGuardKeptStackUnsupported { pc: usize },
    /// A kept-stack branch guard's not-taken (resume) arm READS a regular
    /// Ref register (`< num_regs_r`) that is neither snapshot-live nor
    /// produced inside the arm, so the blackhole resumes it as NULL and
    /// feeds NULL into the consuming op — the boxed-int short-circuit /
    /// conditional-expression resume miscompile (a heap `ConstPtr`, an int
    /// outside the 1-byte immediate range `[0, 256)`, parked in a register
    /// live-across the guard).  The driver maps this to
    /// `TraceAction::AbortPermanent` → `DONT_TRACE_HERE`.  Demoting it to a
    /// plain [`TraceAction::Abort`] would still reach the same terminal state
    /// through pyre's abort ceiling, so the permanent mapping records the
    /// structural nature of this abort without changing runtime behavior.
    BranchGuardUnrestorableKeptStackPermanent { pc: usize },
    /// A callee compiled as its own Finish portal (reached via
    /// `call_user_function_with_eval`) accessed its frame through a
    /// `vable_*` op that found it to be a non-standard virtualizable,
    /// emitting an internal promote `GuardValue` + force store-back. The
    /// Finish-portal compile path does not yet wire a resume snapshot or a
    /// `FieldDescr` for those internal ops (only the inline sub-walk path
    /// does), so compiling the trace trips the optimizer's
    /// `store_final_boxes_in_guard` / `optimize_setfield_gc` invariants.
    /// Abort to the interpreter; the method runs interpreted until the
    /// own-portal callee frame is registered as the standard virtualizable
    /// (a perf follow-up).
    NonStandardVableFinishPortalUnsupported { pc: usize },
    /// A residual call to a pure-Python callee that is inline-eligible
    /// (plain, exact-positional, closure-free, not recursion-bound) but
    /// whose body is NOT a straight-line leaf — it carries an internal loop
    /// or branch (`goto_if_not` / `switch`) or a non-static `vable` op, so
    /// the fast-path register-seeding inline (`try_walker_inline_user_call`)
    /// declines.  Emitting the residual leaves the callee re-interpreted per
    /// iteration (and its short inner loops compile + deopt-storm), strictly
    /// slower than interpreting. The retired MIFrame tracer inlined such
    /// callees via `push_inline_frame` + the `recursive-call-assembler` loop back-edge
    /// (`pyjitpl.rs` opimpl_recursive_call_assembler).  Surface a typed
    /// abort so the key interprets without JIT (`FBW_DECLINED_KEYS`) until
    /// the walker itself covers loop-callee inlining.
    LoopBearingCalleeInlineUnsupported { pc: usize },
    /// An in-flight FOR_ITER body executed a non-journalable
    /// in-place builtin-container mutation (`acc += delta` for an object-/float-
    /// strategy list, `bytearray`, `set`, `dict`, …).  The abort rollback cannot
    /// rewind it and a deliver re-run would double it, so the walk declines BEFORE
    /// the commit and the location interprets permanently (`AbortPermanent`).
    InplaceContainerMutationUnsupported { pc: usize },
    /// Exception-edge bridge: `call_jit` routed the exc edge (published the
    /// standing exception and declined to hand the guard to the blackhole), but
    /// this frame carries no `catch_exception` covering the raise — a state the
    /// routing precondition is supposed to exclude, since a routed walk must
    /// resume at a handler rather than fall through to the NULL raised-call
    /// result.  Abort so the guard failure resumes via the blackhole instead.
    /// Where the handler LEADS does not reach this error: a handler that returns
    /// out of the frame routes like any other (`pyjitpl.py:2530-2546`
    /// `finishframe_exception` jumps to the catch unconditionally, and the
    /// return is `finishframe`'s ordinary case, `pyjitpl.py:2503-2525`).
    ExcEdgeNoInFrameCatch { pc: usize },
    /// The recorded trace passed `warmstate.trace_limit` (`rlib/jit.py:592`
    /// default 6000). RPython checks this after every traced step —
    /// `pyjitpl.py:2865 _interpret` calls `blackhole_if_trace_too_long()` right
    /// after `run_one_step()` — and raises `SwitchToBlackhole(ABORT_TOO_LONG)`.
    TraceTooLong { pc: usize, ops: usize },
}

impl DispatchError {
    /// The variant's identifier as a `'static` string, for the
    /// decline-class census (see [`census_record`]).  One arm per
    /// variant so a new variant fails to compile until it is named here.
    pub(crate) fn variant_name(&self) -> &'static str {
        match self {
            Self::UndecodableOpcode { .. } => "UndecodableOpcode",
            Self::UnsupportedOpname { .. } => "UnsupportedOpname",
            Self::RegisterOutOfRange { .. } => "RegisterOutOfRange",
            Self::DescrIndexOutOfRange { .. } => "DescrIndexOutOfRange",
            Self::ExpectedJitCodeDescr { .. } => "ExpectedJitCodeDescr",
            Self::SubJitCodeNotFound { .. } => "SubJitCodeNotFound",
            Self::InlineCallArityMismatch { .. } => "InlineCallArityMismatch",
            Self::InlineCallIntArityMismatch { .. } => "InlineCallIntArityMismatch",
            Self::InlineCallFloatArityMismatch { .. } => "InlineCallFloatArityMismatch",
            Self::UnexpectedVoidSubReturn { .. } => "UnexpectedVoidSubReturn",
            Self::UnexpectedNonVoidSubReturn { .. } => "UnexpectedNonVoidSubReturn",
            Self::ReraiseWithoutLastExcValue { .. } => "ReraiseWithoutLastExcValue",
            Self::LastExcValueWithoutActiveException { .. } => "LastExcValueWithoutActiveException",
            Self::CatchExceptionWithActiveException { .. } => "CatchExceptionWithActiveException",
            Self::ResidualCallDescrNotCallDescr { .. } => "ResidualCallDescrNotCallDescr",
            Self::ResidualCallArgUnbound { .. } => "ResidualCallArgUnbound",
            Self::ExpectedSwitchDescr { .. } => "ExpectedSwitchDescr",
            Self::SwitchValueNotConcrete { .. } => "SwitchValueNotConcrete",
            Self::GotoIfNotValueNotConcrete { .. } => "GotoIfNotValueNotConcrete",
            Self::IntOvfOperandNotConcrete { .. } => "IntOvfOperandNotConcrete",
            Self::NotInTraceRequiresConcreteExecution { .. } => {
                "NotInTraceRequiresConcreteExecution"
            }
            Self::JitForceVirtualRequiresConcreteResolver { .. } => {
                "JitForceVirtualRequiresConcreteResolver"
            }
            Self::VableBoxNotSeeded { .. } => "VableBoxNotSeeded",
            Self::VableArrayDescrMalformed { .. } => "VableArrayDescrMalformed",
            Self::VableArrayMissingVirtualizableInfo { .. } => "VableArrayMissingVirtualizableInfo",
            Self::VableArrayIndexOutOfRange { .. } => "VableArrayIndexOutOfRange",
            Self::VableArrayIndexNotConcrete { .. } => "VableArrayIndexNotConcrete",
            Self::AbortMarkerReached { .. } => "AbortMarkerReached",
            Self::ConcreteShadowAllocationFailed { .. } => "ConcreteShadowAllocationFailed",
            Self::AbortPermanentMarkerReached { .. } => "AbortPermanentMarkerReached",
            Self::MayForceNullRefArgUnsupported { .. } => "MayForceNullRefArgUnsupported",
            Self::VableEscapedDuringResidualCall { .. } => "VableEscapedDuringResidualCall",
            Self::GuardSnapshotVableUntyped { .. } => "GuardSnapshotVableUntyped",
            Self::GuardResumeCoordinateUnavailable { .. } => "GuardResumeCoordinateUnavailable",
            Self::LastExceptionWithoutActiveException { .. } => {
                "LastExceptionWithoutActiveException"
            }
            Self::JitMergePointGreenKeyUnresolved { .. } => "JitMergePointGreenKeyUnresolved",
            Self::LoopHeaderJdIndexUnresolved { .. } => "LoopHeaderJdIndexUnresolved",
            Self::SubWalkClosedLoop { .. } => "SubWalkClosedLoop",
            Self::BranchGuardKeptStackUnsupported { .. } => "BranchGuardKeptStackUnsupported",
            Self::NonStandardVableFinishPortalUnsupported { .. } => {
                "NonStandardVableFinishPortalUnsupported"
            }
            Self::LoopBearingCalleeInlineUnsupported { .. } => "LoopBearingCalleeInlineUnsupported",
            Self::FieldDescrMissingParentDescr { .. } => "FieldDescrMissingParentDescr",
            Self::OrthodoxSubWalkTraceUnsupported { .. } => "OrthodoxSubWalkTraceUnsupported",
            Self::UnfoldableListAppendResidualUnsupported { .. } => {
                "UnfoldableListAppendResidualUnsupported"
            }
            Self::BranchGuardUnrestorableKeptStackPermanent { .. } => {
                "BranchGuardUnrestorableKeptStackPermanent"
            }
            Self::InplaceContainerMutationUnsupported { .. } => {
                "InplaceContainerMutationUnsupported"
            }
            Self::ExcEdgeNoInFrameCatch { .. } => "ExcEdgeNoInFrameCatch",
            Self::TraceTooLong { .. } => "TraceTooLong",
        }
    }

    /// Construct the callee-inline decline.  The
    /// `LoopBearingCalleeInlineUnsupported` variant is emitted from ~20 sites
    /// (multi-frame seed preconditions, snapshot capture, hazard scan), so
    /// route them through one constructor that records the source location
    /// under `PYRE_LB_SITE` to tell the decline reasons apart in a census.
    #[track_caller]
    #[inline]
    pub(crate) fn callee_inline_unsupported(pc: usize) -> Self {
        if std::env::var_os("PYRE_LB_SITE").is_some() {
            let loc = std::panic::Location::caller();
            eprintln!("[lb-site] {}:{} pc={pc}", loc.file(), loc.line());
        }
        Self::LoopBearingCalleeInlineUnsupported { pc }
    }
}

/// Per-process census of full-body-walk decline classes, keyed by
/// [`DispatchError::variant_name`] (or a synthetic key for the non-`Err`
/// abort outcomes).  Populated only on the cold abort path via
/// [`census_record`]; off the hot trace path entirely.  The dump
/// (`census_dump`) is gated on [`fbw_debug_abort_enabled`], so with the
/// `PYRE_FBW_DEBUG_ABORT` flag unset the only cost is one `BTreeMap`
/// insert per declined walk (already a rare, slow event).
thread_local! {
    static FBW_DECLINE_CENSUS: std::cell::RefCell<std::collections::BTreeMap<&'static str, usize>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

/// Bump the decline count for `name` (a [`DispatchError::variant_name`] or
/// a synthetic outcome key).  Called from the trace-side decline routing.
/// When [`fbw_debug_abort_enabled`] is set, also emits the running census
/// so a corpus run prints `[fbw-census]` lines (the last block printed is
/// the final tally) without needing a process-exit hook.
pub(crate) fn census_record(name: &'static str) {
    FBW_DECLINE_CENSUS.with(|c| {
        *c.borrow_mut().entry(name).or_insert(0) += 1;
    });
    census_dump();
}

thread_local! {
    /// Code objects already counted by
    /// [`census_record_frame_shape_decline`], so a `CurrentFrameOnly` code
    /// entered many times (a hot helper) records exactly one census entry
    /// instead of one per call.  Keyed by the stable `CodeObject` pointer.
    static FRAME_SHAPE_DECLINE_SEEN: std::cell::RefCell<std::collections::BTreeSet<usize>> =
        const { std::cell::RefCell::new(std::collections::BTreeSet::new()) };
}

/// Record — once per distinct code object — the frame-shape decline of a frame
/// that `unsupported_jit_shape` keeps out of the tracer entirely: a
/// `CurrentFrameOnly` FOR_ITER frame (a body with a non-journalable mutator —
/// the #57 gate — or a `finally`-duplicated loop), a `NestedBreakBridgeResume`
/// frame, or a frame whose constant pool overruns the single-byte index
/// encoding.  The tracer never runs for such a frame, so
/// its decline would otherwise leave no census entry and read as a silent
/// no-token gap; recording it here lets the `PYRE_FBW_DEBUG_ABORT` corpus
/// attribute the pre-trace frame-shape decline alongside the traced declines.
/// `code_ptr` is the `CodeObject` pointer, used only to dedup repeated entries
/// of the same declined frame; `kind` names the shape for the census line.
pub fn census_record_frame_shape_decline(code_ptr: usize, kind: &'static str) {
    let first = FRAME_SHAPE_DECLINE_SEEN.with(|s| s.borrow_mut().insert(code_ptr));
    if first {
        census_record(kind);
    }
}

/// Print the accumulated decline census as `[fbw-census] <name>: <count>`
/// lines, sorted by name.  No-op unless [`fbw_debug_abort_enabled`].
/// Safe to call repeatedly (a diagnostic dump, not a reset).
pub(crate) fn census_dump() {
    if !fbw_debug_abort_enabled() {
        return;
    }
    FBW_DECLINE_CENSUS.with(|c| {
        let map = c.borrow();
        if map.is_empty() {
            eprintln!("[fbw-census] (no declines recorded)");
            return;
        }
        for (name, count) in map.iter() {
            eprintln!("[fbw-census] {name}: {count}");
        }
    });
}

/// Carrier-boundary raise seed (`finishframe_exception` at the bridge carrier):
/// set by [`crate::trace::drive_bridge_carrier_walk`] when a depth-2 inlined
/// callee's sub-walk ended in `SubRaise`.
/// [`crate::jitcode_dispatch::dispatch_via_miframe`] reads it once when it sets
/// up the root walk.  With `catch_target` set the root frame's `except` handler
/// covers the CALL and the walk enters at that handler with the caught
/// exception seeded — the same handler-entry reconstruction the walk-level
/// SubRaise routing performs, but at the carrier boundary the sub-walk crossed
/// on its own.  `None` means the root frame has no covering handler: the
/// framestack this trace models is exhausted, so the walk ends immediately with
/// `compile_exit_frame_with_exception` and the interpreter unwinds the
/// remaining Python frames.
#[derive(Clone, Copy)]
pub(crate) struct CarrierRaiseSeed {
    pub exc: OpRef,
    pub exc_concrete: crate::state::ConcreteValue,
    pub catch_target: Option<usize>,
}

thread_local! {
    static FBW_CARRIER_RAISE_SEED: std::cell::Cell<Option<CarrierRaiseSeed>> =
        const { std::cell::Cell::new(None) };
}

pub(crate) fn set_carrier_raise_seed(seed: CarrierRaiseSeed) {
    FBW_CARRIER_RAISE_SEED.with(|c| c.set(Some(seed)));
}

pub(crate) fn take_carrier_raise_seed() -> Option<CarrierRaiseSeed> {
    FBW_CARRIER_RAISE_SEED.with(|c| c.take())
}

/// Walk one opcode at `pc` and return the dispatch outcome plus the
/// next pc. Side effects reach `ctx.trace_ctx` only for opnames whose
/// handler explicitly records (e.g. `ref_return/r` calls
/// `record_finish`).
///
/// The returned `next_pc` is normally `op.next_pc` (linear advance
/// past the operand bytes); branch handlers (`goto/L` etc.) override
/// this with their target.
pub fn step<Sym: WalkSym>(
    code: &[u8],
    pc: usize,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let op: DecodedOp = decode_op_at(code, pc).ok_or(DispatchError::UndecodableOpcode { pc })?;
    if ctx.is_top_level {
        ctx.session.borrow_mut().recording_opcode_position = op.pc;
    }
    // #73: maintain the `-live-` BEFORE anchor.  Every
    // `-live-` byte the walk decodes becomes the resume point preceding the
    // NEXT guard (`pyjitpl.py`, normal guard resume reads at
    // `self.pc - SIZE_LIVE_OP`).  `op.pc` is the genuine jitcode offset of
    // the `-live-` byte.  Side-data only — read under a debug audit gate.
    if op.opname == "live" {
        ctx.live_before_jit_pc = op.pc;
    }
    // #73: maintain the walk-level operand-stack box
    // mirror.  Detects a Python-opcode boundary at this jitcode pc and
    // reconciles the previous opcode's stack effect into `ctx.vstack_boxes`
    // BEFORE this op runs.  Writes ONLY the new `vstack_*` fields — never
    // the existing registers / snapshot / control flow — so the mirror is
    // pure side-data until a later slice makes the read authoritative.  No-op
    // unless the full-body walk owns the virtualizable shadow and the
    // mirror is still valid.
    step_vstack_mirror(ctx, pc);
    let effects_before = fbw_executed_effect_count();
    let result = handle(&op, code, ctx);
    if matches!(
        result,
        Err(DispatchError::LoopBearingCalleeInlineUnsupported { .. })
    ) {
        FBW_STRUCTURAL_ABORT_OPCODE_EFFECTS.with(|c| {
            c.set(Some((
                op.pc,
                fbw_executed_effect_count().saturating_sub(effects_before),
            )))
        });
    }
    result
}

/// Walk the code from `start_pc` until a terminating opcode fires.
/// Returns the terminating outcome plus the pc immediately after the
/// terminator. Top-level callers expect `DispatchOutcome::Terminate`
/// (other variants appear only inside a sub-walk frame entered via
/// `inline_call_r_r/dR>r` — `ref_return/r` and `raise/r` produce
/// `SubReturn` / `SubRaise` there).
///
/// **Top-level uncaught SubRaise**: when an inline_call
/// SubRaise bubbles up through every parent frame without a
/// `catch_exception/L` handler match and reaches the outermost
/// `walk()` invocation, RPython `pyjitpl.py
/// finishframe_exception` records `compile_exit_frame_with_exception(
/// last_exc_box)` — i.e. `FINISH(exc, exit_frame_with_exception_descr_ref)`
/// + raise `ExitFrameWithExceptionRef`. The walker mirrors this on
/// exit: if the loop terminates with `SubRaise` AND `ctx.is_top_level
/// == true`, record the FINISH and convert the outcome to `Terminate`
/// before returning. Sub-walk frames keep returning `SubRaise` to
/// their callers (the unwind continues until either a handler
/// matches or the outermost walker handles it).
pub fn walk<Sym: WalkSym>(
    code: &[u8],
    start_pc: usize,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let mut pc = start_pc;
    loop {
        let opcode_position = pc;
        let (outcome, next_pc) = step(code, pc, ctx)?;
        pc = next_pc;
        // pyjitpl.py:2865 `_interpret`: `blackhole_if_trace_too_long()` runs
        // after every `run_one_step()`. This loop is that loop's counterpart —
        // `step` is `run_one_step` — and the check came with the per-opcode
        // tracing loop it replaced (`pyre-jit/src/eval.rs` still runs it on
        // that path), so nothing bounded a walk that kept recording without
        // reaching a close. `TraceCtx::is_too_long` is the `history.length() >
        // warmrunnerstate.trace_limit` test and `num_ops` is a `Vec::len`, so
        // this is one comparison per step.
        //
        // The walker layer holds `&mut TraceCtx` and cannot reach
        // `MetaInterp::blackhole_if_trace_too_long` for the full bookkeeping;
        // `note_root_trace_too_long` carries the warm-state half, split into
        // the loop and bridge arms `prepare_trace_segmenting` (pyjitpl.py:2833)
        // keeps apart. The `find_biggest_function` →
        // `disable_noninlinable_function` half (pyjitpl.py:2817) still only
        // runs on the per-opcode path.
        //
        // `blackhole_if_trace_too_long` raises AFTER `run_one_step`, so the
        // forward image must carry `pc`, the already-advanced `next_pc`, rather
        // than `opcode_position`.  `latch_trace_too_long_blackhole` copies the
        // live MIFrame registers while this WalkContext still owns them; the
        // run-per-fn epilogue drives that image forward exactly like RPython's
        // `run_blackhole_interp_to_cancel_tracing`.
        //
        // A complete image makes the abort safe regardless of effects.  If the
        // image cannot be built, a zero-effect walk may still take the legacy
        // replay; an effectful walk must retain the former keep-recording
        // fallback because replay would apply an irreversible effect twice.
        // The remaining overshoot is tallied so an unsupported multi-frame
        // shape cannot silently become unbounded.
        if ctx.trace_ctx.is_too_long() {
            // `step` has advanced the register banks for `Continue`. The
            // other outcomes still need the match below to perform their
            // frame transition: in particular, `SubRaise` may enter this
            // frame's handler and `SubReturn` is delivered to its caller by
            // the inline-call boundary. RPython checks the length only after
            // those transitions have happened, so never abort from the
            // pre-transition callee image here — even a zero-effect entry
            // replay would resume the caller without delivering the return or
            // raise that this step produced.
            let snapshot_safe = trace_too_long_blackhole_snapshot_safe(&outcome);
            let blackhole_latched = snapshot_safe && latch_trace_too_long_blackhole(ctx, pc);
            if trace_too_long_abort_safe(&outcome, blackhole_latched, fbw_executed_effect_count()) {
                let ops = ctx.trace_ctx.num_recorded_ops();
                crate::state::note_root_trace_too_long(
                    ctx.trace_ctx.current_merge_points_first_greenkey(),
                    ctx.trace_ctx.resumekey_original_loop_token().cloned(),
                );
                return Err(DispatchError::TraceTooLong {
                    pc: opcode_position,
                    ops,
                });
            } else {
                majit_metainterp::mc_diag_bump(26);
            }
        }
        match outcome {
            DispatchOutcome::Continue => {}
            DispatchOutcome::Terminate
            | DispatchOutcome::SubReturn { .. }
            | DispatchOutcome::SwitchToBlackhole { .. }
            | DispatchOutcome::CloseLoop { .. }
            | DispatchOutcome::CompileTracePending { .. }
            // A multi-frame inlined callee reached its own loop
            // header; propagate up to `try_walker_inline_user_call`, which
            // emits the recursive CALL_ASSEMBLER at the call boundary.
            | DispatchOutcome::SubLoopCalleeCallAssembler { .. } => {
                return Ok((outcome, pc));
            }
            DispatchOutcome::SubRaise { exc, exc_concrete } => {
                // RPython `finishframe_exception`: before
                // unwinding to the caller, scan THIS frame for a matching
                // `catch_exception/L` handler at the post-op position. A
                // residual call that raised inside a try-block resumes its own
                // except body here; only when the current frame has no handler
                // does the exception unwind further. The `inline_call` SubRaise
                // arm performs the symmetric scan for caller frames, so a raise
                // routes to the nearest enclosing handler regardless of whether
                // it sits in the raising frame or an inlining ancestor. Without
                // this current-frame scan, a top-level walk whose raising op
                // sits in a try-block (e.g. a standalone-traced callee whose
                // getitem raises) skipped its own handler and recorded
                // `exit_frame_with_exception`, letting the exception escape a
                // frame that actually catches it.
                if let Some(target) = try_catch_exception_at(code, pc) {
                    // `op.key` carries the full `opname/argcodes` spelling;
                    // `op.opname` is only the part before the `/`.
                    let raised_in_this_frame = decode_op_at(code, opcode_position)
                        .is_some_and(|op| matches!(op.key, "raise/r" | "reraise/"));
                    // finishframe_exception propagates into this frame
                    // before catch_exception/L enters its handler. Record
                    // that frame-boundary step once; handler entry itself
                    // does not add a traceback node.
                    //
                    // The node has to exist on the compiled run too: the
                    // handler is part of the trace, so this frame never
                    // surfaces an error the interpreter's `handle_exception`
                    // could record from.  A raise the `raise/r` / `reraise/`
                    // arm already routed here is the exception — that arm
                    // emits the runtime record itself, so emitting again
                    // would give the frame two nodes.
                    let recording_opcode_position = ctx.session.borrow().recording_opcode_position;
                    let node_position = if ctx.is_top_level {
                        recording_opcode_position
                    } else {
                        opcode_position
                    };
                    let emit_runtime = !raised_in_this_frame
                        && !record_prepend_application_traceback(
                            ctx,
                            exc,
                            exc_concrete,
                            node_position,
                        );
                    record_inline_application_traceback(
                        ctx,
                        exc,
                        exc_concrete,
                        opcode_position,
                        true,
                        emit_runtime,
                    );
                    record_top_level_application_traceback(
                        ctx,
                        exc,
                        exc_concrete,
                        recording_opcode_position,
                        true,
                        emit_runtime,
                    );
                    ctx.last_exc_value = Some(exc);
                    ctx.last_exc_value_concrete = exc_concrete;
                    // pyjitpl.py:2530-2558 `finishframe_exception` only
                    // unwinds frames and selects the handler.  The shared
                    // MetaInterp exception-class state was established by
                    // `execute_ll_raised` / `handle_possible_exception` or
                    // `opimpl_raise` and must survive the frame transition.
                    // The exception is now caught by this frame's handler, so
                    // drain the standing residual-call exception flag the
                    // raising helper published (`try_execute_residual_call_via_
                    // executor` Err arm restores `BH_LAST_EXC_VALUE` so the
                    // interpreter's walker-skip path can re-raise an *uncaught*
                    // exception). Leaving it set would make that path spuriously
                    // re-raise the now-handled exception after the walk ends.
                    // Mirrors `blackhole.rs route_to_catch`'s `BH_LAST_EXC_VALUE
                    // = 0` on handler entry.
                    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
                    // #370: re-seed the operand-stack mirror at the handler
                    // entry so a kept-stack guard inside the handler compiles
                    // from the mirror instead of declining.  The unwind
                    // boundary is otherwise unmodeled, so without this the
                    // mirror latches invalid and every in-handler kept-stack
                    // guard declines.  `vstack_enter_exception_handler` falls
                    // back safely (per-slot NONE hole / `vstack_valid = false`)
                    // for the coordinates it cannot reconstruct.
                    vstack_enter_exception_handler(ctx, target, exc);
                    pc = target;
                    continue;
                }
                if ctx.is_top_level {
                    let recording_opcode_position =
                        ctx.session.borrow().recording_opcode_position;
                    if !recording_instruction_is_bare_reraise(ctx, opcode_position) {
                        record_top_level_application_traceback(
                            ctx,
                            exc,
                            exc_concrete,
                            recording_opcode_position,
                            true,
                            false,
                        );
                    }
                    // The interpreter records this frame's own traceback node
                    // when the trace hands it the exception, and it reads the
                    // raise coordinate out of `frame.last_instr`.  Compiled
                    // code never wrote that field, so publish it here.
                    fbw_publish_exit_last_instr(ctx, recording_opcode_position);
                    // RPython parity: framestack exhausted with no handler
                    // match → `compile_exit_frame_with_exception(last_exc_box)`.
                    // Stash the exception the same way the value-return arms
                    // stash their result (payload + no inline FINISH); the
                    // Terminate arm builds `TraceAction::Finish {
                    // exit_with_exception: true }` so the compile consumer
                    // records the FINISH once against
                    // `exit_frame_with_exception_descr`.  Recording it here too
                    // would double it.
                    fbw_terminate_with_raise(exc, exc_concrete);
                    return Ok((DispatchOutcome::Terminate, pc));
                } else {
                    if !recording_instruction_is_bare_reraise(ctx, opcode_position) {
                        // Emit the node at runtime as well as applying it for
                        // the recording pass.  The top-level arm above can
                        // leave this to the interpreter: its frame is a real
                        // `PyFrame`, so the `exit_frame_with_exception` the
                        // trace finishes with surfaces as an error on that
                        // frame and `handle_operation_error` records the node.
                        // An inlined callee has no frame to return to — the
                        // walk pops it symbolically — so unless the trace
                        // carries the record itself, the callee contributes no
                        // node once the trace runs compiled and the exception
                        // reaches its handler one frame short.
                        let emit_runtime = !record_prepend_application_traceback(
                            ctx,
                            exc,
                            exc_concrete,
                            opcode_position,
                        );
                        record_inline_application_traceback(
                            ctx,
                            exc,
                            exc_concrete,
                            opcode_position,
                            true,
                            emit_runtime,
                        );
                    }
                    return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, pc));
                }
            }
        }
    }
}

/// Read a Ref-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_r[code[pc+1]]` for an `r`-coded operand.
fn read_ref_reg<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_r
        .get(reg)
        .copied()
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_r.len(),
            bank: "r",
        })
}

/// Read an Int-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_i[code[pc+1]]` for an `i`-coded operand.
fn read_int_reg<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_i
        .get(reg)
        .copied()
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_i.len(),
            bank: "i",
        })
}

/// Read a Float-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_f[code[pc+1]]` for an `f`-coded operand.
fn read_float_reg<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_f
        .get(reg)
        .copied()
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_f.len(),
            bank: "f",
        })
}

/// Read a 2-byte little-endian label operand at `pc + 1 +
/// operand_offset`. RPython encoding: `assembler.py:write_label`
/// writes the resolved target as `chr(target & 0xFF)` +
/// `chr((target >> 8) & 0xFF)`, matching `bhimpl_goto`'s
/// `code[pc] | (code[pc+1] << 8)` decode.
pub(crate) fn read_label(code: &[u8], op: &DecodedOp, operand_offset: usize) -> usize {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    lo | (hi << 8)
}

/// Outcome of probing the per-frame raise-bubbling lookahead at
/// `position` (the pc just after a raising op).
///
/// RPython parity: `finishframe_exception` walks
/// through three mutually-exclusive cases after skipping a leading
/// `live/`:
///
///   1. Next op is `catch_exception/L` → jump to the handler target,
///      `raise ChangeFrame`. (Handler matched.)
///   2. Next op is `rvmprof_code/ii` → call `cintf.jit_rvmprof_code(arg1,
///      arg2)` for instrumentation, then fall through to `popframe()`
///      (continue unwinding).
///   3. Otherwise → `popframe()` (continue unwinding).
///
/// Cases 2 and 3 both unwind, but case 2 also fires the rvmprof side
/// effect. RPython at line 2531 invokes `cintf.jit_rvmprof_code(arg1,
/// arg2)` directly during tracing — RPython does NOT record this as
/// an IR op, but the side effect IS observable (it advances the
/// rvmprof profiler state). The helper surfaces the matched register
/// pair via [`FinishframeLookahead::RvmprofCode`] so a future port
/// can invoke pyre's `bh.handle_rvmprof_enter`-equivalent
/// (`pyre-jit/src/call_jit.rs`); the walker records IR directly
/// and the rvmprof side effect is not yet ported, so the caller drops
/// it here — TODO,
/// scoped to the rvmprof profiler instrumentation only (no trace IR
/// effect).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum FinishframeLookahead {
    /// Handler match: caller-frame jump target (2-byte LE label after
    /// `catch_exception/L`). Caller sets `last_exc_value` and resumes
    /// at `target`.
    CatchTarget(usize),
    /// `rvmprof_code/ii` lies on the unwind path. Caller continues
    /// unwinding (no handler match); the runtime `cintf.jit_rvmprof_code`
    /// side effect (`pyjitpl.py`, mirrored by
    /// `blackhole.rs bhimpl_rvmprof_code`) is dropped for now. The op is
    /// only emitted when the interpreter main loop is rvmprof-instrumented,
    /// and the call is itself a no-op unless a vmprof profiler HOOK is
    /// installed, so the drop is inert for ordinary execution.
    ///
    /// Not yet fired here because it needs the concrete operands
    /// `registers_i[arg1_reg].getint()` / `[arg2_reg].getint()`, and the
    /// walker's Concrete shadow bank (`concrete_registers_i`) is not yet
    /// reliably seeded — the same blocker that keeps `goto_if_not/iL` and
    /// `switch/id` on the strict fail-loud fallback (see `WalkContext`).
    /// `arg1_reg` / `arg2_reg` are surfaced so that, once seeding lands,
    /// the call ports directly (`assert arg1 == 1; jit_rvmprof_code(arg1,
    /// arg2)`) without re-decoding.
    #[allow(dead_code)]
    RvmprofCode { arg1_reg: u8, arg2_reg: u8 },
    /// Neither match — unwinding continues with no side effect.
    NoMatch,
}

/// Probe the per-frame raise-bubbling lookahead. RPython parity:
/// `finishframe_exception` line-by-line —
/// `live/` skip then sequential `catch_exception` / `rvmprof_code` /
/// fall-through arms.
fn finishframe_lookahead_at(code: &[u8], position: usize) -> FinishframeLookahead {
    let mut pos = position;
    let Some(op) = decode_op_at(code, pos) else {
        return FinishframeLookahead::NoMatch;
    };
    // RPython `if opcode == op_live: position += SIZE_LIVE_OP`.
    if op.key == "live/" {
        pos = op.next_pc;
    }
    let Some(next) = decode_op_at(code, pos) else {
        return FinishframeLookahead::NoMatch;
    };
    if next.key == "catch_exception/L" {
        let lo = code[next.pc + 1] as usize;
        let hi = code[next.pc + 2] as usize;
        return FinishframeLookahead::CatchTarget(lo | (hi << 8));
    }
    if next.key == "rvmprof_code/ii" {
        // RPython `pyjitpl.py`:
        //   arg1 = frame.registers_i[ord(code[position + 1])].getint()
        //   arg2 = frame.registers_i[ord(code[position + 2])].getint()
        //   assert arg1 == 1
        //   cintf.jit_rvmprof_code(arg1, arg2)
        // Walker surfaces the operand byte indices for the caller to
        // decide whether to symbolically record (today: drop, mirroring
        // RPython's non-record direct cintf call).
        let arg1_reg = code[next.pc + 1];
        let arg2_reg = code[next.pc + 2];
        return FinishframeLookahead::RvmprofCode { arg1_reg, arg2_reg };
    }
    FinishframeLookahead::NoMatch
}

/// Convenience wrapper preserving the legacy
/// `try_catch_exception_at(...) -> Option<target>` shape used by
/// existing callers. Returns `Some(target)` only on the
/// `CatchTarget` arm; `RvmprofCode` and `NoMatch` collapse to `None`
/// (both cases continue unwinding from the caller's POV — the
/// instrumentation side effect is dropped today, matching RPython's
/// non-trace-recorded `cintf` call).
pub(crate) fn try_catch_exception_at(code: &[u8], position: usize) -> Option<usize> {
    match finishframe_lookahead_at(code, position) {
        FinishframeLookahead::CatchTarget(target) => Some(target),
        FinishframeLookahead::RvmprofCode { .. } | FinishframeLookahead::NoMatch => None,
    }
}

/// Exception-edge bridge: route an exception-guard bridge
/// (GUARD_NO_EXCEPTION / GUARD_EXCEPTION) resume to the in-frame `except`
/// handler instead of declining to the blackhole (`call_jit.rs` pending-exc
/// decline). Native backends run the exception-edge bridge unconditionally.
/// The wasm guest's abort-replay exception class (#727) is still open, so it
/// stays off there.
pub fn exc_edge_bridge_enabled() -> bool {
    cfg!(not(target_arch = "wasm32"))
}

/// `PYRE_CARRIER_EXC_RESUME=1` enables the multi-frame (carrier) exception
/// resume: seed the grabbed guard exception onto the bridge sym and route the
/// inlined callee's carrier sub-walk into its own `catch_exception` handler
/// (`finishframe_exception` parity, pyjitpl.py:2530).  Default-off while the
/// #343/#126 depth-2 exception-resume slice is validated bit-exact.
pub fn carrier_exc_resume_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_CARRIER_EXC_RESUME").is_some())
}

/// Mirror of `blackhole.rs BlackholeInterpreter::handle_exception_in_frame`
/// for the walker: locate the `catch_exception/L` that owns an exception-guard
/// resume position, forward case first, then the backward scan.
///
/// Forward case (blackhole.py:396): the `catch_exception` sits directly after
/// the resume `-live-` — explicit `raise` sites, and residual ops whose resume
/// coordinate is their own post-call `-live-` (a mid-Python-op residual such as
/// a binary-op helper resumes at its OWN op's `-live-`, with its catch
/// following it).  Backward case: the after-residual-call layout where the
/// catch sits BEHIND the resume `-live-` (`find_catch_before_resume_live`).
/// Returns the handler target (2-byte LE label), or `None` when the raising op
/// sits outside any in-frame try (propagate).
pub(crate) fn find_catch_for_exc_resume(code: &[u8], resume_live_pos: usize) -> Option<usize> {
    let mut position = resume_live_pos;
    if let Some(op) = decode_op_at(code, position) {
        if op.key == "live/" {
            position = op.next_pc;
        }
        if let Some(op) = decode_op_at(code, position) {
            if op.key == "catch_exception/L" {
                let lo = code[position + 1] as usize;
                let hi = code[position + 2] as usize;
                return Some(lo | (hi << 8));
            }
        }
    }
    find_catch_before_resume_live(code, resume_live_pos)
}

/// Mirror of `blackhole.rs BlackholeInterpreter::find_catch_before_resume_live`
/// for the walker.  An after-residual-call exception guard resumes at the
/// no-exception fallthrough `-live-` (the next opcode after the call); the
/// `catch_exception/L` that belongs to the just-executed raising op sits BEHIND
/// that resume `-live-`.  The caught can-raise op's expansion is `[op, -live-,
/// catch_exception, -live-(block entry), vable stores…]` (the guessexception
/// split closes the block at the op's trailing `-live-`, so the successor block
/// opens with its own `-live-` before the moved stores).  Scan op boundaries
/// backward from `resume_live_pos`, newest first, hopping over exactly one
/// `-live-` (the successor's block-entry marker); accept the `catch_exception`
/// only when it sits immediately before it — any other op there is the raising
/// op itself (uncaught), and a second `-live-` means the raising op sits
/// outside any in-frame try.
/// Returns the handler target (2-byte LE label after `catch_exception/L`), or
/// `None` when the raising op sits outside any in-frame try (propagate).
/// The `catch_exception/L` target of the can-raise op whose OWN trailing
/// `-live-` sits at `resume_live_pos`.
///
/// A caught can-raise op expands to `[op, -live-, catch_exception,
/// -live-(block entry), vable stores…]`.  [`find_catch_before_resume_live`]
/// keys off the SUCCESSOR block-entry `-live-` and reads backward from it; a
/// caller holding the op's own trailing `-live-` instead is one op short of
/// that coordinate, so the same scan walks straight past the catch. Read
/// forward for those callers: when `resume_live_pos` is a `-live-` whose very
/// next op is a `catch_exception`, that catch is the one guarding the op.
pub(crate) fn catch_target_after_resume_live(code: &[u8], resume_live_pos: usize) -> Option<usize> {
    let live = decode_op_at(code, resume_live_pos)?;
    if live.key != "live/" {
        return None;
    }
    let catch = decode_op_at(code, live.next_pc)?;
    if catch.key != "catch_exception/L" {
        return None;
    }
    let lo = *code.get(catch.pc + 1)? as usize;
    let hi = *code.get(catch.pc + 2)? as usize;
    Some(lo | (hi << 8))
}

pub(crate) fn find_catch_before_resume_live(code: &[u8], resume_live_pos: usize) -> Option<usize> {
    let mut pcs: Vec<usize> = crate::jitcode_runtime::decoded_ops(code)
        .map(|op| op.pc)
        .filter(|&pc| pc < resume_live_pos)
        .collect();
    pcs.sort_unstable_by(|a, b| b.cmp(a));
    let mut crossed_block_entry_live = false;
    for pc in pcs {
        let op = decode_op_at(code, pc)?;
        if op.key == "catch_exception/L" {
            let lo = code[pc + 1] as usize;
            let hi = code[pc + 2] as usize;
            return Some(lo | (hi << 8));
        }
        if op.key == "live/" {
            if crossed_block_entry_live {
                // Second `-live-`: the raising op has no catch.
                return None;
            }
            crossed_block_entry_live = true;
            continue;
        }
        if crossed_block_entry_live {
            // The op immediately before the block-entry `-live-` is not a
            // `catch_exception` — it is the raising op itself (uncaught).
            return None;
        }
    }
    None
}

/// Byte offset of the 2-byte label operand inside an op's operand block,
/// derived from the operand signature in `key` (the part after `/`, up to the
/// `>` that introduces the result). Register operands (`i` / `r` / `f`) are one
/// byte each, so the offset is the number of them preceding the `L`.
///
/// `None` when the op carries no label, or when an operand whose width this
/// decode does not model (a var-list, a descr) sits in front of it.
fn label_operand_offset(key: &str) -> Option<usize> {
    let signature = key.split('/').nth(1)?.split('>').next()?;
    let mut offset = 0usize;
    for operand in signature.chars() {
        match operand {
            'L' => return Some(offset),
            'i' | 'r' | 'f' => offset += 1,
            _ => return None,
        }
    }
    None
}

/// Does the `except` handler at `catch_target` flow back into this frame's loop
/// (reaching a `jit_merge_point` back-edge), rather than returning out of the
/// frame (`*_return`)?
///
/// Sole caller: [`decline_inline_caller_frame_for_catch_marker`].  The
/// exception-edge bridge router itself does NOT consult this — it routes on the
/// `catch_exception` alone, the way `finishframe_exception`
/// (`pyjitpl.py:2530-2546`) does, and a handler that returns out of the frame is
/// `finishframe`'s ordinary case (`pyjitpl.py:2503-2525`).
///
/// What the predicate still gates is INLINING a CLOSURE callee at a caller's
/// in-try CALL.  A non-rejoining handler is an `except E as e` body, and the
/// callee's free variables resolve through cells that body's implicit cleanup
/// stores `None` into and then clears; the inlined read answers `None`
/// (`synth/exception_as_cell_cleanup`, dynasm).  A callee with no free
/// variables cannot reach a caller cell and is inlined either way.
///
/// Bounded forward reachability from `catch_target`, following `goto`/
/// `goto_if_not` successors: `true` as soon as any path reaches a
/// `jit_merge_point`; `false` if every reachable path terminates at a `*_return`
/// (or the scan hits an un-followed control op / the bound, which conservatively
/// declines).
pub(crate) fn exc_handler_rejoins_loop(code: &[u8], catch_target: usize) -> bool {
    let mut visited = std::collections::HashSet::new();
    let mut work = vec![catch_target];
    let mut budget = 4096usize;
    while let Some(pc) = work.pop() {
        if budget == 0 {
            return false;
        }
        budget -= 1;
        if !visited.insert(pc) {
            continue;
        }
        let Some(op) = decode_op_at(code, pc) else {
            continue;
        };
        if op.key.starts_with("jit_merge_point") {
            return true;
        }
        if matches!(
            op.key,
            "ref_return/r" | "int_return/i" | "float_return/f" | "void_return/"
        ) {
            // Frame-return terminal on this path; do not enqueue successors.
            continue;
        }
        match op.key {
            "goto/L" => work.push(read_label(code, &op, 0)),
            "goto_if_not/iL" => {
                // `iL`: 1B int register + 2B LE label.
                work.push(read_label(code, &op, 1));
                work.push(op.next_pc);
            }
            key if key.starts_with("switch") => {
                // Multi-target dispatch not followed; leave this path un-proven
                // (routing declines unless another path rejoins the loop).
            }
            _ => work.push(op.next_pc),
        }
    }
    false
}

/// True when a path reachable from `position` reads the walker's active
/// exception (`last_exception/>i` / `last_exc_value/>r`) before reaching the
/// `catch_exception/L` that would replace it.
///
/// The jitcode is one flat byte stream whose blocks are stitched together by
/// `goto/L` labels, so the block that physically follows an op is frequently
/// not its successor. Advancing by `op.next_pc` alone therefore walks out of
/// the current block and answers from whatever block the assembler happened to
/// place next — reporting that block's `catch_exception/L` while the real
/// successor reads the exception. The residual-call sites act on that answer by
/// clearing `last_exc_value`, so the read then finds no active exception and
/// the walk aborts (`LastExcValueWithoutActiveException`) for the whole run.
///
/// Labels are followed instead: `goto/L` continues at its target, a
/// conditional branch queues its taken arm and continues on the fall-through,
/// and ops that end a path stop it — `catch_exception/L` because the handler it
/// introduces installs a fresh exception, `raise/r` because it replaces the
/// current one, `unreachable/` because it has no successor, and the `*_return`
/// family because the frame is gone. Targets are visited once, which bounds the
/// walk across back edges.
///
/// The answer is an over-approximation of the reads: whenever the successor set
/// is not decodable — `switch/id`, whose targets live in a descr this scan
/// cannot read, or any other label-carrying op whose shape is not modelled —
/// the exception is reported as read, as it is for an op the decode cannot read
/// at all. Over-reporting only leaves a recorded exception standing for the
/// caller, whereas under-reporting drops one a later read still needs.
fn reads_last_exc_before_next_catch(code: &[u8], position: usize) -> bool {
    // Arms queued by a conditional branch, and the jump targets already
    // started from. Both stay empty on a straight-line answer.
    let mut pending: Vec<usize> = Vec::new();
    let mut started: Vec<usize> = Vec::new();
    let mut pc = position;
    loop {
        let path_continues = match decode_op_at(code, pc) {
            // The decode fails on an opcode byte outside the instruction table
            // and on a payload the stream is too short to hold, not only at the
            // end of the code — the ops beyond it are unknown either way.
            None => return true,
            Some(op) => match op.key {
                // `opimpl_reraise` re-raises `last_exc_value`, and
                // `opimpl_goto_if_exception_mismatch` asserts it before testing
                // the class, so both read the exception exactly like
                // `last_exc_value/>r`.
                "last_exception/>i"
                | "last_exc_value/>r"
                | "reraise/"
                | "goto_if_exception_mismatch/iL" => return true,
                "catch_exception/L" | "raise/r" | "unreachable/" | "int_return/i"
                | "int_return/c" | "ref_return/r" | "float_return/f" | "void_return/" => false,
                "goto/L" => {
                    let target = read_label(code, &op, 0);
                    let fresh = !started.contains(&target);
                    if fresh {
                        started.push(target);
                        pc = target;
                    }
                    fresh
                }
                key if key.starts_with("goto_if") || key.contains("jump_if_ovf") => {
                    let Some(offset) = label_operand_offset(key) else {
                        return true;
                    };
                    let target = read_label(code, &op, offset);
                    if !started.contains(&target) {
                        started.push(target);
                        pending.push(target);
                    }
                    pc = op.next_pc;
                    true
                }
                key => {
                    if key == "switch/id" || label_operand_offset(key).is_some() {
                        return true;
                    }
                    pc = op.next_pc;
                    true
                }
            },
        };
        if !path_continues {
            match pending.pop() {
                Some(next) => pc = next,
                None => return false,
            }
        }
    }
}

/// Read a 2-byte little-endian descr index operand and resolve to
/// the descr from [`WalkContext::descr_refs`]. RPython equivalent:
/// `BlackholeInterpreter.descrs[code[pc] | (code[pc+1] << 8)]`
/// (`blackhole.py` setup + per-`bhimpl_*` site).
fn read_descr<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<DescrRef, DispatchError> {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    let index = lo | (hi << 8);
    ctx.descr_refs
        .get(index)
        .cloned()
        .ok_or(DispatchError::DescrIndexOutOfRange {
            pc: op.pc,
            index,
            len: ctx.descr_refs.len(),
        })
}

fn concrete_int_for_switch<Sym: WalkSym>(
    op: &DecodedOp,
    value: OpRef,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<i64, DispatchError> {
    match ctx.trace_ctx.concrete_of_opref(value) {
        Some(Value::Int(v)) => Ok(v),
        _ => Err(DispatchError::SwitchValueNotConcrete { pc: op.pc, value }),
    }
}

// Walker guard recording (`GuardNoException`, `GuardNotForced` after
// residual_call) pairs every guard with
// `walker_capture_snapshot_for_last_guard`, the walker-side port of
// RPython's `capture_resumedata(after_residual_call=True)`
// (`pyjitpl.py`).  RPython `pyjitpl.py
// generate_guard` walks `metainterp.framestack` and consults per-opcode
// liveness (`pyjitpl.py get_list_of_active_boxes`) to encode the
// live `i`/`r`/`f` registers in i→r→f order plus virtualizable / vref
// boxes.  Walker's helper today omits per-PC liveness narrowing (future
// follow-up: thread the `op_live` byte table through `SubJitCodeBody`)
// and conservatively snapshots every non-`OpRef::NONE` register —
// over-capture is correctness-preserving because the optimizer's
// `store_final_boxes_in_guard` (`optimizeopt/mod.rs`) derives
// `op.fail_args` from the snapshot via `store_final_boxes(liveboxes)`,
// so dead registers are dropped before they reach the backend.  Walker
// IR is no longer rolled back via `cut_trace` for the production
// dispatch; the snapshot must therefore be RPython-orthodox.

/// Read a Ref-bank variadic operand list (`R` argcode): 1 length byte
/// followed by `len` register bytes. Returns the resolved [`OpRef`]s
/// in jitcode order plus the total operand byte width (so callers can
/// skip past or compute downstream operand offsets).
///
/// RPython parity: `assembler.py:write_varlist` emits exactly this
/// shape — `chr(len(args))` followed by one byte per arg register.
fn read_ref_var_list<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<(Vec<OpRef>, usize), DispatchError> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let reg = code[len_pc + 1 + i] as usize;
        let opref = ctx
            .registers_r
            .get(reg)
            .copied()
            .ok_or(DispatchError::RegisterOutOfRange {
                pc: op.pc,
                reg,
                len: ctx.registers_r.len(),
                bank: "r",
            })?;
        out.push(opref);
    }
    Ok((out, 1 + len))
}

/// Read a Ref-bank register operand's concrete shadow value.
/// Mirrors [`read_ref_reg`] but indexes into
/// `ctx.concrete_registers_r`. Returns `ConcreteValue::Null` when the
/// register is out of range — symmetric with `concrete_value_at`'s
/// fallback at `state.rs`. Out-of-range OpRef reads still surface
/// `RegisterOutOfRange` via [`read_ref_reg`]; this helper assumes the
/// OpRef read succeeded, so a missing concrete slot is "stack tail not
/// yet seeded" not "register byte out of range".
fn read_ref_reg_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> ConcreteValue {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.concrete_registers_r
        .get(reg)
        .copied()
        .unwrap_or(ConcreteValue::Null)
}

/// Write a Ref-bank register and its concrete shadow in lock-step.
/// Replaces the inlined
/// `registers_r.get_mut(dst).ok_or(...)?; *slot = value` pattern at
/// every walker handler that writes `registers_r[dst]`.  The concrete
/// shadow update is the WHOLE POINT of this helper: the shadow MUST
/// stay in sync with the symbolic side or
/// downstream consumers (`raise/r` GUARD_CLASS, future
/// `getfield_gc_r` cache lookups) will silently mis-fire.
///
/// `concrete` semantics:
/// * `ConcreteValue::Ref(ptr)` — the handler knows the concrete result
///   (e.g. `ref_copy/r>r` propagating from the source slot's shadow,
///   `raise/r` setting the just-raised exception's concrete).
/// * `ConcreteValue::Null` — the handler doesn't know (most recorded
///   ops: field reads, residual calls, …).  Downstream GUARD_CLASS
///   gates treat Null as "skip the guard", same as slots the snapshot
///   never populated.
fn write_ref_reg<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    value: OpRef,
    concrete: ConcreteValue,
) -> Result<(), DispatchError> {
    let len = ctx.registers_r.len();
    let slot = ctx
        .registers_r
        .get_mut(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc,
            reg: dst,
            len,
            bank: "r",
        })?;
    *slot = value;
    // Snapshot is sized to `registers_r.len()` at dispatch entry, so
    // a dst-in-bounds OpRef write implies in-bounds for the shadow.
    // `get_mut` defensively to tolerate sub-walk shadows that lag the
    // OpRef bank if a future caller mis-sizes them.
    //
    // collapse non-Ref ConcreteValue (Int / Float)
    // to Null before storing into the Ref shadow.  `concrete_from_
    // recorded_opref` returns whatever kind the per-OpRef concrete
    // table holds; a kind mismatch (e.g. boxed Int returned through a
    // Ref result slot) would otherwise leak Int/Float bits into
    // `concrete_registers_r`, breaking ref-only downstream consumers
    // (`getfield_gc_r` sanity loads, `raise/r` GUARD_CLASS) that
    // expect `ConcreteValue::Ref(_)` or `Null`.
    let sanitized = match concrete {
        ConcreteValue::Ref(_) | ConcreteValue::Null => concrete,
        ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => {
            ConcreteValue::Null
        }
    };
    if let Some(c_slot) = ctx.concrete_registers_r.get_mut(dst) {
        *c_slot = sanitized;
    }
    // #73: record the box just written as the candidate
    // operand-stack TOS for the current Python opcode.  A value-producing
    // opcode (LOAD_*, BINARY_OP, COPY, …) lands its result on the stack
    // TOS; this is the last Ref it writes, so capturing it here lets
    // `reconcile_vstack_at_boundary` reconstruct the new TOS without a
    // per-opcode hook.  Cheap unconditional write to a new side-field;
    // only consumed when the mirror is valid (never alters existing state).
    ctx.vstack_last_ref = value;
    Ok(())
}

/// Write a pyre scalar virtualizable Ref field without stamping operand TOS.
///
/// Pyre's scalar virtualizable fields are `last_instr(0)`, `pycode(1)`,
/// `valuestackdepth(2)`, `debugdata(3)`, `lastblock(4)`, and `w_globals(5)`
/// (`virtualizable_gen.rs`, `NUM_VABLE_SCALARS = 6`).  They are frame
/// bookkeeping; the Python operand stack lives in the separate
/// `locals_cells_stack_w` array (`virtualizable_gen.rs`,
/// `pyre-interpreter/src/pyframe.rs`).  PyPy's `interp_jit.py`
/// grounds the same scalar-vs-array split, and `pyjitpl.py
/// get_list_of_active_boxes` sources liveness-indexed register boxes, not
/// scalar virtualizable fields.
fn write_vable_field_ref_reg<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    value: OpRef,
    concrete: ConcreteValue,
) -> Result<(), DispatchError> {
    let saved = ctx.vstack_last_ref;
    write_ref_reg(ctx, pc, dst, value, concrete)?;
    ctx.vstack_last_ref = saved;
    Ok(())
}

/// Int-bank twin of [`read_ref_reg_concrete`] (Int-bank concrete shadow).
/// Reads the Int-bank slot at the operand index from
/// `ctx.concrete_registers_i`.  Returns `ConcreteValue::Null` for
/// out-of-range reads — the only legal time the slice is shorter than
/// `registers_i` is at test fixtures that pass `&mut []`, and those
/// don't trigger `goto_if_not/iL` / `switch/id` paths.
fn read_int_reg_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> ConcreteValue {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.concrete_registers_i
        .get(reg)
        .copied()
        .unwrap_or(ConcreteValue::Null)
}

/// Float-bank twin of [`read_int_reg_concrete`]. Reads the Float-bank slot at
/// the operand index and resolves its concrete value from the OpRef's value
/// carrier. Returns `ConcreteValue::Null` when the register is out of range or
/// has no concrete Float value.
pub(crate) fn read_float_reg_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> ConcreteValue {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_f
        .get(reg)
        .and_then(|&value| ctx.trace_ctx.concrete_of_opref(value))
        .map_or(ConcreteValue::Null, |value| match value {
            Value::Float(v) => ConcreteValue::Float(v),
            _ => ConcreteValue::Null,
        })
}

/// Int-bank twin of [`write_ref_reg`] (Int-bank concrete shadow).  Writes an Int
/// register and its concrete shadow in lock-step.  Mirrors the
/// Ref-bank contract: every walker handler that writes
/// `registers_i[dst]` MUST also write `concrete_registers_i[dst]` so
/// downstream `goto_if_not/iL` / `switch/id` can fold the branch.
///
/// `concrete` semantics:
/// * `ConcreteValue::Int(v)` — the handler knows the concrete result
///   (e.g. `int_copy/i>i` propagating from the source slot's shadow,
///   an `int_<binop>` fold of two concrete inputs).
/// * `ConcreteValue::Null` — the handler doesn't know (e.g. residual
///   `Call*I`, `getfield_gc_i` cache miss).  Downstream consumers
///   surface `GotoIfNotValueNotConcrete` for unknown branch inputs.
fn write_int_reg<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    value: OpRef,
    concrete: ConcreteValue,
) -> Result<(), DispatchError> {
    let len = ctx.registers_i.len();
    let slot = ctx
        .registers_i
        .get_mut(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc,
            reg: dst,
            len,
            bank: "i",
        })?;
    *slot = value;
    // Mirror `write_ref_reg`'s defensive get_mut.  Test fixtures pass
    // an empty `concrete_registers_i` slice; production callers
    // (Concrete shadow seeding) size it to `registers_i.len()` at dispatch entry.
    //
    // Symmetry with `write_ref_reg`: collapse
    // non-Int ConcreteValue to Null before storing into the Int
    // shadow so a kind-mismatched stamp can't leak Ref/Float bits
    // into `concrete_registers_i`.
    let sanitized = match concrete {
        ConcreteValue::Int(_) | ConcreteValue::Null => concrete,
        // `is_int(bool_obj)` is true and `ConcreteValue::Bool::getint()`
        // coerces to `bool as i64`, so booleans flow safely through the
        // Int shadow.
        ConcreteValue::Bool(v) => ConcreteValue::Int(v as i64),
        ConcreteValue::Ref(_) | ConcreteValue::Float(_) => ConcreteValue::Null,
    };
    if let Some(c_slot) = ctx.concrete_registers_i.get_mut(dst) {
        *c_slot = sanitized;
    }
    Ok(())
}

/// Derive a `ConcreteValue` for shadow write-back from a freshly
/// recorded `OpRef` via `concrete_of_opref` (concrete_of_opref derivation).
///
/// RPython parity: `pyjitpl.py:execute_with_descr` /
/// `rpython/jit/metainterp/executor.py` stamps `box.value` on every
/// executed op result through the per-opcode LLOp executor — `Box.value`
/// IS the load-bearing concrete channel.  Pyre's `concrete_of_opref`
/// table-lookup is the orthodox shadow of that channel: constant pool
/// (`history.py`), virtualizable boxes
/// (`pyjitpl.py`), `set_opref_concrete` stamps from
/// `binop_int_record` / `unop_int_record`, and standard virtualizable
/// box hits all surface here.
///
/// Returns `ConcreteValue::Null` when `concrete_of_opref` yields `None`
/// (no concrete known) — the caller's downstream `goto_if_not/iL` /
/// GUARD_CLASS dispatch treats Null as "skip the fold / skip the guard".
/// A residual `Value::Ref(GcRef(usize::MAX))` storage placeholder
/// (`vable_setfield`) is likewise mapped to Null since it signals "no
/// concrete known" rather than an actual pointer.
#[inline]
fn concrete_from_recorded_opref<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    opref: OpRef,
) -> ConcreteValue {
    match ctx.trace_ctx.concrete_of_opref(opref) {
        Some(Value::Int(v)) => ConcreteValue::Int(v),
        Some(Value::Float(v)) => ConcreteValue::Float(v),
        Some(Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE => {
            ConcreteValue::Ref(r.as_usize() as pyre_object::PyObjectRef)
        }
        _ => ConcreteValue::Null,
    }
}

/// Read concrete shadow values for a Ref-bank variadic operand list.
/// Parallels [`read_ref_var_list`] — reads the
/// same byte indices but resolves through `ctx.concrete_registers_r`.
/// Used by `inline_call_*` to propagate per-arg concrete shadow into
/// the callee's fresh shadow Vec.
fn read_ref_var_list_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Vec<ConcreteValue> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    (0..len)
        .map(|i| {
            let reg = code[len_pc + 1 + i] as usize;
            ctx.concrete_registers_r
                .get(reg)
                .copied()
                .unwrap_or(ConcreteValue::Null)
        })
        .collect()
}

/// Read concrete shadow values for an Int-bank variadic operand list.
/// Int twin of [`read_ref_var_list_concrete`] — same byte indices,
/// resolved through `ctx.concrete_registers_i`.  Used by `inline_call_*`
/// to propagate each int arg's concrete shadow into the callee's fresh
/// shadow Vec (`setup_call` int-bank parity), so a callee body can fold
/// a `goto_if_not/iL` / `switch/id` over a concrete primitive-int arg.
fn read_int_var_list_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Vec<ConcreteValue> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    (0..len)
        .map(|i| {
            let reg = code[len_pc + 1 + i] as usize;
            ctx.concrete_registers_i
                .get(reg)
                .copied()
                .unwrap_or(ConcreteValue::Null)
        })
        .collect()
}

/// Read an Int-bank variadic operand list (`I` argcode). Same shape as
/// [`read_ref_var_list`] but indexes into `registers_i`. RPython
/// `assembler.py:write_varlist` emits a single shape regardless of
/// kind; the kind letter (`I` / `R` / `F`) only steers which register
/// file the bytes index into.
fn read_int_var_list<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<(Vec<OpRef>, usize), DispatchError> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let reg = code[len_pc + 1 + i] as usize;
        let opref = ctx
            .registers_i
            .get(reg)
            .copied()
            .ok_or(DispatchError::RegisterOutOfRange {
                pc: op.pc,
                reg,
                len: ctx.registers_i.len(),
                bank: "i",
            })?;
        out.push(opref);
    }
    Ok((out, 1 + len))
}

/// Read a Float-bank variadic operand list (`F` argcode). Mirror of
/// [`read_int_var_list`] for the float bank.
fn read_float_var_list<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<(Vec<OpRef>, usize), DispatchError> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let reg = code[len_pc + 1 + i] as usize;
        let opref = ctx
            .registers_f
            .get(reg)
            .copied()
            .ok_or(DispatchError::RegisterOutOfRange {
                pc: op.pc,
                reg,
                len: ctx.registers_f.len(),
                bank: "f",
            })?;
        out.push(opref);
    }
    Ok((out, 1 + len))
}

/// RPython `pyjitpl.py opimpl_switch`:
///
/// * read the traced value box and concrete `valuebox.getint()`
/// * on hit, `implement_guard_value(valuebox, orgpc)` and jump target
/// * on miss, emit `INT_EQ(valuebox, ConstInt(key))` plus `GUARD_FALSE`
///   for every `switchdict.const_keys_in_order`, then fall through
///
/// Each `GUARD_VALUE` / `GUARD_FALSE` attaches a production resume snapshot
/// via `walker_capture_snapshot_for_last_guard`, mirroring `pyjitpl.py
/// opimpl_switch`'s `generate_guard(..., resumepc=orgpc) →
/// capture_resumedata(orgpc)`.
fn dispatch_switch_id<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let valuebox = read_int_reg(code, op, 0, ctx)?;
    let descr = read_descr(code, op, 1, ctx)?;
    let switchdict = descr
        .as_switch_descr()
        .ok_or(DispatchError::ExpectedSwitchDescr { pc: op.pc })?;
    let search_value = concrete_int_for_switch(op, valuebox, ctx)?;

    if let Some(target) = switchdict.lookup(search_value) {
        if !valuebox.is_constant() {
            let expected = ctx.trace_ctx.const_int(search_value);
            ctx.trace_ctx
                .record_guard(OpCode::GuardValue, &[valuebox, expected], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            ctx.trace_ctx.replace_box(valuebox, expected);
            for slot in ctx.registers_i.iter_mut() {
                if *slot == valuebox {
                    *slot = expected;
                }
            }
        }
        return Ok((DispatchOutcome::Continue, target));
    }

    if !valuebox.is_constant() {
        // pyjitpl.py opimpl_switch miss path — emit IntEq +
        // GuardFalse for every key in switchdict (the trace bails out
        // if any subsequent execution lands on a missed key).
        // Box(value) parity: stamp each IntEq result with the
        // (concrete_value == key) bool when valuebox's Box.value
        // resolves an Int.
        let valuebox_concrete = match ctx.trace_ctx.box_value(valuebox) {
            Some(majit_ir::Value::Int(n)) => Some(n),
            _ => None,
        };
        for &key in switchdict.const_keys_in_order() {
            let keybox = ctx.trace_ctx.const_int(key);
            // pyjitpl.py:609-613 `opimpl_switch` miss uses
            // `execute(INT_EQ, ...)` for each key.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::IntEq, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::IntEq, majit_metainterp::counters::RECORDED_OPS);
            let eqbox = ctx.trace_ctx.record_op(OpCode::IntEq, &[valuebox, keybox]);
            if let Some(v) = valuebox_concrete {
                ctx.trace_ctx
                    .set_opref_concrete(eqbox, majit_ir::Value::Int((v == key) as i64));
            }
            ctx.trace_ctx.record_guard(OpCode::GuardFalse, &[eqbox], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Bridge an `MIFrame`'s register banks +
/// trace recorder + last-exc state into a `WalkContext` and run
/// `walk()` against the supplied jitcode body.
///
/// RPython parity context: in RPython the metainterp loop iterates
/// over `metainterp.framestack[-1].pc` calling `bytecode_step` which
/// dispatches to the right `opimpl_*`. There's no separate "walker
/// entry" because the metainterp loop *is* the walker. This entry
/// point is pyre's equivalent: it drives the walker against the
/// supplied `MIFrame` state and is the sole production tracing path.
///
/// Field plumbing:
/// * `registers_r/i/f` — allocated fresh per call sized to
///   `top_num_regs_* + top_constants_*.len()`, then populated by
///   the inline `setup_call` from `argboxes_*` and constant slots.
///   PyPy parity: `pyjitpl.py MIFrame.__init__` allocates
///   the bank vectors at frame construction; `:188 setup_call`
///   populates slots `[0..argboxes.len())` from the caller's
///   argboxes. Walker handlers writing dst slots (`int_copy`,
///   `binop_int_record`, etc.) mutate them in place; the banks are
///   dropped when this function returns (matching PyPy's per-frame
///   lifetime).
/// * `trace_ctx` — borrowed mutably from `miframe.ctx`'s
///   `TraceCtx`. Recording (`record_op`, `finish`, etc.) goes
///   through this.
/// * `last_exc_value` — reads `sym.last_exc_box` as the initial
///   value (`OpRef::NONE` collapses to `None`). On exit the
///   walker's final `last_exc_value` is mirrored back if non-None,
///   so a `raise/r` -> `catch_exception/L` -> handler trace
///   leaves `sym.last_exc_box` pointing at the in-flight exc OpRef
///   (parity with RPython metainterp.last_exc_value).
/// * `descr_refs`, `sub_jitcode_lookup` — caller-provided, same
///   contract as direct `walk()` callers. Production callers wire
///   `crate::jitcode_runtime::all_descrs()` + a JitCode-resolving
///   closure over `crate::jitcode_runtime::all_jitcodes()`.
///
/// `is_top_level` selects the outer-frame semantic:
///
/// * `true` — outermost trace entry. `*_return/*` arms record
///   `Finish(value, done_with_this_frame_descr_<kind>)` and a `raise/r`
///   that is never caught records
///   `Finish(exc, exit_frame_with_exception_descr_ref)`.
/// * `false` — sub-frame entry: `*_return/*` arms surface
///   `SubReturn { result }` and uncaught `raise/r` arms surface
///   `SubRaise { exc }` to the caller. The shadow validator (Phase
///   D-3) drives this for per-Python-opcode arms — a Python-opcode arm
///   compiled by the codewriter ends with `*_return/*` (since each arm
///   is a self-contained sub-jitcode invoked from the outer dispatcher
///   via `inline_call_r_r/dR>r`), so shadow mode must NOT emit a FINISH
///   per Python opcode.
///
/// Sub-walks driven by `inline_call_r_r/dR>r` recursion always set
/// `is_top_level=false` regardless of this caller-side flag (the
/// recursion constructs its own `WalkContext`).
///
/// **Production wiring**: the full-body-walk walker
/// (`full_body_walk_trace`) is the caller, dispatching each JitCode
/// opcode through this entry as it walks the body.
///
/// Bridge exception seeding follows `pyjitpl.py`: only a
/// `ResumeGuardExcDescr`/`ResumeGuardCopiedExcDescr` source guard carries
/// exception state into bridge tracing. Operand-stack values are never scanned
/// to infer a standing exception.
fn seed_standing_exception_for_walk<Sym: WalkSym>(sym: &mut Sym, trace_ctx: &mut TraceCtx) {
    if trace_ctx.is_bridge_trace && !trace_ctx.bridge_source_is_exception_guard() {
        return;
    }

    // `_prepare_exception_resumption` (pyjitpl.py:3125-3126) reads the
    // exception off THIS failure's deadframe (`cpu.grab_exc_value`), and every
    // bridge compile runs on a fresh `MetaInterp`, so a previous compile's
    // `last_exc_value` can never leak into a new bridge trace.  Pyre's sym
    // persists across walks; the routed publish (`BH_LAST_EXC_VALUE`, set from
    // the same grab) is the fresh-failure signal and must OVERWRITE any
    // exception state a previous walk left on the sym.  A preseeded sym is
    // kept only when no fresh signal exists — the multi-frame carrier walk
    // re-seeds per frame after the first frame drained the cell.
    let bh_exc = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
    if bh_exc != 0 {
        let exc = bh_exc as pyre_object::PyObjectRef;
        if !exc.is_null() && unsafe { pyre_object::is_exception(exc) } {
            let exc_box = trace_ctx.const_ref(exc as i64);
            sym.set_current_exc_value(exc);
            sym.set_current_exc_box(exc_box);
            sym.set_last_exc_value(exc);
            sym.set_last_exc_box(exc_box);
            sym.set_class_of_last_exc_is_const(true);
            return;
        }
    }

    if trace_ctx.is_bridge_trace && trace_ctx.bridge_source_is_exception_guard() {
        // Null arm (`pyjitpl.py:3152-3154`): the deadframe published NO
        // exception, and for an exception-guard bridge that publish is the
        // sole authority — `clear_exception()` on the fresh MetaInterp.
        // Keeping a preseeded sym here would walk the exception-flavor
        // continuation for a no-exception failure: a loop whose exception
        // guard was recorded through a raising iteration then compiles the
        // HANDLER as its no-exception bridge, and every non-raising
        // iteration runs the handler body.
        sym.set_current_exc_value(pyre_object::PY_NULL);
        sym.set_current_exc_box(OpRef::NONE);
        sym.set_last_exc_value(pyre_object::PY_NULL);
        sym.set_last_exc_box(OpRef::NONE);
        sym.set_class_of_last_exc_is_const(false);
        return;
    }

    if !sym.last_exc_box().is_none() {
        return;
    }

    let current = pyre_interpreter::eval::get_current_exception();
    if !current.is_null() && unsafe { pyre_object::is_exception(current) } {
        let exc_box = trace_ctx.const_ref(current as i64);
        sym.set_current_exc_value(current);
        sym.set_current_exc_box(exc_box);
        sym.set_last_exc_value(current);
        sym.set_last_exc_box(exc_box);
        sym.set_class_of_last_exc_is_const(true);
    }
}

/// Resolve the top-level walk's live frame code when the promoted `pycode`
/// green has no concrete shadow.
///
/// RPython keeps `MIFrame.jitcode` on each live frame and
/// `opimpl_jit_merge_point` reads the greens from that frame.  The full-body
/// walker has the same per-frame identity in `snapshot_sym.jitcode`; using it
/// here restores the missing concrete shadow without borrowing a caller or
/// portal-global anchor.  Inline sub-walks deliberately do not use this
/// fallback: their code comes from their own `InlineCalleeConsts`, and the
/// existing loop-callee handling below decides whether they may cross a merge
/// point.
fn top_level_live_code<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) -> Option<*const ()> {
    if !ctx.is_top_level || ctx.fbw_mode.snapshot_sym.is_null() {
        return None;
    }
    let sym = unsafe { &*ctx.fbw_mode.snapshot_sym };
    if sym.jitcode().is_null() {
        return None;
    }
    let raw = unsafe { (&(*sym.jitcode()).payload).code_ptr };
    if raw.is_null() {
        None
    } else {
        Some(pyre_interpreter::live_code_wrapper(raw as *const ()) as *const ())
    }
}

fn guard_current_frame_globals_identity<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    expected_globals: pyre_object::PyObjectRef,
) -> Result<bool, DispatchError> {
    if expected_globals.is_null() || majit_gc::can_move(majit_ir::GcRef(expected_globals as usize))
    {
        return Ok(false);
    }
    let Some(w_globals_op) = ctx
        .trace_ctx
        .virtualizable_box_at(VABLE_NAMESPACE_FIELD_IDX)
    else {
        return Ok(false);
    };
    let expected = ctx.trace_ctx.const_ref(expected_globals as i64);
    if w_globals_op.is_constant() {
        return Ok(ctx.trace_ctx.const_value(w_globals_op) == Some(expected_globals as i64));
    }
    // pypy/objspace/std/celldict.py `elidable_promote('0,1,2')` promotes
    // `w_dict`; mirror that by pinning the runtime frame's w_globals identity.
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[w_globals_op, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx.replace_box(w_globals_op, expected);
    for slot in ctx.registers_r.iter_mut() {
        if *slot == w_globals_op {
            *slot = expected;
        }
    }
    Ok(true)
}

fn replace_movable_load_global_namespace_with_frame_globals<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ei: &majit_ir::EffectInfo,
    allboxes: &mut [OpRef],
) {
    if ei.pyre_helper != majit_ir::PyreHelperKind::LoadGlobal {
        return;
    }
    let Some(ns_box) = allboxes.get_mut(1) else {
        return;
    };
    let Some(Value::Ref(majit_ir::GcRef(ns_ptr))) = ctx.trace_ctx.box_value(*ns_box) else {
        return;
    };
    if !majit_gc::can_move(majit_ir::GcRef(ns_ptr)) {
        return;
    }
    if let Some(w_globals_op) = ctx
        .trace_ctx
        .virtualizable_box_at(VABLE_NAMESPACE_FIELD_IDX)
    {
        *ns_box = w_globals_op;
    }
}

/// Line-by-line port of `pyjitpl.py MetaInterp._build_allboxes`.
/// Permutes a flat `argboxes` array (concat of i-list ++ r-list ++ f-list
/// in source order) so positions match the callee's `descr.get_arg_types()`
/// ABI ordering. Returns `[funcbox, ...permuted_argboxes]`.
///
/// RPython operates on a flat `argboxes` of typed `Box` objects + reads
/// `box.type`. The walker has only `OpRef`s, so the type is supplied
/// out-of-band via the `argbox_types` parallel array. By construction
/// the operand decoders (`read_int_var_list` / `read_ref_var_list` /
/// `read_float_var_list`) tag each entry with its bank, so the parallel
/// array is correct without needing a runtime type query.
///
/// The RPython `prepend_box` parameter is unused at every
/// `residual_call*` call site (only `conditional_call*` uses it, not
/// yet ported), so it's omitted from the walker signature. Add it back
/// when porting `opimpl_conditional_call*`.
fn build_allboxes(
    funcbox: OpRef,
    argboxes: &[OpRef],
    argbox_types: &[Type],
    arg_types: &[Type],
) -> Vec<OpRef> {
    debug_assert_eq!(
        argboxes.len(),
        argbox_types.len(),
        "argboxes and argbox_types must align",
    );
    // RPython line 1961: `allboxes = [None] * (len(argboxes)+1 + …)`.
    let total = arg_types.len() + 1;
    let mut allboxes: Vec<OpRef> = Vec::with_capacity(total);
    // RPython line 1966: `allboxes[i] = funcbox`.
    allboxes.push(funcbox);
    // RPython line 1968: `src_i = src_r = src_f = 0`.
    let mut src_i = 0usize;
    let mut src_r = 0usize;
    let mut src_f = 0usize;
    // RPython line 1969-1989: outer `for kind in descr.get_arg_types()`
    // with one type-filter `while True` loop per kind.
    for &kind in arg_types {
        let box_oref = match kind {
            Type::Int => loop {
                // RPython line 1971-1975: advance src_i past non-INT
                // entries until an INT box is found.
                let b = argboxes[src_i];
                let bt = argbox_types[src_i];
                src_i += 1;
                if bt == Type::Int {
                    break b;
                }
            },
            Type::Ref => loop {
                // RPython line 1977-1981.
                let b = argboxes[src_r];
                let bt = argbox_types[src_r];
                src_r += 1;
                if bt == Type::Ref {
                    break b;
                }
            },
            Type::Float => loop {
                // RPython line 1983-1987 (kind == 'L' long-long path
                // not separately modeled — pyre's Type::Float covers
                // both).
                let b = argboxes[src_f];
                let bt = argbox_types[src_f];
                src_f += 1;
                if bt == Type::Float {
                    break b;
                }
            },
            // RPython line 1988-1989: `else: raise AssertionError`.
            // Type::Void in arg_types is an internal invariant violation.
            Type::Void => panic!("_build_allboxes: arg_types must not contain Void"),
        };
        allboxes.push(box_oref);
    }
    debug_assert_eq!(allboxes.len(), total, "allboxes shape post-condition");
    allboxes
}

/// Decode the descr index from a 2-byte LE operand. Companion to
/// [`read_descr`] for callers that need the raw index for error
/// reporting (e.g. `ResidualCallDescrNotCallDescr`).
fn decode_descr_index(code: &[u8], op: &DecodedOp, operand_offset: usize) -> usize {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    lo | (hi << 8)
}

/// `pyjitpl.py heapcache.call_loopinvariant_known_result`
/// short-circuit: when the EffectInfo's extraeffect is `EF_LOOPINVARIANT`
/// AND the heapcache has a cached result for `(descr_index, allboxes[0])`,
/// the trace skips re-recording the `CALL_LOOPINVARIANT_*` op and the
/// caller reuses the cached OpRef.  Returns `None` for non-loopinvariant
/// EI or a cache miss; the caller then falls through to the normal record
/// path and follows up with [`loopinvariant_now_known`] to populate the
/// cache for subsequent matching calls.
///
/// RPython upstream (`heapcache.py`) keys the lookup by descr
/// **identity** and `allboxes[0].getint()`.  Upstream's
/// `do_residual_or_indirect_call` (`pyjitpl.py`) reaches
/// `do_residual_call` for **both** non-`Const` `funcbox` and `Const`
/// funcboxes whose address has no registered jitcode — the
/// `isinstance(funcbox, Const)` guard only short-circuits to
/// `perform_call` when `bytecode_for_address` resolves a jitcode.
/// In the residual path `allboxes[0].getint()` is well-defined
/// regardless of `Const`-ness because every Box subclass exposes
/// `getint()` over its runtime int (`history.BoxInt._value` /
/// `ConstInt.value`).
///
/// pyre's [`TraceCtx::concrete_of_opref`]
/// (`majit-metainterp/src/trace_ctx.rs`) reconstructs the
/// concrete int from the per-trace constant pool only for constant
/// OpRefs; non-constant OpRefs are symbolic at the dispatcher
/// layer and carry no runtime int the trace-time walker can read.
/// When `funcptr` is non-constant we skip the cache entirely —
/// using `funcptr.0` as a sentinel would key on symbolic identity
/// rather than the concrete callee, risking false hits across two
/// different non-const funcptrs that share an OpRef after IR
/// renaming, and false misses across two different OpRefs aliasing
/// the same concrete callee.  Returning `None` is the conservative
/// choice (the caller falls through to record the call); the cost
/// is a missed cache hit on non-const funcptrs that upstream would
/// have caught.  Convergence with upstream's full coverage requires
/// threading concrete-int shadow alongside OpRef for non-const ints
/// — significant work remaining.
///
/// `descr_key` is the descriptor's stable identity key (`Descr::index()`),
/// matching upstream's identity comparison on `descr` more closely than
/// the operand-encoded descr-table slot.
#[inline]
fn loopinvariant_lookup<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    ei: &majit_ir::EffectInfo,
    descr_key: u32,
    funcptr: OpRef,
) -> Option<OpRef> {
    if ei.extraeffect != majit_ir::ExtraEffect::LoopInvariant {
        return None;
    }
    let arg0_int = funcptr_concrete_int(ctx, funcptr)?;
    ctx.trace_ctx
        .heap_cache()
        .call_loopinvariant_known_result(descr_key, arg0_int)
        .map(|(opref, _resvalue)| opref)
}

/// `pyjitpl.py heapcache.call_loopinvariant_now_known`: after
/// recording a fresh `CALL_LOOPINVARIANT_*` op, remember the
/// `(descr_index, allboxes[0].getint()) -> result` mapping so the
/// next matching call short-circuits via [`loopinvariant_lookup`].
/// No-op for non-loopinvariant EI, and no-op when `funcptr` is
/// non-constant (no concrete int key — see [`loopinvariant_lookup`]).
///
/// `resvalue` is stored as `0`.  RPython's upstream caller
/// (`pyjitpl.py`) stores `res` — the concrete value returned
/// by `execute_varargs` after actually running the callee.  pyre-
/// jit-trace records symbolically without executing, so no concrete
/// result exists at this point: `record_op_with_descr` returns a
/// freshly-minted SSA OpRef whose runtime value is only known when
/// the compiled trace later executes.  The cached `_resvalue` is
/// unused by [`loopinvariant_lookup`]'s consumer (only the symbolic
/// OpRef is read for register writeback), so the `0` placeholder is
/// observationally equivalent.  Convergence with upstream requires
/// either threading the concrete result up from the executing trace
/// (concrete shadow tracking) or dropping the field from
/// the cache shape entirely (separate cleanup).
#[inline]
fn loopinvariant_now_known<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ei: &majit_ir::EffectInfo,
    descr_key: u32,
    funcptr: OpRef,
    result: OpRef,
) {
    if ei.extraeffect != majit_ir::ExtraEffect::LoopInvariant {
        return;
    }
    let Some(arg0_int) = funcptr_concrete_int(ctx, funcptr) else {
        return;
    };
    ctx.trace_ctx
        .heap_cache_mut()
        .call_loopinvariant_now_known(descr_key, arg0_int, result, 0);
}

/// Resolve a residual-call funcptr OpRef to the concrete function
/// pointer integer that RPython's heapcache keys on
/// (`heapcache.py` calls `allboxes[0].getint()`).
///
/// Returns `Some(int)` when `funcptr` is a constant int OpRef whose
/// value lives in pyre's per-trace constant pool. For the RPython-legal
/// `EF_LOOPINVARIANT` direct-call producer, `call.py` asserts
/// no runtime args and emits a constant function box; that is the path
/// this cache is meant to mirror. General residual calls can arrive
/// from indirect calls with non-constant funcboxes
/// (`pyjitpl.py`), so `None` means "skip the loop-invariant
/// cache" rather than inventing an alias-prone sentinel key.
#[inline]
fn funcptr_concrete_int<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    funcptr: OpRef,
) -> Option<i64> {
    if !funcptr.is_constant() {
        return None;
    }
    match ctx.trace_ctx.concrete_of_opref(funcptr) {
        Some(majit_ir::Value::Int(v)) => Some(v),
        _ => None,
    }
}

/// Returns `true` when the jitcode body contains any `catch_exception/L`
/// op — i.e. the source function has a `try`/`except` handler.  Used by
/// the residual-call fast paths that conservatively decline a handler-
/// bearing body to the generic walk (which resumes a `GUARD_NO_EXCEPTION`
/// deopt into the handler correctly) rather than to their concrete fold.
fn jitcode_has_exception_handler(code: &[u8]) -> bool {
    crate::jitcode_runtime::decoded_ops(code).any(|op| op.opname == "catch_exception")
}

/// Maps a freshly-boxed `W_Bool` opref (the `jit_bool_value_from_truth(t)`
/// result a compare specialization writes to the value-stack slot) back to its
/// raw truth Int opref `t` (#62).  The `COMPARE_OP` specialization boxes the
/// `int_lt`/… result because the generic `compare` helper returns a Ref, but
/// the immediately-following `POP_JUMP_IF_*` lowers to an `is_true` residual
/// (`residual_call_r_i`) that unboxes it straight back to an Int for the
/// branch. That box→stack→unbox round-trip was absent from the retired
/// MIFrame path, which branched on the raw compare result. When the `is_true`
/// residual's Ref arg is a mapped boxed bool we fold its result to the raw
/// truth Int (bool→int is value-preserving), eliding the may-force unbox; the
/// now-dead box + stack store are then DCE'd by the optimizer.
///
/// pyre-only side table, NOT an upstream structure: `jtransform.py`
/// `optimize_goto_if_not` fuses compare+branch at codewriter (graph-rewrite)
/// time — it removes the compare op (`block.operations.remove(op)`) and folds
/// it into `block.exitswitch`, so PyPy never materializes a boxed bool or an
/// `is_true` unbox and needs no runtime side table.  The full-body walker
/// consumes a JitCode that did NOT receive that fusion (the `COMPARE_OP`
/// specialization boxes the result; `POP_JUMP_IF_*` lowers to a separate
/// `is_true` residual), so this is a RUNTIME reconstruction of that STATIC
/// fusion.  Read+write are both gated on `WalkContext::is_authoritative_executor`
/// (true only inside the two FBW walk entry points) and the map is cleared at
/// every walk boundary by `bool_box_truth_reset` — see there — so it cannot leak
/// across traces; OpRef SSA-uniqueness (`recorder.rs`) keeps a key from ever
/// re-binding within one walk.
thread_local! {
    static BOOL_BOX_TRUTH: std::cell::RefCell<Vec<(OpRef, OpRef)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn bool_box_truth_record(boxed: OpRef, truth: OpRef) {
    BOOL_BOX_TRUTH.with(|m| m.borrow_mut().push((boxed, truth)));
}

/// If `boxed` is a recorded freshly-boxed bool, return its raw truth Int opref.
fn bool_box_truth_lookup(boxed: OpRef) -> Option<OpRef> {
    BOOL_BOX_TRUTH.with(|m| {
        m.borrow()
            .iter()
            .rev()
            .find(|(b, _)| *b == boxed)
            .map(|(_, t)| *t)
    })
}

/// Clear the [`BOOL_BOX_TRUTH`] map at the start of an authoritative walk so a
/// prior aborted walk's entries never leak into the next one.  This is the
/// reset boundary for the walk-local thread-local; it is called at the two FBW
/// walk entry points (`trace.rs` `full_body_walk_trace` at walk start, and
/// after `probe_walk_perfn_jitcode` discards its throwaway trace).
pub fn bool_box_truth_reset() {
    BOOL_BOX_TRUTH.with(|m| m.borrow_mut().clear());
}

/// PyPy `_opimpl_residual_call{1,2,3}` (pyjitpl.py) port
/// for the **non-elidable** call shapes — companion to
/// [`try_fold_pure_call_via_executor`] which handles the elidable case.
///
/// Upstream calls `executor.execute_varargs(opnum, argboxes, descr,
/// exc=can_raise, pure=False)` which dispatches through
/// `cpu.bh_call_*` and clears/records BH exception state via
/// `metainterp.clear_exception()` + `metainterp.execute_raised(...)`.
/// This walker-friendly counterpart returns `Result<i64, i64>` directly
/// (via [`majit_metainterp::executor::execute_residual_call`]) so the
/// caller can wire the BH exception into `WalkContext.last_exc_value`
/// without dragging in a `MetaInterp` seam.
///
/// **Why this exists**: the walk is the sole execution leg for the
/// traced iteration — the synchronous marker walk returns its terminal
/// outcome without re-running `execute_opcode_step`. For opcodes whose body contains a
/// non-elidable `residual_call_*` (`store_subscr_fn` /
/// `set_current_exception` / etc.), the helper is never invoked → heap
/// mutation never happens → next read derefs stale container → SIGBUS
/// (5 STORE_SUBSCR-hot benches).  The
/// orthodox fix is to widen this function across `Call*` /
/// `CallLoopinvariant*` / `CallMayForce*` shapes, mirroring PyPy's
/// `_opimpl_residual_call*` which concrete-executes *every* residual
/// call regardless of EI.
///
/// **Caller contract**:
/// * `call_opcode` must be one of the non-elidable residual call shapes:
///   `Call{I,R,F,N}` (the default arm), `CallLoopinvariant{I,R,F,N}`
///   (the `EF_LOOPINVARIANT` arm), or `CallMayForce{I,R,F,N}` (the
///   `forces_virtual_or_virtualizable` arm — only when no active
///   virtualizable is present, see force-virtual gate below).
///   * `CallPure*` is excluded — that's [`try_fold_pure_call_via_executor`]'s
///     job.
///   * `CallReleaseGil*` / `CallAssembler*` are excluded from THIS
///     function's direct opcode set, but for distinct reasons:
///     - `CallReleaseGil*` IS concrete-executed — `do_residual_call`
///       (`pyjitpl.py`) runs step 2 with `opnum1 =
///       CALL_MAY_FORCE_*` for the *whole* forces branch, and release-gil
///       is a sub-case inside it.  [`direct_call_release_gil`] records the
///       `CALL_RELEASE_GIL_*` op then re-enters this function with a
///       `CallMayForce*` opcode and the original `allboxes`, so the GIL
///       wrapper runs once during tracing exactly like a may-force call
///       (the GIL transition is inert single-threaded during recording).
///     - `CallAssembler*` is correctly NOT executed here: an assembler
///       token re-enters the JIT, it is not a plain C func address, so the
///       raw-`func_ptr` executor would fault.  Recursive `CALL_ASSEMBLER`
///       is folded by [`try_walker_call_assembler_self_recursive`], which
///       concrete-executes the callee through its own `CallMayForceR` path
///       on the real callable (token lives only in the recorded op).
///   * **`CallMayForce*` vable token protocol**: PyPy `do_residual_call`
///     (`pyjitpl.py`) concrete-executes every `CallMayForce*`
///     via `executor.execute_varargs`, bracketed by the heap halves of
///     the token protocol: `vinfo.tracing_before_residual_call(virtualizable)`
///     (pyjitpl.py, sets `TOKEN_TRACING_RESCALL`) before the
///     call and `vinfo.tracing_after_residual_call(virtualizable)`
///     (pyjitpl.py) after it.  This function mirrors both
///     halves around `execute_residual_call` whenever the jitdriver has
///     an active `standard_virtualizable_box()` with a live heap pointer
///     — the bracket `vable_and_vrefs_before_residual_call` /
///     `vable_after_residual_call` describes upstream.  A
///     cleared token after the call means the callee forced the
///     virtualizable: surface [`DispatchError::VableEscapedDuringResidualCall`]
///     (`SwitchToBlackhole(ABORT_ESCAPE)` parity, pyjitpl.py).
///     With no active vable the bracket is skipped — nothing to force.
///     The IR half (`FORCE_TOKEN` + `SETFIELD_GC(vable_token_descr)`)
///     stays at the dispatcher's
///     [`maybe_walker_vable_and_vrefs_before_residual_call`] call site.
/// * `allboxes[0]` is the funcbox (per `build_allboxes` layout); the
///   remaining slots are user args in `descr.arg_types()` ABI order.
///
/// **Authoritative-executor gate**: fires ONLY when the walk is the
/// sole concrete-execution leg
/// ([`WalkContext::is_authoritative_executor`]) — the production
/// full-body walk and its inline sub-walks qualify
/// (`eval_loop_jit` skips `execute_opcode_step` for walker-handled
/// opcodes).  In shadow / diagnostic-probe mode the flag is `false`, so
/// the call is recorded symbolically only — re-executing there would
/// double the side effects (the concrete interpreter already ran it)
/// or corrupt the live heap under the discard-the-trace probe.
///
/// **Return value**:
/// * `Ok(ResidualExecOutcome::Executed(Ok(_)))` — helper executed normally, `recorded` OpRef
///   stamped with the concrete result.
/// * `Ok(ResidualExecOutcome::Executed(Err(bh_exc)))` — helper raised; `bh_exc` is the wrapped
///   `PyError` pointer (from `BH_LAST_EXC_VALUE`).  Caller is
///   responsible for routing into `WalkContext.last_exc_value` so the
///   downstream `GuardNoException` walker handler picks it up; the
///   `recorded` OpRef is NOT stamped (no concrete result).
/// * `Ok(ResidualExecOutcome::Declined(_))` — fold declined (preconditions not met: not the
///   authoritative executor, opcode out of set, funcbox non-const, arity
///   exceeds [`MAX_HOST_CALL_ARITY`], any operand lacks a concrete
///   `box_value`, or any Ref arg is NULL), or the call's result is void
///   (recorded symbolically; the compiled loop applies the side effect —
///   see the void branch below).  The trace still has the recorded call
///   op for the optimizer to consume later; walker falls through as if
///   this function did not exist.
/// * `Err(VableEscapedDuringResidualCall)` — the callee forced the
///   active virtualizable during a `CallMayForce*`; the walk must abort
///   (pyjitpl.py ABORT_ESCAPE parity, see the token-protocol bullet
///   above).
///
/// **Wire status**: invoked from all three
/// dispatch entry points (`dispatch_residual_call_iRd_kind`,
/// `dispatch_residual_call_iIRd_kind`, `dispatch_residual_call_iIRFd_kind`)
/// alongside [`try_fold_pure_call_via_executor`].  The may-force /
/// can-raise path wires the `Err` exception through
/// `WalkContext.last_exc_value`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidualDecline {
    ValueUnavailable,
    Symbolic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidualExecOutcome {
    Executed(Result<i64, i64>),
    Declined(ResidualDecline),
}

/// `pyjitpl.py` short-circuit guard.  RPython `do_residual_call`
/// runs `_do_jit_force_virtual` (`pyjitpl.py`) when
/// `effectinfo.oopspecindex == OS_JIT_FORCE_VIRTUAL`:
///
/// ```text
/// def _do_jit_force_virtual(self, allboxes):
///     if (self.jitdriver_sd.virtualizable_info is None and
///         self.jitdriver_sd.greenfield_info is None):
///         return None
///     if len(allboxes) == 2:
///         [vrefbox] = allboxes[1:]
///         standard_box = self.virtualizable_boxes[-1].getref_base()
///         if standard_box != vrefbox.getref_base():    # concrete pointer compare
///             return None
///         return self.virtualizable_boxes[-1]
///     ...
/// ```
///
/// PyPy returns one of `vref_box` / `standard_box` / `None`; on the
/// `None` fall-through it records the normal `CALL_MAY_FORCE_*`.  The
/// walker CANNOT reproduce this faithfully because the comparison
/// `standard_box.getref_base() != vrefbox.getref_base()` requires a
/// concrete `*mut PyObject` for the `vrefbox` Ref OpRef; pyre's
/// symbolic walker only carries `concrete_vable_ptr` for the active
/// virtualizable, not a per-OpRef → concrete-pointer map.
///
/// Choices considered:
///  1. Silent fall-through (always record `CALL_MAY_FORCE_*`) — would
///     silently emit IR on a path PyPy folds away whenever the vref
///     IS the standard virtualizable.  Trace divergence with no diff
///     report; rejected.
///  2. OpRef-equality short-circuit (`if vrefbox_opref == standard_box_opref:
///     short-circuit; else fall through`) — sufficient for the same-OpRef
///     case but UNSOUND for the different-OpRef-same-concrete-pointer case
///     (walker would record `CALL_MAY_FORCE_*` while PyPy folds; mismatch
///     vs. live tracer's IR).  Rejected.
///  3. Fail-loud (current).  STRICTER than PyPy: the walker stops with a
///     typed error rather than emit divergent IR, which immediately
///     flags any producer that starts emitting
///     `OopSpecIndex::JitForceVirtual`.
///
/// Convergence path back to 1:1 PyPy parity: an OpRef → concrete-pointer
/// resolver.  When that lands the guard becomes a real
/// `_do_jit_force_virtual()` body that returns `Some(vref_opref)` /
/// `Some(standard_opref)` / `None` and threads through the dispatcher
/// like the release-gil short-circuit does today.
///
/// Production reach today: zero — `OopSpecIndex::JitForceVirtual` is
/// set only by `jtransform.rs jit.force_virtual` lowering, which
/// our benchmarks don't reach.  The guard is fail-loud futureproofing
/// for the day a producer (e.g. an explicit `jit.force_virtual` callee
/// from `pyre_interpreter`) lights up the path.
#[inline]
fn do_jit_force_virtual_guard(ei: &majit_ir::EffectInfo, pc: usize) -> Result<(), DispatchError> {
    if ei.oopspecindex == OopSpecIndex::JitForceVirtual {
        return Err(DispatchError::JitForceVirtualRequiresConcreteResolver { pc });
    }
    Ok(())
}

/// `pyjitpl.py _get_list_of_active_boxes` parity for the
/// walker-emitted snapshot: read each live register from its
/// kind-specific bank in (int, ref, float) order, dropping non-live
/// slots regardless of whether the OpRef happens to be set.  The
/// liveness lookup matches the decoder side
/// (`state::frame_value_count_at` /
/// `frame_liveness_reg_indices_by_bank_at`), so encoder and decoder
/// agree byte-for-byte on the snapshot shape consumed at resume.
///
/// Returns an empty vector when no liveness is registered for the
/// `(jitcode_index, pc)` pair (skeleton payload or out-of-range PC);
/// the downstream optimizer surfaces the empty snapshot as a no-op.
/// `true` when `op` is a `ConstPtr` carrying the NULL gcref (`GcRef(0)`) —
/// the encoding an unset vable shadow slot decodes to.  Distinct from
/// `OpRef::NONE` (no value at all).
fn opref_is_null_const_ptr(op: OpRef) -> bool {
    op.as_const_ptr().is_some_and(|g| g.is_null())
}

/// The source selected for `collect_outer_active_boxes` entry metadata.  The
/// audit compares the JitCode-PC variants to the legacy Python-PC tables.
#[derive(Clone, Copy)]
enum OuterActiveBoxesEntryTwin {
    Plain,
    Trivia,
}

impl OuterActiveBoxesEntryTwin {
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Trivia => "trivia",
        }
    }
}

fn collect_outer_active_boxes<Sym: WalkSym>(
    sym: &Sym,
    trace_ctx: &mut TraceCtx,
    regs_i: &[OpRef],
    regs_r: &[OpRef],
    regs_f: &[OpRef],
    outer_jitcode_index: u32,
    guard_present: bool,
    // The resume-carried coordinate remains the bank-liveness source. Entry
    // metadata can be derived from an earlier raw operation coordinate, so it
    // must travel independently rather than inheriting that marker word.
    carried_jitcode_pc: i32,
    entry_jitcode_pc: i32,
    entry_twin: OuterActiveBoxesEntryTwin,
    entry_caller: &'static str,
    vstack: Option<&[OpRef]>,
    kept_recovered: &[(u16, OpRef)],
) -> Vec<OpRef> {
    // `#124` Approach B: resolve the base live-box set through the carried
    // JitCode coordinate so the encoder's color set matches the decoder's
    // (`setup_bridge_sym` / `rebuild_inline_callee`), which read the same
    // carried word.  With the flag off `carried_jitcode_pc` is ignored and
    // this is the plain py_pc→jitcode-translation query.
    let banks = crate::state::frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
        outer_jitcode_index as i32,
        carried_jitcode_pc,
    );
    let mut active = Vec::with_capacity(banks.int.len() + banks.ref_.len() + banks.float.len());
    // RPython `pyjitpl.py _get_list_of_active_boxes` reads
    // `self.registers_X[index]` directly per liveness index.  Pyre
    // diverges on the Ref bank for portal-owner frames: the
    // `registers_r` semantic-mirror write in
    // `write_stack_slot` (trace_opcode.rs) is retired so stack-slot colors
    // sit at `OpRef::NONE` in `sym.registers_r`; the authoritative
    // shadow lives in `trace_ctx.virtualizable_boxes`.  Codewriter
    // liveness also force-alives `portal_frame_reg` / `portal_ec_reg`
    // (codewriter.rs `filter_liveness_in_place`) — these are
    // scratch colors past `nlocals + max_stackdepth` that have no
    // semantic frame slot, sourced instead from `sym.frame` /
    // `sym.execution_context` (`interp_jit.py reds = ['frame', 'ec']`).
    //
    // Mirror the retired MIFrame snapshot materializer
    // (trace_opcode.rs `get_list_of_active_boxes`): map each
    // live Ref color to its semantic index via
    // `semantic_ref_slot_for_reg_color`, read the vable shadow for
    // portal-owner frames, and route the two portal red regs through
    // `sym.frame` / `sym.execution_context` directly.
    let (nlocals, valid_stack_only, owns_vable, portal_frame_reg, portal_ec_reg) =
        if sym.jitcode().is_null() {
            (0usize, 0usize, false, u16::MAX, u16::MAX)
        } else {
            unsafe {
                let jc = &*sym.jitcode();
                let payload = &jc.payload;
                // Operand-stack depth at the snapshot coordinate. The liveness
                // banks (`frame_liveness_reg_indices_by_bank_at`) are read at
                // that coordinate too, so the per-PC color→slot window
                // (`pcdep_opt`'s stack clamp below) must use its depth — NOT
                // `sym.valuestackdepth` (the walker's *current* position). For
                // the per-opcode entry caller the two coincide, but a guard
                // resuming at a not-taken branch target with a kept operand-stack
                // temp (conditional expr / short-circuit / chained compare,
                // #124/#281) resumes at a depth `> 0` while the walker stands at
                // depth 0 — using the current depth there drops the kept temp's
                // semantic slot, corrupting the frame.
                let stack_depth_at_pc = if payload.code_ptr.is_null() {
                    0usize
                } else {
                    match (entry_twin, entry_jitcode_pc >= 0) {
                        (OuterActiveBoxesEntryTwin::Plain, true) => payload
                            .depth_for_jitcode_pc_pred(entry_jitcode_pc as usize)
                            .unwrap_or(0)
                            as usize,
                        (OuterActiveBoxesEntryTwin::Trivia, true) => payload
                            .depth_trivia_for_jitcode_pc(entry_jitcode_pc as usize)
                            .unwrap_or(0)
                            as usize,
                        // Defensive default for a genuinely absent entry
                        // coordinate (marker miss); 0-fire across the corpus.
                        _ => 0,
                    }
                };
                (
                    sym.nlocals(),
                    stack_depth_at_pc,
                    sym.owns_virtualizable_shadow(),
                    payload.metadata.portal_frame_reg,
                    payload.metadata.portal_ec_reg,
                )
            }
        };
    // #348: per-snapshot-coordinate color→slot entries — the color→slot source
    // the inversions below consult for every drained (non-portal) jitcode, the
    // per-program-point color space the `-live-` markers carry. Branch-guard
    // resumes (`guard_present`) are fully covered here; a per-opcode-entry
    // resume whose live Ref colors are all constants/leaked carries no entry (the
    // resume snapshot records Variables only), so `semantic_ref_slot_for_reg_color`
    // returns `None` and the live color falls to the `regs_r[color]` walk-bank
    // read below. Portal-bridge frames carry no `pcdep` entry (empty by install),
    // so for them `pcdep_opt` is `None`, `semantic_ref_slot_for_reg_color`
    // returns `None`, and every live color falls to the `regs_r[color]`
    // walk-bank read.
    let pcdep_entries: Vec<(u8, u16, u16)> = if sym.jitcode().is_null() {
        Vec::new()
    } else {
        unsafe {
            let jc = &*sym.jitcode();
            match (entry_twin, entry_jitcode_pc >= 0) {
                (OuterActiveBoxesEntryTwin::Plain, true) => jc
                    .payload
                    .pcdep_for_jitcode_pc(entry_jitcode_pc as usize)
                    .unwrap_or_default(),
                (OuterActiveBoxesEntryTwin::Trivia, true) => jc
                    .payload
                    .pcdep_trivia_for_jitcode_pc(entry_jitcode_pc as usize)
                    .map(ToOwned::to_owned)
                    .unwrap_or_default(),
                // Defensive default for a genuinely absent entry coordinate
                // (marker miss); 0-fire across the corpus.
                _ => Vec::new(),
            }
        }
    };
    let pcdep_opt: Option<&[(u8, u16, u16)]> =
        (!pcdep_entries.is_empty()).then(|| pcdep_entries.as_slice());
    let (guard_pcdep_entries, guard_stack_only) = if guard_present {
        if sym.jitcode().is_null() {
            (Vec::new(), 0usize)
        } else {
            unsafe {
                let jc = &*sym.jitcode();
                let cjc = carried_jitcode_pc;
                let entries = if cjc >= 0 {
                    jc.payload
                        .pcdep_for_jitcode_pc(cjc as usize)
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                let depth = if jc.payload.code_ptr.is_null() {
                    0usize
                } else if cjc >= 0 {
                    jc.payload
                        .depth_for_jitcode_pc_pred(cjc as usize)
                        .unwrap_or(0) as usize
                } else {
                    0
                };
                (entries, depth)
            }
        }
    } else {
        (Vec::new(), 0usize)
    };
    // Int / Float bank diagnostic panic: pyre's banks are sized to the
    // jitcode's `num_regs_X`, which the codewriter co-publishes with the
    // liveness side-table, so every liveness index is in range by
    // construction.  A miss here is a tracer-side invariant violation
    // (size mismatch) — panic loudly so the bug surfaces at the encode
    // site instead of bleeding NONE values into `encode_snapshot_boxes`
    // where `get_opref_type(NONE)` panics with no breadcrumb pointing at
    // the source.
    let live_i = banks.int.clone();
    let live_r = banks.ref_.clone();
    let live_f = banks.float.clone();
    let (ni, nr, nf) = (regs_i.len(), regs_r.len(), regs_f.len());
    let vable_len = trace_ctx.virtualizable_boxes_len().unwrap_or(0);
    // Int / Float bank candidates: pyre's vable static fields decode as
    // Int (last_instr, valuestackdepth, etc.); if liveness expects an Int
    // register that the register banks did not fill, the candidate fill is
    // one of these sym shadow OpRefs.  Including them in the diagnostic
    // lets fallback work jump straight to the right source.
    let vable_vsd = sym.vable_valuestackdepth();
    let vable_last_instr = sym.vable_last_instr();
    let dump_ctx = |bank: &'static str, reg_idx: u32| -> String {
        let int_hint = if bank == "int" {
            format!(
                ", sym.vable_valuestackdepth={vable_vsd:?}, sym.vable_last_instr={vable_last_instr:?}"
            )
        } else {
            String::new()
        };
        format!(
            "collect_outer_active_boxes: liveness-active {bank} \
             register {reg_idx} holds OpRef::NONE \
             (outer_jitcode_index={outer_jitcode_index}, entry_jitcode_pc={entry_jitcode_pc}, \
              nlocals={nlocals}, owns_vable={owns_vable}, \
              vable_len={vable_len}, \
              num_regs_i={ni}, num_regs_r={nr}, num_regs_f={nf}, \
              live_banks_i={live_i:?}, live_banks_r={live_r:?}, live_banks_f={live_f:?}\
              {int_hint})",
        )
    };
    for &idx in &banks.int {
        let v = regs_i
            .get(idx as usize)
            .copied()
            .unwrap_or_else(|| panic!("{}", dump_ctx("int", idx)));
        if v == OpRef::NONE {
            panic!("{}", dump_ctx("int", idx));
        }
        active.push(v);
    }
    // #420: exact not-taken edge-move recovery, keyed by resume-merge color.
    // The guard trampoline's `ref_copy(dst <- src)` gave this kept slot's live
    // guard-pc source value directly, exact for any kept-stack depth.
    let kept_recovered: std::collections::HashMap<u32, OpRef> = kept_recovered
        .iter()
        .map(|&(dst, v)| (dst as u32, v))
        .collect();
    // Mirror-sourced kept operand-stack slots: at a branch guard the
    // not-taken arm preserves the operand-stack bottom, so the live walk-level
    // box mirror (`ctx.vstack_boxes`, indexed by absolute operand-stack depth)
    // holds the exact kept value for resume operand slot `s`.  This replaces
    // the stale `registers_r[merge_color]` / edge-recovery color heuristics
    // below, which read a color the regalloc reused between the guard pc and
    // the resume pc (the #424 merge-color-staleness corruption).  Scoped to
    // the branch-guard reconstruction (`guard_present`); the merge-color
    // heuristics below remain only as the fallback for slots the mirror does
    // not cover (mirror invalid, or an Int-bank temp the Ref-only mirror does
    // not hold).
    let vstack_mirror: Option<&[OpRef]> = vstack.filter(|_| guard_present);
    for &idx in &banks.ref_ {
        let color = idx as usize;
        if let Some(mirror) = vstack_mirror {
            // Resume operand-stack slot for this live Ref color; the mirror
            // box for that slot is the kept value (bottom-anchored: resume
            // slot `s` == `vstack_boxes[s]`, the same mapping `mirror_covers_
            // kept` validates).
            if let Some(sem) = crate::state::semantic_ref_slot_for_reg_color(
                nlocals,
                valid_stack_only,
                pcdep_opt.unwrap_or(&[]),
                color,
            ) {
                if sem >= nlocals {
                    if let Some(&m) = mirror.get(sem - nlocals) {
                        if m != OpRef::NONE && !opref_is_null_const_ptr(m) {
                            active.push(m);
                            continue;
                        }
                    }
                }
            }
        }
        if let Some(&rv) = kept_recovered.get(&idx) {
            // Edge-move-resolved kept operand; overrides the unwritten
            // merge-color read for this not-taken-arm operand-stack slot.
            active.push(rv);
            continue;
        }
        let fallback = || {
            regs_r
                .get(color)
                .copied()
                .unwrap_or_else(|| panic!("{}", dump_ctx("ref", idx)))
        };
        let semantic_idx = crate::state::semantic_ref_slot_for_reg_color(
            nlocals,
            valid_stack_only,
            pcdep_opt.unwrap_or(&[]),
            color,
        );
        // Portal-red routing applies only to the force-alived SCRATCH case
        // (`filter_liveness_in_place` keeps `portal_frame_reg`/`portal_ec_reg`
        // in every `-live-` R-bank even where the color names no frame slot).
        // If the walk's live register bank already carries a real box, snapshot
        // that box first: it is the direct `get_list_of_active_boxes`
        // (`pyjitpl.py`) source for this live color, and bridge resume uses
        // the same color-indexed bank to recover the EC red (`state.rs`).
        // Fall back to the named red field only when the bank is empty.  If both
        // are empty, the force-alived red is dead at this capture point, so encode
        // `CONST_NULL` (history.py), matching the union-liveness arm below.
        //
        // The jitcode register allocator ALSO assigns these colors to real frame
        // slots at other PCs (e.g. a call-result register live across a later
        // call), where `pcdep_color_slots` maps the color to a semantic slot.
        // Routing such a colliding color to `sym.frame`/`sym.execution_context`
        // encodes the wrong box; a color that names a live semantic slot therefore
        // takes the normal slot-value paths below.
        let is_portal_red_scratch = semantic_idx.is_none()
            && ((color as u16 == portal_frame_reg && portal_frame_reg != u16::MAX)
                || (color as u16 == portal_ec_reg && portal_ec_reg != u16::MAX));
        let value = if is_portal_red_scratch {
            let live_reg = regs_r
                .get(color)
                .copied()
                .filter(|&v| v != OpRef::NONE && !opref_is_null_const_ptr(v));
            let red_field = if color as u16 == portal_frame_reg {
                sym.frame()
            } else if !sym.execution_context().is_none() {
                // EC red already seeded on this snapshot path.
                sym.execution_context()
            } else if !sym.frame().is_none() {
                // Adapter / inline-caller snapshot path leaves
                // `sym.execution_context` unseeded (`OpRef::NONE`).  This is the
                // pre-guard inline-parent-frame collection (the paused caller's
                // active boxes are built BEFORE the callee sub-walk records its
                // guards), so recording the recovery getfield here is
                // well-ordered. Recover the EC from the frame the same way the
                // `MIFrame::ensure_execution_context` does
                // (trace_opcode.rs): record `getfield
                // frame.execution_context` and route that OpRef through as the
                // portal EC red, so the resume snapshot never pushes NONE for
                // `interp_jit.py reds = ['frame', 'ec']`.  A NONE EC escapes
                // as a null execution-context pointer and SIGSEGVs (rc=139) or
                // trips the Ref-bank NONE guard (rc=101).
                //
                // The post-guard snapshot-capture path
                // (`walker_capture_snapshot_for_last_guard`) reaches this fn with
                // the outer full-body `sym`, whose EC is eagerly recovered at
                // walk entry (`seed_execution_context_for_walk`) — so on that
                // path `sym.execution_context` is already real above and this
                // branch (which would record AFTER the guard, a use-before-def)
                // is not taken.
                trace_ctx.record_op_with_descr(
                    OpCode::GetfieldGcR,
                    &[sym.frame()],
                    crate::descr::pyframe_execution_context_descr(),
                )
            } else {
                // Neither EC nor frame is recoverable: keep the raw NONE so the
                // downstream Ref-bank NONE guard surfaces the unrecoverable case
                // instead of silently masking it.
                sym.execution_context()
            };
            live_reg
                .or_else(|| {
                    (red_field != OpRef::NONE && !opref_is_null_const_ptr(red_field))
                        .then_some(red_field)
                })
                .unwrap_or_else(|| OpRef::const_ptr(majit_ir::GcRef(0)))
        } else if owns_vable {
            match semantic_idx {
                Some(s_idx) if s_idx < nlocals + valid_stack_only => {
                    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                    let vbox = trace_ctx.virtualizable_box_at(nvs + s_idx);
                    let walk_box = regs_r.get(color).copied();
                    if s_idx >= nlocals {
                        // Operand-stack slot.  `pyjitpl.py` snapshots
                        // `self.registers_r[index]`, but pyre's stack
                        // `write_ref_reg` mirror was retired: outside a
                        // branch guard's own per-PC color map, the walk
                        // register may hold a stale value from a prior SSA def
                        // that shared the color.  Prefer the live register
                        // (the upstream source) when the guard PC's
                        // `pcdep_color_slots` proves this color owns the same
                        // stack slot at the guard capture point — there the
                        // register read means exactly `registers_r[index]`;
                        // where ownership is unprovable the virtualizable
                        // shadow remains authoritative.
                        let shadow_is_real = vbox.is_some_and(|b| !opref_is_null_const_ptr(b));
                        let walk_real =
                            walk_box.filter(|&v| v != OpRef::NONE && !opref_is_null_const_ptr(v));
                        let guard_pc_proves_slot = guard_present
                            && crate::state::semantic_ref_slot_for_reg_color(
                                nlocals,
                                guard_stack_only,
                                &guard_pcdep_entries,
                                color,
                            ) == Some(s_idx);
                        if guard_pc_proves_slot {
                            walk_real.or(vbox).unwrap_or_else(fallback)
                        } else if shadow_is_real {
                            vbox.unwrap_or_else(fallback)
                        } else {
                            walk_real.unwrap_or_else(|| vbox.unwrap_or_else(fallback))
                        }
                    } else {
                        // At a branch guard the walk register is the live
                        // `MIFrame.registers_r[index]` binding captured by
                        // `_get_list_of_active_boxes`.  The virtualizable
                        // shadow can still hold the loop-entry value for a
                        // local overwritten earlier in this arm, so prefer the
                        // guard-state register and retain the shadow as the
                        // fallback.  Non-branch captures keep the shadow-first
                        // portal-local path.
                        let walk_is_real = walk_box
                            .is_some_and(|b| b != OpRef::NONE && !opref_is_null_const_ptr(b));
                        let shadow_is_real = vbox.is_some_and(|b| !opref_is_null_const_ptr(b));
                        if guard_present && walk_is_real {
                            walk_box.unwrap_or_else(fallback)
                        } else if shadow_is_real {
                            vbox.unwrap_or_else(fallback)
                        } else {
                            match walk_box {
                                Some(v) if v != OpRef::NONE && !opref_is_null_const_ptr(v) => v,
                                _ => vbox.unwrap_or_else(fallback),
                            }
                        }
                    }
                }
                // `semantic_idx` is `None`: this Ref color names no live
                // frame slot at the resume PC.  Under the canonical splice a
                // single `-live-` marker is SHARED across a range of Python
                // PCs at different stack depths, and its liveness banks carry
                // the UNION of those PCs' live colors.  Resuming at a
                // shallower PC therefore sees colors that are dead here:
                //   - a mapped frame stack slot BEYOND this resume's live
                //     window (`semantic_ref_slot_for_reg_color` is `None`
                //     because the slot sits past `valid_stack_only`), or
                //   - under free (non-identity) coloring, a NON-frame SSA
                //     temp (in neither the local nor the stack color map)
                //     that is live only at another PC the marker spans.
                // Both are dead at this PC: the trace produced no box for a
                // dead value, so `registers_r[color]` is NONE.  Both decode
                // paths already tolerate this — the blackhole rebuild
                // (`state.rs` ref-bank loop) and the bridge decoder drop a
                // Ref color whose semantic slot is `None`.  Keep encode
                // symmetric: substitute `CONST_NULL` (history.py) so the
                // positional snapshot/liveness count stays aligned rather
                // than failing loud on the unsourceable dead slot.  A color
                // the trace DID produce here (e.g. an op hoisted ahead of its
                // consumer block's marker so it is live-in at the resume
                // marker) carries its real `registers_r` box — `regs_r[color]`
                // is non-NONE precisely when the value is genuinely live, so
                // this read is the same `get_list_of_active_boxes` parity the
                // mapped arm above uses.  Under the walker (gate-off) each PC
                // owns a depth-narrowed marker, so this arm never fires.
                None => match regs_r.get(color).copied() {
                    Some(v) if v != OpRef::NONE => v,
                    _ => OpRef::const_ptr(majit_ir::GcRef(0)),
                },
                _ => fallback(),
            }
        } else {
            fallback()
        };
        if value == OpRef::NONE {
            panic!("{}", dump_ctx("ref", idx));
        }
        active.push(value);
    }
    for &idx in &banks.float {
        let v = regs_f
            .get(idx as usize)
            .copied()
            .unwrap_or_else(|| panic!("{}", dump_ctx("float", idx)));
        if v == OpRef::NONE {
            panic!("{}", dump_ctx("float", idx));
        }
        active.push(v);
    }
    active
}

/// Sync the symbolic vable `last_instr` at an intermediate `jit_merge_point`
/// reached mid-trace (the `jit_merge_point` REGISTER branch — a bridge
/// re-entering the inner loop, returning `Continue`): override `last_instr`
/// to `merge_pc - 1` (a resume into the target loop must re-enter at the
/// header opcode) and mirror it into the `virtualizable_boxes` shadow that
/// `close_loop_args_at`'s JUMP-arg derivation reads.
///
/// Without the override the JUMP into the existing loop carries the LAST
/// GUARD's published `last_instr` (e.g. 104 instead of header-1=86 on
/// fannkuch), so a vable sync inside the target loop would resume the
/// interpreter at the wrong bytecode (permutation state never reaches its
/// exit condition → non-crashing infinite loop).
///
/// No heap writeback: the vable stays virtual across the merge edge and is
/// rebuilt from guard resume-data on failure, matching the loop-close path.
fn sync_intermediate_merge_point_last_instr(ctx: &mut TraceCtx, merge_pc: usize) {
    if ctx.standard_virtualizable_box().is_none() {
        return;
    }
    let last_instr_value = merge_pc as i64 - 1;
    let opref = ctx.const_int(last_instr_value);
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx,
        "last_instr",
        opref,
        Value::Int(last_instr_value),
    );
}

/// Call-scoped inputs for one guard-snapshot capture — the explicit-parameter
/// port of `capture_resumedata`'s keyword arguments (pyjitpl.py),
/// alongside the already-explicit `after_residual_call`.
#[derive(Clone, Copy, Default)]
pub(crate) struct GuardCaptureScope<'a> {
    /// Request that a residual call's `GUARD_NO_EXCEPTION` route its resume
    /// through the call's OWN post-call `catch_exception` instead of the
    /// generic post-call fallthrough. The snapshot helper only acts on this
    /// when the call's CALL pc is directly covered by an enclosing
    /// exception-table handler: it then carries the CALL jitcode offset so a
    /// deopt resumes at the call's own catch. The blackhole's
    /// `handle_exception_in_frame` routes the raise to the enclosing handler
    /// instead of escaping the frame. Without this, a
    /// residual resumes at the NEXT opcode, whose own catch receives only a
    /// raise from that opcode, not from the call itself; a residual whose CALL
    /// pc sits directly under a try (its fallthrough may leave the covered
    /// region — e.g. a FOR_ITER-next fallthrough is the continue-arm body,
    /// reached only on a non-null item, which carries no catch for the call's
    /// own raise) needs its own catch to receive the raise. Uncovered
    /// residuals fall back to the fallthrough resume even when this is set.
    pub residual_call_catch_resume: bool,

    /// Bridge-entry flavor guard (`_prepare_exception_resumption`,
    /// pyjitpl.py:3125-3173): the walk-entry `position` IS the resume
    /// coordinate — the failing source guard's own carried word, already a
    /// decodable `-live-` startpoint the runtime just resolved.  Carry it
    /// verbatim and key the resume py off its forward twin.  The default
    /// twin lookups compensate op-START keys (the after-residual advance,
    /// the block-head re-key) and would move an already-advanced coordinate
    /// a second time — on a call inside a try-block that lands on the
    /// physically-following `except` handler block, so the entry guard's
    /// own bridge resumes INSIDE the handler.  None outside the bridge-entry
    /// captures.
    pub carried_resume_jit_pc: Option<usize>,

    /// The branch guard's own jitcode `op.pc` for a kept-stack branch guard
    /// (#124). The snapshot helper is invoked with the *resume* coordinate
    /// (`other_target`, the not-taken arm), so the guard's own coordinate is
    /// otherwise unavailable to the encoder. `collect_outer_active_boxes`
    /// reads this to recover the kept operand-stack value(s): a branch guard's
    /// not-taken arm resumes at a merge point whose live Ref color was minted
    /// by the not-taken edge's register move and is therefore unwritten in the
    /// walk's register file at the guard point; the value lives instead in the
    /// guard-pc-only color the edge move reads from. None outside the gated
    /// kept-stack path.
    pub branch_guard_jitcode_pc: Option<usize>,

    /// `#420` not-taken edge-move recovery: the decoded `ref_copy`
    /// parallel-move list of a kept-stack branch guard's trampoline, as
    /// `(resume-merge color, guard-state kept value)` pairs. The branch handler
    /// reads each `ref_copy(dst <- src)` of the not-taken edge and supplies
    /// `(dst, registers_r[src])`; `collect_outer_active_boxes` and the
    /// `stack_sync` vable overlay read it so every kept operand-stack slot
    /// resumes with the exact value the edge move would produce, instead of the
    /// stale merge-color read. This replaces the positional depth-1 heuristic
    /// with the exact, depth-independent edge resolution. Empty outside the
    /// gated kept-stack path.
    pub branch_guard_kept_recovered: &'a [(u16, OpRef)],
}

/// `rlib/jit.py` `max_unroll_recursion` default (= warmstate
/// `DEFAULT_MAX_UNROLL_RECURSION`).
const FBW_MAX_INLINE_RECURSION: usize = 7;

/// Upper bound on the parameter count the self-recursive `CALL_ASSEMBLER` fold
/// accepts. Accumulator/linear recursion is low-arity; the cap keeps a
/// pathological signature off the fold path.
const FBW_REC_CA_MAX_PARAMS: usize = 8;

/// One paused caller frame the multi-frame inline snapshot must carry
/// (#68/#124).  Computed at the inline CALL site — where the caller's live
/// register banks are still in scope — and read back at guard-capture time
/// (where the walk context is the callee's, so the caller banks are gone).
#[derive(Clone)]
struct InlineParentFrame {
    /// The caller's jitcode index (`(*JitCode).index`).
    jitcode_index: u32,
    /// The caller CALL opcode's JitCode pc. A nested inline-capture abort
    /// carries this native coordinate to the interpreter-flush boundary.
    call_jitcode_pc: Option<usize>,
    /// Concrete operand-stack slots at the caller CALL, keyed by absolute
    /// `locals_cells_stack_w` index. Used only by the abort-point flush.
    call_stack_overrides: Vec<(usize, pyre_object::PyObjectRef)>,
    /// Complete concrete image needed to turn this paused caller into an
    /// `MIFrame` if a force aborts in a descendant residual call.  Captured
    /// while the caller's register banks are still live; `None` is a
    /// best-effort decline that leaves the existing guard snapshot usable.
    blackhole: Option<InlineParentBlackhole>,
    /// Authoritative derivation of the caller's Python resume pc.  The
    /// Python-native consumers resolve this at their boundary; a CALL inside
    /// a try-block (catch marker) still declines multi-frame.
    resume_coord: ParentResumeCoord,
    /// Codewrite-time jitcode-keyed marker for the CALL fallthrough resume
    /// point, queried at the CALL site.  Bridge-root frames have no CALL
    /// offset in hand and carry `None`.
    resume_marker_jit_pc: Option<usize>,
    /// The caller's `in_a_call` active boxes at its resolved resume pc — the liveness
    /// at the return point with the not-yet-produced call-result slot nulled
    /// (`get_list_of_active_boxes(in_a_call=true)` parity, trace_opcode.rs).
    boxes: Vec<OpRef>,
}

#[derive(Clone)]
struct InlineParentBlackhole {
    /// Position immediately after the CALL.  The callee blackhole writes its
    /// return value through `call_result_reg()` before this caller runs.
    resume_pc: usize,
    int_values: Vec<(usize, i64)>,
    /// `(register color, value)`, read out of the color-indexed
    /// `concrete_registers_r` shadow and restored into the MIFrame's
    /// color-indexed Ref bank.
    ref_values: Vec<(usize, pyre_object::PyObjectRef)>,
    /// Float values have no concrete shadow bank; retain their OpRefs and
    /// resolve them at force time while the trace context is still live.
    float_values: Vec<(usize, OpRef)>,
}

/// Whether any frame the trace already models would catch an exception raised
/// below it — i.e. whether a raise escaping the callee about to be inlined has
/// to be routed back INTO the traced region.
///
/// Each paused caller on the framestack is asked whether its own pending CALL
/// sits inside a try-block, the same `after_residual_call_resume` →
/// `catch_exception/L` lookup [`decline_inline_caller_frame_for_catch_marker`]
/// uses. Level 0's parent is the snapshot root, so the scan covers the root
/// frame and every intermediate one; the innermost CALL needs no test here
/// because a nested try-block CALL already declines multiframe on its own.
///
/// Unknown answers count as catching: a parent with no snapshot, an
/// unresolvable jitcode, or a CALL with no recorded pc could all be hiding a
/// handler, and admitting one wrongly costs a guard storm.
pub(crate) fn inline_chain_catches_a_raise(session: &std::cell::RefCell<WalkSession>) -> bool {
    session.borrow().framestack.iter().any(|frame| {
        let Some(parent) = frame.parent.as_ref() else {
            return true;
        };
        let Some(call_pc) = parent.call_jitcode_pc else {
            return true;
        };
        let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(parent.jitcode_index as i32)
        else {
            return true;
        };
        match pjc.after_residual_call_resume_for_jitcode_pc(call_pc) {
            // No after-call resume marker: the CALL is not in a try-block.
            None => false,
            Some(resume) => try_catch_exception_at(pjc.jitcode.code.as_slice(), resume).is_some(),
        }
    })
}

/// The derivation flavor of a paused caller frame's Python resume pc.
#[derive(Clone, Copy)]
enum ParentResumeCoord {
    /// `resume_py_pc = backxlat_py_pc(jitcode_index, jitcode_pc)`. Used by
    /// both bridge-root and reconstructed-recipe parent frames.
    Backxlat(usize),
    /// `resume_py_pc = semantic_fallthrough_pc(code,
    /// python_pc_for_jitcode_pc(metadata, call_jitcode_pc))`.
    CallFallthrough(usize),
}

/// RAII guard for one framestack level. Pop on drop so `?` and nested
/// sub-walks unwind to the caller's level.
struct InlineFrameGuard<'a>(&'a std::cell::RefCell<WalkSession>);

thread_local! {
    /// Top-level caller CALL native JitCode coordinate and concrete
    /// operand-stack slots stashed by
    /// [`fbw_abort_nested_unjournaled_residual`] at the nested-inline decline,
    /// read back by the trace loop after the walk unwinds
    /// ([`fbw_abort_outer_resume_take`]) to drive the abort-point flush.
    /// The trailing `usize` is the executed-effect odometer at that CALL, the
    /// `WalkEndResume::Rewind` snapshot the flush re-checks before committing.
    static FBW_ABORT_OUTER_RESUME: std::cell::Cell<Option<(u32, usize, usize)>> =
        const { std::cell::Cell::new(None) };
    static FBW_ABORT_OUTER_STACK_OVERRIDES: std::cell::RefCell<Vec<(usize, pyre_object::PyObjectRef)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Raw pointer to the walk session whose framestack currently holds
    /// inline frames, for [`fbw_store_journal_root_walker`]: the session
    /// lives on the walk driver's Rust stack, which the GC cannot scan, and
    /// each `InlineParentFrame::call_stack_overrides` slot holds a
    /// nursery-resident ref across residual-call allocations.  Set by
    /// [`InlineFrameGuard::enter`] and restored on drop; null outside any
    /// inline sub-walk.
    static ACTIVE_WALK_SESSION: std::cell::Cell<*const std::cell::RefCell<WalkSession>> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

impl<'a> InlineFrameGuard<'a> {
    fn enter(
        session: &'a std::cell::RefCell<WalkSession>,
        w_code: usize,
        parent: Option<InlineParentFrame>,
    ) -> Self {
        session.borrow_mut().framestack.push(InlineFrame {
            w_code,
            parent,
            entry_executed_effects: fbw_executed_effect_count(),
        });
        ACTIVE_WALK_SESSION.with(|c| c.set(session as *const _));
        InlineFrameGuard(session)
    }
}

impl Drop for InlineFrameGuard<'_> {
    fn drop(&mut self) {
        let mut session = self.0.borrow_mut();
        session.framestack.pop();
        if session.framestack.is_empty() {
            ACTIVE_WALK_SESSION.with(|c| c.set(std::ptr::null()));
        }
    }
}

/// Terminal disposition of a walk kept for the no-replay exit:
/// the top-level frame's concrete return value, or the concrete
/// exception object of the uncaught raise that ended the walk
/// (`opimpl_raise` → `finishframe_exception` →
/// `compile_exit_frame_with_exception`, pyjitpl.py +
/// 3238-3242; the caller consumes it as `ExitFrameWithExceptionRef`,
/// jitexc.py).
#[derive(Clone, Copy)]
pub enum FinishConcrete {
    Return(ConcreteValue),
    Raise(ConcreteValue),
}

/// # Safety
/// `data` must come from [`capture_fbw_finish_concrete_root_area`], and the
/// owning thread must be quiesced.
pub unsafe fn fbw_finish_concrete_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let cell = unsafe { &*(data as *const std::cell::Cell<Option<FinishConcrete>>) };
    let Some(finish) = cell.get() else {
        return;
    };
    let (value, is_raise) = match finish {
        FinishConcrete::Return(value) => (value, false),
        FinishConcrete::Raise(value) => (value, true),
    };
    if let ConcreteValue::Ref(ptr) = value {
        let mut gcref = majit_ir::GcRef(ptr as usize);
        visitor(&mut gcref);
        // The visitor may forward (relocate) the ref; write it back so
        // the stashed value points at the moved object.
        let value = ConcreteValue::Ref(gcref.0 as pyre_object::PyObjectRef);
        cell.set(Some(if is_raise {
            FinishConcrete::Raise(value)
        } else {
            FinishConcrete::Return(value)
        }));
    }
}

/// One in-flight FOR_ITER continuation entry (#57 Option C): the item the
/// FOR_ITER `for_iter_next` residual consumed on the authoritative walk, the
/// body coordinate to resume at (the FOR_ITER `py_pc + 1` continue-arm
/// fallthrough, resolved only at a match point),
/// and whether a body effect committed since THIS consume.
///
/// `body_effect_since_consume` is the R1 double-apply guard, per entry: an
/// irreversible heap mutation that succeeded after this consume (a body the
/// store/append journals do NOT cover).  Re-running this iteration's body on
/// delivery would re-apply it, so [`fbw_foriter_inflight_take`] refuses
/// delivery when set.  A mutation committed while several FOR_ITER items are
/// in flight is "after" every one of them (re-running ANY of their bodies
/// re-applies it), so the executor marks the flag on EVERY active entry; a
/// fresh consume's own entry starts clear.
#[derive(Clone, Copy, Debug)]
enum InflightForiterBody {
    /// Legacy entry-PC fallback for a per-opcode walk or fixture with no
    /// full-body JitCode coordinate.
    Py(usize),
    /// The outer Python JitCode identity and the `for_iter_next` residual's
    /// own JitCode pc. The body fallthrough is derived only at a match point.
    Jit {
        outer_jitcode_index: u32,
        op_pc: usize,
    },
}

#[derive(Clone, Copy)]
struct InflightForiter {
    item: pyre_object::PyObjectRef,
    body: InflightForiterBody,
    body_effect_since_consume: bool,
    /// The walk re-reached this FOR_ITER's consume after the item's body ran
    /// (a NEW `for_iter_next` attempt was dispatched for the same body).
    /// A completed entry must never be re-delivered — its body already ran
    /// during the walk — but a flush may still adopt the walk end state at
    /// the header WITHOUT delivery (the interpreter re-attempts the consume).
    body_completed: bool,
}

thread_local! {
    /// Undo log for the walked region's eagerly executed list stores:
    /// `(list, key, displaced_value)` triples pushed by the `STORE_SUBSCR`
    /// specializations before they mutate the list.  Upstream executes
    /// every traced operation concretely (pyjitpl.py
    /// execute_and_record) and never re-runs the traced region, so the
    /// walker applies the store at trace time too.  Pyre's transitional
    /// non-commit paths instead RE-RUN the region (the legacy
    /// replay-from-snapshot), which would re-apply the store against the
    /// already-mutated heap (a swap re-reads its own output and swaps
    /// back) — so a walk that does not commit its end state
    /// (`flush_walk_end_state_to_frame`) rolls these entries back in
    /// reverse order, restoring the pre-walk heap the replay expects.  A
    /// committing walk drops the log (the mutation is already applied,
    /// exactly once).  Dies with the replay paths.  Entries are GC roots
    /// via [`fbw_store_journal_root_walker`].
    static FBW_STORE_JOURNAL: std::cell::RefCell<Vec<[pyre_object::PyObjectRef; 3]>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo log for the walked region's eagerly executed list APPENDS:
    /// `(list, length_before_append)` pairs pushed by the `list.append`
    /// specialization before it grows the list.  Same rationale as
    /// [`FBW_STORE_JOURNAL`] — the append is admitted only when
    /// `w_list_can_append_without_realloc` holds, so the undo is a pure
    /// length rewind (`w_list_int_set_len`, no reallocation, no boxing) and
    /// the backing array still has the slot.  A committing walk drops the
    /// log; a non-commit walk rewinds each list's length in reverse push
    /// order so the legacy replay re-appends against the pre-walk heap.
    /// Entries' list refs are GC roots via [`fbw_store_journal_root_walker`].
    static FBW_APPEND_JOURNAL: std::cell::RefCell<Vec<(pyre_object::PyObjectRef, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo log for Empty-list first-append promotion during pyre's
    /// speculative full-body walk.  Entries ride the same commit/rollback
    /// lifecycle as [`FBW_APPEND_JOURNAL`]: a committed walk keeps the typed
    /// strategy, while a replayed walk restores the list to Empty so replay
    /// can execute the append from the original shape.  This exists only for
    /// pyre's speculative-replay walk and can be removed when
    /// single-executor tracing lands (gh#73/#34).  Entries are GC roots via
    /// [`fbw_store_journal_root_walker`].
    static FBW_APPEND_PROMOTE_JOURNAL: std::cell::RefCell<Vec<pyre_object::PyObjectRef>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo log for the walked region's eagerly executed module-global
    /// `IntMutableCell` stores: `(cell, intvalue_before)` pairs pushed by the
    /// StoreName/StoreGlobal cell fold before it writes `cell.intvalue` in
    /// place.  Same rationale as [`FBW_STORE_JOURNAL`] — the walker is the
    /// authoritative executor, so a folded store must apply its concrete
    /// effect at walk time (the residual it replaces would have run
    /// `write_cell` via `try_execute_residual_call_via_executor`); a
    /// non-commit walk restores each cell's prior `intvalue` in reverse push
    /// order so the legacy replay re-applies the store against the pre-walk
    /// heap.  Cells are immovable (`malloc_typed`; the fold's `can_move`
    /// gate) and stay reachable from their module dict slot, so entries need
    /// no GC-root forwarding.
    static FBW_CELL_STORE_JOURNAL: std::cell::RefCell<Vec<(pyre_object::PyObjectRef, i64)>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo log for the walked region's eagerly executed
    /// `set_current_exception` stores against the LIVE per-thread
    /// `ExecutionContext.sys_exc_value`.  The authoritative walk lowers
    /// PUSH_EXC_INFO / POP_EXCEPT to `set_current_exception` residuals and
    /// applies their concrete effect at walk time
    /// ([`try_walker_lower_exc_info_residual`]) so a following
    /// `get_current_exception` reads the right value.  Same rationale as
    /// [`FBW_STORE_JOURNAL`]: on a non-commit exit the legacy replay re-runs
    /// the walked region and must find the pre-walk `sys_exc_value`.  Without
    /// this journal an exception that propagates OUT of an except-handler
    /// (the handler body itself raises) aborts the walk BEFORE its paired
    /// POP_EXCEPT restore runs, leaving the live EC holding the caught
    /// exception; the next frame reads it as the active exception and chains
    /// it as a spurious `__context__`.  Each entry is the displaced prior
    /// value (`get_current_exception()` read just before the eager store);
    /// the rollback re-applies them in reverse push order so the final write
    /// restores the walk-entry value.  Entries are GC roots via
    /// [`fbw_store_journal_root_walker`].
    static FBW_SYS_EXC_JOURNAL: std::cell::RefCell<Vec<pyre_object::PyObjectRef>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo entry for the concrete traceback-head store performed while a
    /// bridge-entry exception arm is recorded: `(exception, attached_node)`.
    /// The two arms are mutually exclusive and each is entered at most once
    /// per walk session, so one slot covers the concrete attach.  It shares
    /// the store journal's lifecycle: a committing walk keeps the node and
    /// clears the entry, while a discarded walk removes the node only when it
    /// is still the exception's current traceback head.  Both refs are GC
    /// roots via [`fbw_store_journal_root_walker`].
    static FBW_TRACEBACK_STORE_JOURNAL:
        std::cell::RefCell<Option<(pyre_object::PyObjectRef, pyre_object::PyObjectRef)>> =
        const { std::cell::RefCell::new(None) };

    /// In-flight FOR_ITER continuation (#57 Option C): `(consumed_item,
    /// body coordinate)` stashed when the FOR_ITER `for_iter_next` residual
    /// ([`PyreHelperKind::ForIterNext`]) runs concretely on the
    /// authoritative walk and advances the real shared heap iterator.  The
    /// advance is an irreversible side effect with no journal undo
    /// (`functional.rs` `current += step`); on a successful CloseLoop commit
    /// the walk-end flush adopts the advanced state, so the stash is dropped
    /// ([`fbw_store_journal_commit`]).  On a trace ABORT the walk discards
    /// its recording but the iterator stays advanced — so the stash is
    /// DELIVERED to the live frame (the consumed item pushed, the frame
    /// repositioned at its resolved body pc) instead of dropping the iteration, the
    /// `_copy_data_from_miframe` continue-forward analog (blackhole.py).
    /// The body coordinate represents the FOR_ITER continue-arm fallthrough
    /// (`py_pc + 1`, codewriter.rs continue arm). The item ref is a GC root via
    /// [`fbw_store_journal_root_walker`].  Cleared at walk start
    /// ([`fbw_store_journal_reset`]).
    ///
    /// A LIFO stack, not a single slot: a walk that descends into a NESTED
    /// FOR_ITER has BOTH the outer loop's consumed item and the inner loop's
    /// consumed item in flight at once.  Each [`InflightForiter`] is keyed by
    /// its resolved body pc (the FOR_ITER's own pc + 1, derived from the
    /// consuming op's pc), so a re-consume of the SAME FOR_ITER (the prior iteration's
    /// body completed) replaces that loop's entry while a consume of a
    /// DIFFERENT (nested) FOR_ITER pushes a new entry — the outer entry is no
    /// longer destroyed by the inner consume.  The abort delivery
    /// ([`fbw_foriter_inflight_take`]) still consumes only the most-recent
    /// (top) entry, matching the single-slot behaviour; preserving the
    /// remaining entries is the representation S2 needs to deliver each at its
    /// true frame slot.
    static FBW_FORITER_INFLIGHT: std::cell::RefCell<Vec<InflightForiter>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo log for a bridge/retrace recording walk's eager range-iterator
    /// cursor advance.  The main walk leaves the advance unjournaled and relies
    /// on in-flight FOR_ITER forward-delivery to recover the consumed item on
    /// abort; the bridge/retrace abort path has no such delivery, so a bridge
    /// walk records `(iter, pre_current, pre_remaining)` here and restores the
    /// cursor when it does NOT commit — leaving the recording side-effect
    /// neutral so the interpreter resume re-consumes the item exactly once.
    /// Only populated while `is_bridge_trace`; empty (no-op) on the main walk.
    static FBW_BRIDGE_ITER_JOURNAL: std::cell::RefCell<Vec<(pyre_object::PyObjectRef, i64, i64)>> =
        const { std::cell::RefCell::new(Vec::new()) };

    static FBW_UNJOURNALED_VALUE_UNAVAILABLE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    static FBW_UNJOURNALED_SYMBOLIC: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };

    static FBW_EXECUTED_RESIDUAL_VOID: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static FBW_EXECUTED_RESIDUAL_MAYFORCE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static FBW_EXECUTED_RESIDUAL_PLAIN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };

    /// gh#467 executed-effect odometer: bumped only when the walk concretely
    /// applies an effect — a store/append/cell journal push, a Void / mutator-
    /// tagged residual executed by [`try_execute_residual_call_via_executor`],
    /// or a residual whose concrete run entered a user Python frame (a value-
    /// returning dunder that may mutate).  A declined residual recorded only
    /// symbolically sets an unjournaled flag ([`FBW_UNJOURNALED_VALUE_UNAVAILABLE`]
    /// / [`FBW_UNJOURNALED_SYMBOLIC`]) but does NOT move this
    /// counter: no effect executed.  The inline abort-forward-flush gate
    /// (`try_walker_inline_user_call` / gh#467) snapshots both signals at the
    /// CALL.  A nonzero count delta means the callee attempt cannot be
    /// discarded and re-executed without risking a double.
    static FBW_EXECUTED_EFFECT_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };

    /// Effect-count delta of the exact JitCode opcode that most recently
    /// returned `LoopBearingCalleeInlineUnsupported`.  The structural
    /// mid-body carrier accepts only a zero delta: pyre re-runs the enclosing
    /// Python opcode boundary, unlike byte-exact `blackhole_from_resumedata`
    /// (`pyjitpl.py`), so any already-applied effect in that opcode must
    /// keep the legacy path.
    static FBW_STRUCTURAL_ABORT_OPCODE_EFFECTS: std::cell::Cell<Option<(usize, usize)>> =
        const { std::cell::Cell::new(None) };

    /// gh#467 inline-abort forward-flush carrier: latched by
    /// [`try_walker_inline_user_call`] when a supported abort fires inside a
    /// TOP-level inline sub-walk whose callee executed no concrete effect.
    /// Holds `(outer CALL python pc, [callable, null_or_self, args...])` — the
    /// exact operand stack the interpreter's CALL opcode expects — so the walk
    /// driver can flush the outer frame AT that CALL and resume the interpreter
    /// forward (re-executing the callee from scratch) instead of rolling back
    /// and replaying the loop body from entry.  The `PyObjectRef`s are rooted by
    /// [`fbw_store_journal_root_walker`] across the abort unwind's allocations.
    static FBW_ABORT_CALL_RESUME: std::cell::RefCell<Option<InlineAbortCarrier>> =
        const { std::cell::RefCell::new(None) };

    /// B3: the set of OpRefs the walker built inline via
    /// [`try_walker_trace_exception_new`] (the virtualizable `NewWithVtable`
    /// exception). The immediately-following `RaiseVarargs` residual consults
    /// it to take the instance fast path — skipping the
    /// residual `normalize_raise_varargs_jit` publish + `GUARD_EXCEPTION`
    /// and emitting `__context__` as a `SetfieldGc` on the (still virtual)
    /// exception.  Reset at each FBW walk entry via `fbw_store_journal_reset`
    /// so a stale OpRef key from a prior recorder cannot leak across walks.
    static FBW_BUILT_EXC: std::cell::RefCell<std::collections::HashSet<OpRef>> =
        std::cell::RefCell::new(std::collections::HashSet::new());

    /// B3: LIFO stack of the previous-exception slot value
    /// saved by each lowered `PUSH_EXC_INFO` (`get_current_exception` arm),
    /// paired with its live concrete.  `POP_EXCEPT`'s restore consumes the top
    /// entry, so the slot is set back to the TRUE saved prev (None for an outer
    /// handler, the outer exception for a nested one) instead of the operand-
    /// stack value the codewriter threads into the `set_current_exception`
    /// residual — which the walker resolves to the just-caught exception, not
    /// the saved prev.  Restoring the saved prev is what lets the balanced
    /// save/restore cancel (the locally-caught exception de-escapes and DCEs)
    /// and keeps `sys.exc_info()` correct after the handler unwinds.  Reset per
    /// walk via `fbw_store_journal_reset`.
    static FBW_EXC_PREV: std::cell::RefCell<Vec<(OpRef, pyre_object::PyObjectRef)>> =
        std::cell::RefCell::new(Vec::new());

    /// Set when a `PUSH_EXC_INFO` prev save (`get_current_exception`) is
    /// lowered; the immediately-following `set_current_exception` is then that
    /// PUSH's slot store (it stores the EXC and leaves the saved prev on
    /// [`FBW_EXC_PREV`]) rather than a `POP_EXCEPT` restore (which pops it).
    /// The two `SetCurrentException` shapes are otherwise identical.
    static FBW_EXC_PENDING_PUSH_SET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[derive(Clone, Debug)]
pub(crate) enum InlineAbortCarrier {
    Entry {
        /// The outer CALL's native coordinate, resolved at the interpreter
        /// flush boundary.
        outer_jitcode_index: u32,
        call_jitcode_pc: usize,
        call_stack: Vec<pyre_object::PyObjectRef>,
        /// [`FBW_EXECUTED_EFFECT_COUNT`] at the CALL this carrier resumes at,
        /// sampled when the latch was set.  The flush that consumes this
        /// carrier re-executes that CALL, so it is
        /// [`crate::trace::WalkEndResume::Rewind`] and must prove the odometer
        /// has not moved since.
        entry_executed_effects: usize,
    },
    MidBody(MidBodyPayload),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MidBodyAbortKind {
    Marker,
    Structural,
}

#[derive(Clone, Debug)]
pub(crate) struct MidBodyPayload {
    pub abort_kind: MidBodyAbortKind,
    /// The outer CALL's native coordinate, resolved at the interpreter flush
    /// boundary.
    pub outer_jitcode_index: u32,
    pub call_jitcode_pc: usize,
    pub call_stack_len: usize,
    /// The rebuilt callee's native abort coordinate, resolved at the
    /// interpreter flush boundary.
    pub callee_jitcode_index: u32,
    pub abort_jitcode_pc: usize,
    /// Forward-carried Python instruction PC for `abort_jitcode_pc`, stamped
    /// once at capture so the flush reads a scalar instead of re-inverting the
    /// coordinate through the jitcode->py tables.
    pub callee_py_pc: usize,
    pub w_code: pyre_object::PyObjectRef,
    pub w_globals: pyre_object::PyObjectRef,
    pub live_locals: Vec<Option<ConcreteValue>>,
    pub live_stack: Vec<ConcreteValue>,
    pub return_value: pyre_object::PyObjectRef,
    /// What [`InlineAbortCarrier::Entry`] would have carried for the same
    /// abort.  The rebuild is preferred, but it can still decline at the flush
    /// (an unsourceable outer local, a handler-bearing body); dropping to the
    /// legacy replay there would re-open the double-apply this carrier family
    /// exists to close, so the entry rewind stands behind it.  `None` when the
    /// entry latch's own gate refused.
    pub entry_fallback: Option<EntryFallback>,
}

#[derive(Clone, Debug)]
pub(crate) struct EntryFallback {
    pub call_stack: Vec<pyre_object::PyObjectRef>,
    /// See [`InlineAbortCarrier::Entry::entry_executed_effects`].
    pub entry_executed_effects: usize,
}

struct FbwStoreJournalRootArea {
    stores: *const std::cell::RefCell<Vec<[pyre_object::PyObjectRef; 3]>>,
    appends: *const std::cell::RefCell<Vec<(pyre_object::PyObjectRef, usize)>>,
    append_promote: *const std::cell::RefCell<Vec<pyre_object::PyObjectRef>>,
    abort_overrides: *const std::cell::RefCell<Vec<(usize, pyre_object::PyObjectRef)>>,
    cell_stores: *const std::cell::RefCell<Vec<(pyre_object::PyObjectRef, i64)>>,
    sys_exc: *const std::cell::RefCell<Vec<pyre_object::PyObjectRef>>,
    traceback_store:
        *const std::cell::RefCell<Option<(pyre_object::PyObjectRef, pyre_object::PyObjectRef)>>,
    foriter: *const std::cell::RefCell<Vec<InflightForiter>>,
    bridge_iter: *const std::cell::RefCell<Vec<(pyre_object::PyObjectRef, i64, i64)>>,
    abort_resume: *const std::cell::RefCell<Option<InlineAbortCarrier>>,
    active_session: *const std::cell::Cell<*const std::cell::RefCell<WalkSession>>,
    escape_flush_undo: *const std::cell::RefCell<Option<EscapeFlushUndo>>,
    single_frame_blackhole: *const std::cell::RefCell<Option<LatchedSingleFrameBlackhole>>,
    multi_frame_blackhole: *const std::cell::RefCell<Option<LatchedMultiFrameBlackhole>>,
}

thread_local! {
    static FBW_STORE_JOURNAL_ROOT_AREA: FbwStoreJournalRootArea = FbwStoreJournalRootArea {
        stores: FBW_STORE_JOURNAL.with(|value| value as *const _),
        appends: FBW_APPEND_JOURNAL.with(|value| value as *const _),
        append_promote: FBW_APPEND_PROMOTE_JOURNAL.with(|value| value as *const _),
        abort_overrides: FBW_ABORT_OUTER_STACK_OVERRIDES.with(|value| value as *const _),
        cell_stores: FBW_CELL_STORE_JOURNAL.with(|value| value as *const _),
        sys_exc: FBW_SYS_EXC_JOURNAL.with(|value| value as *const _),
        traceback_store: FBW_TRACEBACK_STORE_JOURNAL.with(|value| value as *const _),
        foriter: FBW_FORITER_INFLIGHT.with(|value| value as *const _),
        bridge_iter: FBW_BRIDGE_ITER_JOURNAL.with(|value| value as *const _),
        abort_resume: FBW_ABORT_CALL_RESUME.with(|value| value as *const _),
        active_session: ACTIVE_WALK_SESSION.with(|value| value as *const _),
        escape_flush_undo: escape_flush_undo_cell_ptr(),
        single_frame_blackhole: single_frame_blackhole_cell_ptr(),
        multi_frame_blackhole: multi_frame_blackhole_cell_ptr(),
    };
}

/// Whether the opcode at `resume_py_pc` in the live frame's code is a
/// FOR_ITER — a non-FOR_ITER resume pc that merely happens to satisfy
/// `body_pc == some entry's resolved body pc` is Shape B.
fn foriter_header_at(frame: usize, resume_py_pc: usize) -> bool {
    let frame_ptr = frame as *const u8;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return false;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    matches!(
        pyre_interpreter::decode_instruction_at(unsafe { &*raw_code }, resume_py_pc),
        Some((pyre_interpreter::Instruction::ForIter { .. }, _))
    )
}

fn exact_floor_segment_anchor(
    metadata: &crate::PyJitCodeMetadata,
    py_pc: usize,
    jit_pc: usize,
) -> bool {
    crate::pyjitcode::floor_segment_for_jitcode_pc(&metadata.py_floor_by_jit_pc, jit_pc)
        .is_some_and(|(start, py)| start == jit_pc && py as usize == py_pc)
}

/// The jitcode ops the callee-rebuild leg would re-run if it resumed at the
/// Python pc owning `jit_pc` — everything from that pc's floor-segment start up
/// to `jit_pc`.  Empty exactly when [`exact_floor_segment_anchor`] holds, so a
/// non-empty list names what makes the anchor inexact.
pub(crate) fn floor_segment_ops_before(
    metadata: &crate::PyJitCodeMetadata,
    code: &[u8],
    jit_pc: usize,
) -> Vec<String> {
    let Some((mut pc, _py)) =
        crate::pyjitcode::floor_segment_for_jitcode_pc(&metadata.py_floor_by_jit_pc, jit_pc)
    else {
        return vec!["<no floor segment>".to_string()];
    };
    let mut keys = Vec::new();
    while pc < jit_pc {
        let Some(op) = crate::jitcode_runtime::decode_op_at(code, pc) else {
            keys.push("<undecodable>".to_string());
            break;
        };
        keys.push(format!(
            "{}@f{}",
            op.key,
            code.get(op.pc + 1).copied().unwrap_or(u8::MAX)
        ));
        pc = op.next_pc;
    }
    keys
}

/// Admit an abort pc at the semantic head of a portal-shaped Python opcode —
/// that is, one whose whole jitcode prefix inside its own floor segment writes
/// nothing but THIS frame's own virtualizable: operand-stack slots
/// (`setarrayitem_vable_r`) and the two int bookkeeping fields,
/// `last_instr`/`valuestackdepth` (`setfield_vable_i`).
///
/// The callee-rebuild leg reconstructs the frame wholesale — a fresh `PyFrame`
/// whose locals, stack area, `valuestackdepth` and `last_instr` all come from
/// the carried payload — so every one of those writes is erased before the
/// rebuilt frame runs, and the resume state at `py_pc` is unaffected by how far
/// into the opcode's spill the walk got.  `live_stack`'s depth is
/// `depth_at_py_pc[py_pc]` (`depth_for_jitcode_pc_pred` is keyed per Python pc),
/// i.e. the depth BEFORE the opcode, which is what resuming at `py_pc` needs.
///
/// `getarrayitem_vable_r` is NOT admitted even though it too only reads the
/// vable: it writes a jitcode REGISTER, and the payload sources `live_stack`
/// and `live_locals` out of `concrete_registers_r` by color.  A clobbered color
/// would be read back as a live value.
///
/// Anything else, or a write aimed at another frame, declines.
fn portal_vable_bookkeeping_anchor(
    metadata: &crate::PyJitCodeMetadata,
    built_as_portal: bool,
    portal_frame_reg: u16,
    perfn_descrs: &[majit_metainterp::jitcode::RuntimeBhDescr],
    code: &[u8],
    py_pc: usize,
    jit_pc: usize,
    mut python_pc_for_op: impl FnMut(usize) -> usize,
) -> bool {
    if exact_floor_segment_anchor(metadata, py_pc, jit_pc) {
        return true;
    }
    if !built_as_portal || portal_frame_reg > u8::MAX as u16 {
        return false;
    }
    let Some((mut pc, floor_py_pc)) =
        crate::pyjitcode::floor_segment_for_jitcode_pc(&metadata.py_floor_by_jit_pc, jit_pc)
    else {
        return false;
    };
    if floor_py_pc as usize != py_pc || pc >= jit_pc {
        return false;
    }
    while pc < jit_pc {
        let Some(op) = crate::jitcode_runtime::decode_op_at(code, pc) else {
            return false;
        };
        if op.next_pc > jit_pc
            || python_pc_for_op(op.pc) != py_pc
            || code.get(op.pc + 1).copied() != Some(portal_frame_reg as u8)
        {
            return false;
        }
        match op.key {
            // Which stack slot a spill addresses does not matter: the whole
            // stack area is rewritten from `live_stack`.
            "setarrayitem_vable_r/rirdd" => {}
            // `rid`: opcode, frame reg, value reg, little-endian descr index.
            // The int vable fields are `last_instr` (0) and `valuestackdepth`
            // (2), both reassigned by the rebuild; a non-`VableField` descr is
            // not frame bookkeeping.
            "setfield_vable_i/rid" => {
                let Some((&lo, &hi)) = code.get(op.pc + 3).zip(code.get(op.pc + 4)) else {
                    return false;
                };
                let descr_index = lo as usize | ((hi as usize) << 8);
                if !matches!(
                    perfn_descrs.get(descr_index),
                    Some(majit_metainterp::jitcode::RuntimeBhDescr::Descr(
                        majit_translate::jitcode::BhDescr::VableField { .. }
                    ))
                ) {
                    return false;
                }
            }
            // A pure vable read writes no frame state — only a jitcode
            // register, and the rebuild discards those: it reconstructs a
            // PyFrame, sets `last_instr = callee_py_pc - 1` and resumes the
            // PLAIN interpreter (`frame.execute_frame`), which never observes a
            // jitcode register bank.  So whether this op ran before the abort
            // makes no difference to the frame the rebuild hands over.  That is
            // a weaker requirement than the spill stores above, which are
            // admitted only because the rebuild rewrites the areas they touch.
            "getarrayitem_vable_r/ridd>r" => {}
            _ => return false,
        }
        pc = op.next_pc;
    }
    pc == jit_pc
}

/// Read one exact Ref slot from a callee sub-walk's fresh-frame virtualizable
/// shadow. The shadow's `fold_frame_reg` witnesses the strict-fold shape; an
/// entry whose recorded `frame_reg` matches witnesses the live multi-frame
/// inline shape. A foreign frame, missing, non-Ref, or null value is ambiguous
/// and declines.
fn callee_vable_ref_at(
    shadow: Option<&CalleeLocalsShadow>,
    frame_reg: u16,
    slot: usize,
) -> Option<ConcreteValue> {
    if frame_reg == u16::MAX {
        return None;
    }
    let shadow = shadow?;
    let entry = shadow.concrete.get(&(slot as i64)).copied()?;
    let strict_fold_witness = shadow.fold_frame_reg == frame_reg;
    let inline_scope_witness = entry.frame_reg == frame_reg;
    if !strict_fold_witness && !inline_scope_witness {
        return None;
    }
    match entry.value {
        Value::Ref(r) if r.as_usize() != 0 => {
            Some(ConcreteValue::Ref(r.as_usize() as pyre_object::PyObjectRef))
        }
        Value::Int(_) | Value::Float(_) | Value::Ref(_) | Value::Void => None,
    }
}

/// Scoped setter for [`SELFREC_CA_FOLD_ACTIVE`]. Restores the prior value so
/// nested folds and early `?` unwinds cannot strand the exemption enabled.
struct SelfRecCaFoldGuard {
    prior: bool,
}

impl SelfRecCaFoldGuard {
    fn enter() -> Self {
        let prior = SELFREC_CA_FOLD_ACTIVE.with(|c| {
            let prior = c.get();
            c.set(true);
            prior
        });
        Self { prior }
    }
}

impl Drop for SelfRecCaFoldGuard {
    fn drop(&mut self) {
        SELFREC_CA_FOLD_ACTIVE.with(|c| c.set(self.prior));
    }
}

struct ExceptionStringInlineGuard {
    prior: bool,
}

impl ExceptionStringInlineGuard {
    fn enter() -> Self {
        let prior = EXCEPTION_STRING_INLINE_ACTIVE.with(|c| {
            let prior = c.get();
            c.set(true);
            prior
        });
        Self { prior }
    }
}

impl Drop for ExceptionStringInlineGuard {
    fn drop(&mut self) {
        EXCEPTION_STRING_INLINE_ACTIVE.with(|c| c.set(self.prior));
    }
}

/// # Safety
/// `data` must come from [`capture_fbw_store_journal_root_area`], and the
/// owning thread must be quiesced.
pub unsafe fn fbw_store_journal_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let area = unsafe { &*(data as *const FbwStoreJournalRootArea) };
    let stores = unsafe { &mut *(*area.stores).as_ptr() };
    for triple in stores.iter_mut() {
        for slot in triple.iter_mut() {
            visitor(unsafe { &mut *(slot as *mut pyre_object::PyObjectRef).cast() });
        }
    }
    let appends = unsafe { &mut *(*area.appends).as_ptr() };
    for (list, _len) in appends.iter_mut() {
        visitor(unsafe { &mut *(list as *mut pyre_object::PyObjectRef).cast() });
    }
    // SAFETY: promoted-list refs can be nursery-resident across the rest of
    // the walk; a minor collection may move them before rollback restores
    // Empty, so each journal slot is a root.
    let append_promote = unsafe { &mut *(*area.append_promote).as_ptr() };
    for list in append_promote.iter_mut() {
        visitor(unsafe { &mut *(list as *mut pyre_object::PyObjectRef).cast() });
    }
    // Nested-inline abort outer-frame stash: PyObjectRef slots kept across the
    // rest of the walk; only the ref slot is a root.
    let abort_overrides = unsafe { &mut *(*area.abort_overrides).as_ptr() };
    for (_slot, value) in abort_overrides.iter_mut() {
        visitor(unsafe { &mut *(value as *mut pyre_object::PyObjectRef).cast() });
    }
    // Inline parent-frame stack overrides on the active walk session's
    // framestack. The session lives on the walk driver's Rust stack (which the
    // GC cannot scan) and each override slot holds a nursery-resident ref.
    let session_ptr = unsafe { (*area.active_session).get() };
    if !session_ptr.is_null() {
        // SAFETY: `ACTIVE_WALK_SESSION` is set by `InlineFrameGuard::enter` and
        // cleared when the last frame pops, so a non-null pointer refers to a
        // session that is live for the duration of this quiesced walk.
        let session = unsafe { &mut *(*session_ptr).as_ptr() };
        for frame in session.framestack.iter_mut() {
            if let Some(parent) = frame.parent.as_mut() {
                for (_slot, value) in parent.call_stack_overrides.iter_mut() {
                    visitor(unsafe { &mut *(value as *mut pyre_object::PyObjectRef).cast() });
                }
                if let Some(blackhole) = parent.blackhole.as_mut() {
                    for (_color, value) in blackhole.ref_values.iter_mut() {
                        visitor(unsafe { &mut *(value as *mut pyre_object::PyObjectRef).cast() });
                    }
                }
            }
        }
    }
    // Cell-store journal: the cell is immovable (`malloc_typed`) so no
    // forwarding happens, but a mid-walk rebind can drop the module dict's
    // only reference — rooting it keeps the rollback's `intvalue` restore
    // from writing into a freed block.
    let cell_stores = unsafe { &mut *(*area.cell_stores).as_ptr() };
    for (cell, _intvalue) in cell_stores.iter_mut() {
        visitor(unsafe { &mut *(cell as *mut pyre_object::PyObjectRef).cast() });
    }
    // The displaced `sys_exc_value` entries are exception objects that may be
    // nursery-resident and no longer referenced elsewhere once the eager store
    // overwrote the EC slot; forward each so a minor collection during the rest
    // of the walk cannot free/move the value the rollback restores.
    let sys_exc = unsafe { &mut *(*area.sys_exc).as_ptr() };
    for displaced in sys_exc.iter_mut() {
        // SAFETY: `PyObjectRef` and `GcRef` share the usize repr; the
        // borrowed area keeps the Vec storage alive for the visit.
        visitor(unsafe { &mut *(displaced as *mut pyre_object::PyObjectRef).cast() });
    }
    // The exception and freshly attached traceback node remain live until the
    // walk commits or rolls back.  The entry may be their only scanned owner
    // after later concrete execution changes the exception's head.
    let traceback_store = unsafe { &mut *(*area.traceback_store).as_ptr() };
    if let Some((exception, node)) = traceback_store.as_mut() {
        visitor(unsafe { &mut *(exception as *mut pyre_object::PyObjectRef).cast() });
        visitor(unsafe { &mut *(node as *mut pyre_object::PyObjectRef).cast() });
    }
    // #57 Option C: each captured in-flight FOR_ITER item is nursery-resident
    // across the rest of the walk (subsequent residual calls allocate and a
    // minor collection moves nursery objects), so forward every entry's item
    // as a root.
    let foriter = unsafe { &mut *(*area.foriter).as_ptr() };
    for entry in foriter.iter_mut() {
        // SAFETY: as above — only the `PyObjectRef` slot is a root; the
        // `usize` body pc and the bool flag are plain scalars.
        visitor(unsafe { &mut *(&mut entry.item as *mut pyre_object::PyObjectRef).cast() });
    }
    // The escape-flush undo snapshot stays armed from force time until the
    // abort epilogue consumes it, and its slots can be the ONLY reference to
    // pre-walk locals the flush displaced (the live-frame slots now hold
    // walk-end values) — forward each so a minor collection on the abort
    // unwind cannot move/free what `restore_escape_flush_undo` writes back.
    let escape_undo = unsafe { &mut *(*area.escape_flush_undo).as_ptr() };
    if let Some(undo) = escape_undo.as_mut() {
        for slot in undo.slots.iter_mut() {
            visitor(unsafe { &mut *(slot as *mut pyre_object::PyObjectRef).cast() });
        }
    }
    // The vable-force and trace-too-long MIFrame images survive the dispatch
    // unwind in TLS. Their Option<i64> Ref banks are not otherwise visible to
    // the collector, so forward every populated color until the walk-end
    // handler takes the latch. The adopters bridge the pre-drive publication
    // window and the blackhole drivers then install their own packed roots.
    let single_frame_blackhole = unsafe { &mut *(*area.single_frame_blackhole).as_ptr() };
    if let Some(latched) = single_frame_blackhole.as_mut() {
        for value in latched.miframe.ref_values.iter_mut().flatten() {
            visitor(unsafe { &mut *(value as *mut i64).cast() });
        }
        if latched.last_exc_value != 0 {
            visitor(unsafe { &mut *(&mut latched.last_exc_value as *mut i64).cast() });
        }
    }
    let multi_frame_blackhole = unsafe { &mut *(*area.multi_frame_blackhole).as_ptr() };
    if let Some(latched) = multi_frame_blackhole.as_mut() {
        for frame in latched.framestack.frames.iter_mut() {
            for value in frame.ref_values.iter_mut().flatten() {
                visitor(unsafe { &mut *(value as *mut i64).cast() });
            }
        }
        if latched.last_exc_value != 0 {
            visitor(unsafe { &mut *(&mut latched.last_exc_value as *mut i64).cast() });
        }
    }
    // The bridge/retrace iterator cursor journal holds a range iterator across
    // the rest of an authoritative bridge walk. When the parent compiled trace
    // materialized that iterator via `NewWithVtable` (`CallMallocNursery`) it is
    // nursery-resident, so a minor collection during the remaining walk moves it
    // before the non-commit rollback restores its cursor — forward each iterator
    // so `w_range_iter_set_cursor` writes through the live pointer, not a stale
    // moved one. The `(pre_current, pre_remaining)` pair are plain scalars.
    let bridge_iter = unsafe { &mut *(*area.bridge_iter).as_ptr() };
    for entry in bridge_iter.iter_mut() {
        visitor(unsafe { &mut *(&mut entry.0 as *mut pyre_object::PyObjectRef).cast() });
    }
    // gh#467: the latched forward-flush operand stack (callable + args) is
    // nursery-resident across the abort unwind — the flush boxes Int/Float
    // locals, which can trigger a minor collection that moves these refs before
    // they are written into the frame array — so forward every slot as a root.
    let abort_resume = unsafe { &mut *(*area.abort_resume).as_ptr() };
    if let Some(carrier) = abort_resume.as_mut() {
        match carrier {
            InlineAbortCarrier::Entry { call_stack, .. } => {
                for slot in call_stack {
                    visitor(unsafe { &mut *(slot as *mut pyre_object::PyObjectRef).cast() });
                }
            }
            InlineAbortCarrier::MidBody(payload) => {
                visitor(unsafe {
                    &mut *(&mut payload.w_code as *mut pyre_object::PyObjectRef).cast()
                });
                visitor(unsafe {
                    &mut *(&mut payload.w_globals as *mut pyre_object::PyObjectRef).cast()
                });
                if !payload.return_value.is_null() {
                    visitor(unsafe {
                        &mut *(&mut payload.return_value as *mut pyre_object::PyObjectRef).cast()
                    });
                }
                for slot in payload.live_locals.iter_mut().flatten() {
                    if let ConcreteValue::Ref(value) = slot {
                        visitor(unsafe { &mut *(value as *mut pyre_object::PyObjectRef).cast() });
                    }
                }
                for slot in &mut payload.live_stack {
                    if let ConcreteValue::Ref(value) = slot {
                        visitor(unsafe { &mut *(value as *mut pyre_object::PyObjectRef).cast() });
                    }
                }
                if let Some(fallback) = payload.entry_fallback.as_mut() {
                    for slot in &mut fallback.call_stack {
                        visitor(unsafe { &mut *(slot as *mut pyre_object::PyObjectRef).cast() });
                    }
                }
            }
        }
    }
}

/// #73: classification of a Python opcode's effect on the walk-level
/// symbolic operand stack ([`WalkContext::vstack_boxes`]), used by
/// [`reconcile_vstack_at_boundary`] to update `vstack_boxes` at an
/// opcode boundary.  The depth delta is already known from
/// `depth_at_py_pc`; this only decides WHERE the new boxes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VstackOpClass {
    /// A single net-result lands on the new TOS, and its box is the last
    /// Ref written via [`write_ref_reg`] during this opcode
    /// (`vstack_last_ref`).  Truncate to the new depth, then overwrite the
    /// new TOS with `vstack_last_ref`.  Covers value producers whose
    /// result is the topmost stack slot: LOAD_FAST / LOAD_CONST /
    /// LOAD_GLOBAL(result) / LOAD_NAME / LOAD_ATTR / BINARY_OP /
    /// BINARY_SUBSCR / COMPARE_OP / unary ops / CALL /
    /// IS_OP / CONTAINS_OP / single-result BUILD_*.
    ResultToTos,
    /// The opcode only pops (and/or stores to a local/global/attr/subscr,
    /// or is an unconditional control transfer).  Truncate to the new
    /// depth WITHOUT touching the surviving TOS — the box already in that
    /// slot is the live value.  Covers POP_TOP / POP_JUMP_IF_* /
    /// STORE_FAST / STORE_GLOBAL / STORE_NAME / STORE_ATTR /
    /// STORE_SUBSCR / STORE_SLICE / JUMP_* / RETURN_* / DELETE_*.
    PopOnlyOrSideStore,
    /// `SWAP(i)` permutes the operand stack: it exchanges the TOS box with
    /// the box `i` positions below it (the `localsplus[depth-1]` /
    /// `localsplus[depth-i]` exchange in `swap_values`).  Net depth is
    /// unchanged; the carried `usize` is the decoded `i`.  Reconcile applies
    /// the same exchange to `vstack_boxes`, so a value kept across a SWAP
    /// (e.g. `a + ((i & 1) or 5)` reorders its operands with SWAP before the
    /// short-circuit guard) lands on the right slot instead of latching the
    /// mirror invalid.
    Swap(usize),
    /// `COPY(i)` duplicates the box `i` positions from the top onto the new
    /// TOS (`opcode_copy_value` = `push(peek_at(i-1))`).  Net depth +1; the
    /// new TOS box is `vstack_boxes[depth-1-i]` (the COPIED slot), NOT the
    /// last Ref written — only `COPY 1` (dup of TOS, the and/or/condexpr and
    /// chained-comparison truthiness dup) coincides with `vstack_last_ref`.
    /// Reconcile sources the duplicated slot directly so `COPY i>1`
    /// (duplicating a deeper operand) is faithful.  The carried `usize` is
    /// the decoded `i`.
    Copy(usize),
    /// An exception-machinery opcode (PUSH_EXC_INFO / CHECK_EXC_MATCH /
    /// RERAISE / WITH_EXCEPT_START / RAISE_VARARGS) reached INSIDE an
    /// exception handler.  Its operand-stack effect is not a simple
    /// producer/pop — the unwinder and the exc-info machinery rewrite the
    /// stack — but the lowering writes every resulting operand slot through
    /// `setarrayitem_vable_r`, so the virtualizable shadow holds the correct
    /// post-opcode operand stack.  Reconcile truncates to the new depth and
    /// reseeds the slots from the shadow (`reseed_vstack_from_shadow`); a
    /// slot the shadow cannot source (a genuine NULL exc-info slot, an
    /// Int/Float temp) stays a NONE hole and `mirror_covers_kept` declines
    /// for it (the conservative fallback).  Reached with a valid mirror only
    /// via [`vstack_enter_exception_handler`]; without that handler-entry
    /// reseed the mirror is already invalid at the unwind boundary, so this
    /// arm is inert on the non-exception path.
    ShadowReseed,
    /// `UNPACK_SEQUENCE` / `UNPACK_EX`: pop one sequence and push its
    /// elements (net push > 1), each written through `setarrayitem_vable_r`.
    /// A single `vstack_last_ref` write cannot reconstruct the whole pushed
    /// group, but every pushed element IS in the virtualizable shadow, so
    /// reconcile clears the affected range `[pop_point .. new_depth)` (from
    /// the popped sequence slot upward) to NONE holes and the general
    /// hole-fill sources them from the shadow.  Slots BELOW the popped
    /// sequence keep their mirror-tracked boxes (the hole-fill never
    /// overwrites a non-NONE slot).  Never latches the mirror invalid.
    MultiResultFromShadow,
    /// Anything that does not fit the shapes above — FOR_ITER or any opcode
    /// this classifier does not recognise.  Latches `vstack_valid = false`
    /// so the overlay omits those slots, which resume re-materializes (zero
    /// regression).
    Unmodeled,
}

/// The frame whose JitCode byte offsets the branch-resume gate readers
/// ([`branch_resume_target_stack_depth`] and the kept-slot hazard checks)
/// resolve through.  Holds the frame's `PyJitCode` payload — its `metadata`
/// (the tables for the jitcode-pc → Python-pc inversion) and `code_ptr` (the
/// liveness key).
///
/// Two constructors mark the frame model: the gate must read the frame whose
/// jitcode the `target` offset indexes, NOT a single global.
/// * [`outer`](Self::outer) — the outermost portal/main frame held by
///   `fbw_mode.snapshot_sym`.
/// * [`current`](Self::current) — the innermost inlined callee being
///   sub-walked (framestack top), or the portal frame when no
///   sub-walk is active.  Mirrors the callee derivation in
///   `walker_capture_multi_frame_inline_snapshot` (the framestack
///   → `ensure_jitcode_index` → `pyjitcode_for_jitcode_index` chain) so the
///   gate and the snapshot encoder consult one consistent active frame.
struct ActiveResumeFrame(std::sync::Arc<crate::PyJitCode>);

impl ActiveResumeFrame {
    /// The outermost portal/main frame (`fbw_mode.snapshot_sym`).  `None`
    /// outside a full-body walk (tests or diagnostic callers).
    fn outer<Sym: WalkSym>(snapshot_sym: *const Sym) -> Option<Self> {
        let full_body_sym = snapshot_sym;
        if full_body_sym.is_null() {
            return None;
        }
        // SAFETY: identical contract to the gate readers — the pointer is set
        // only for the lifetime of the full-body `dispatch_via_miframe`, and
        // only the immutable `payload` Arc is cloned.
        let sym = unsafe { &*full_body_sym };
        if sym.jitcode().is_null() {
            return None;
        }
        let jc = unsafe { &*sym.jitcode() };
        Some(ActiveResumeFrame(jc.payload.clone()))
    }

    /// The active frame at the current walk point: the innermost inlined
    /// callee when a sub-walk is in progress, else the portal frame.
    fn current<Sym: WalkSym>(
        session: &std::cell::RefCell<WalkSession>,
        snapshot_sym: *const Sym,
    ) -> Option<Self> {
        let current_code = session.borrow().framestack.last().map(|frame| frame.w_code);
        match current_code {
            Some(callee_w_code) => {
                let idx = crate::state::ensure_jitcode_index(callee_w_code as *const ())?;
                let pjc = crate::state::pyjitcode_for_jitcode_index(idx)?;
                Some(ActiveResumeFrame(pjc))
            }
            None => Self::outer(snapshot_sym),
        }
    }

    fn vstack_coordinate_for_jitcode_pc(
        &self,
        jit_pc: usize,
    ) -> Option<(u32, *const pyre_interpreter::CodeObject, usize)> {
        let pjc = &self.0;
        if pjc.code_ptr.is_null() {
            return None;
        }
        let py_pc = vstack_containing_py_pc(&pjc.metadata, jit_pc);
        // Raw py_pc-keyed static-liveness read: the unpopulated-twin fallback
        // (skeleton / fixture) and the audit oracle.
        let raw_depth = || {
            crate::liveness::liveness_for(pjc.code_ptr)
                .depth_at_py_pc()
                .get(py_pc as usize)
                .copied()
                .unwrap_or(0) as usize
        };
        let depth = if pjc.depth_containing_populated() {
            let depth = pjc.depth_containing_for_jitcode_pc(jit_pc).unwrap_or(0) as usize;
            if pcmap_containing_audit_enabled() {
                assert_eq!(
                    depth,
                    raw_depth(),
                    "PYRE_PCMAP_CONTAINING_AUDIT: resume-frame containing-depth twin diverged at jit_pc {jit_pc} (py {py_pc})"
                );
            }
            depth
        } else {
            raw_depth()
        };
        Some((py_pc, pjc.code_ptr, depth))
    }

    fn body_matches(&self, sub_body: &SubJitCodeBody) -> bool {
        let code = self.0.jitcode.code.as_slice();
        code.len() == sub_body.code.len() && code.as_ptr() == sub_body.code.as_ptr()
    }
}

/// Walker-side port of `pyjitpl.py handle_possible_exception`'s
/// exception branch.
///
/// Emit `GuardException(exc_type_const)` + snapshot, reading the
/// exception's `ob_header.ob_type` from
/// `WalkContext::last_exc_value_concrete` to pin the class.  Mirrors
/// RPython's `class_of_box_known` shape: the guard recovery has the
/// exact subclass available at replay, matching `opimpl_raise`'s
/// `GUARD_CLASS(exc, cls_of_box(exc))` pattern (`pyjitpl.py`).
///
/// Falls back to `GuardNoException` when the concrete exception
/// pointer is unavailable (sub-walk shadow gap or non-Ref concrete) —
/// the residual call op stays in the trace and the optimizer's
/// per-call guard-emission still catches exception divergence at
/// replay, just without the class pin.
fn walker_record_guard_exception<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>, pc: usize) {
    let exc_obj = match ctx.last_exc_value_concrete {
        ConcreteValue::Ref(p) if !p.is_null() => p,
        _ => {
            ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
            let _ = walker_capture_snapshot_for_last_guard(ctx, pc);
            return;
        }
    };
    let class_of_last_exc_is_const = ctx.fbw_mode.class_of_last_exc_is_const;
    // `pyjitpl.py` / `pyjitpl.rs`: always emit
    // `GuardException` with a const class pin, but keep its live result box
    // unless the exception class was already proven constant.
    // Pyre's `W_BaseException.ob_header.ob_type` is the per-`ExcKind`
    // `PyType` static (`interp_exceptions.rs::exc_kind_to_pytype`), matching
    // upstream `OBJECT.typeptr = specific class` (`rclass.py`).
    let exc_type_ptr = unsafe {
        (*(exc_obj as *const pyre_object::interp_exceptions::W_BaseException))
            .ob_header
            .ob_type as i64
    };
    let exc_type_const = ctx.trace_ctx.const_int(exc_type_ptr);
    let guard_op = ctx
        .trace_ctx
        .record_guard(OpCode::GuardException, &[exc_type_const], 0);
    let _ = walker_capture_snapshot_for_last_guard(ctx, pc);
    // `op.setref_base(val)` supplies the recording-time shadow without
    // changing the guard result's live replay identity.
    ctx.trace_ctx.set_opref_concrete(
        guard_op,
        majit_ir::Value::Ref(majit_ir::GcRef(exc_obj as usize)),
    );
    ctx.last_exc_value = Some(if class_of_last_exc_is_const {
        ctx.trace_ctx.const_ref(exc_obj as usize as i64)
    } else {
        guard_op
    });
    ctx.fbw_mode.class_of_last_exc_is_const = true;
}

fn clear_walk_exception<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>) {
    ctx.last_exc_value = None;
    ctx.last_exc_value_concrete = ConcreteValue::Null;
}

fn direct_call_release_gil<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ei: &majit_ir::EffectInfo,
    allboxes: &[OpRef],
    descr: DescrRef,
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
    dst: usize,
    pc: usize,
    caller: &'static str,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    // pyjitpl.py `vable_and_vrefs_before_residual_call` —
    // release-gil is unconditionally a forces sub-case
    // (`pyjitpl.py` sits inside the forces-virtual-or-virtualizable
    // branch), so no `emit_guard_not_forced` gate is needed here.
    // RPython's pre-call vrefs heap mutation
    // (`pyjitpl.py vrefs_before_residual_call`) and the
    // after-call helpers (`pyjitpl.py`) are handled by the
    // residual-call execution path — see
    // [`walker_vable_and_vrefs_before_residual_call`] for the IR-vs-heap
    // split rationale.
    maybe_walker_vable_and_vrefs_before_residual_call(ctx);
    // pyjitpl.py: realfuncaddr, saveerr = effectinfo.call_release_gil_target
    let (realfuncaddr, saveerr) = ei.call_release_gil_target;
    // pyjitpl.py: funcbox/savebox ConstInt
    let savebox = ctx.trace_ctx.const_int(saveerr as i64);
    let funcbox_real = ctx.trace_ctx.const_int(realfuncaddr as i64);
    // pyjitpl.py: opnum = rop.call_release_gil_for_descr(calldescr).
    // resoperation.py maps the descr's normalized result
    // type to {CALL_RELEASE_GIL_I, CALL_RELEASE_GIL_F, CALL_RELEASE_GIL_N};
    // 'r' is explicitly skipped (`# no such thing`).
    let opcode = match dst_bank {
        'i' => OpCode::CallReleaseGilI,
        'f' => OpCode::CallReleaseGilF,
        'v' => OpCode::CallReleaseGilN,
        'r' => panic!(
            "{caller}: CALL_RELEASE_GIL_R has no upstream counterpart \
             (resoperation.py:1243-1244 `# no such thing`); a 'r'-result \
             release-gil callee cannot be lowered to an IR op the \
             optimizer/backend can consume."
        ),
        _ => unreachable!(
            "{caller}: dst_bank '{dst_bank}' not supported by direct_call_release_gil \
             (callers must pass 'i' / 'f' / 'v' per resoperation.py:1240-1248)"
        ),
    };
    // pyjitpl.py: history.record_nospec(opnum,
    //                          [savebox, funcbox] + argboxes[1:], ..., calldescr)
    let mut new_args = Vec::with_capacity(allboxes.len() + 1);
    new_args.push(savebox);
    new_args.push(funcbox_real);
    if allboxes.len() > 1 {
        new_args.extend_from_slice(&allboxes[1..]);
    }
    // Ordering note: the walker records the op here and concrete-executes
    // below, whereas `do_residual_call` executes the `CALL_MAY_FORCE_*`
    // first and only records (`direct_call_release_gil`, pyjitpl.py)
    // after `vrefs_after_residual_call()`. This record-then-execute order
    // is a structural property of the FBW residual-call path shared by the
    // generic walker — not specific to release-gil — so the result still
    // back-patches `recorded`; a 1:1 execute-then-record reorder would
    // require restructuring the whole walker residual-call flow.
    let recorded = ctx.trace_ctx.record_op_with_descr(opcode, &new_args, descr);
    // The forces-branch concrete-execute and the heapcache invalidation
    // both key on the corresponding `CALL_MAY_FORCE_*` (`pyjitpl.py-
    // 2072 opnum1`), NOT `CALL_RELEASE_GIL_*`.
    let mayforce_opnum = match dst_bank {
        'i' => OpCode::CallMayForceI,
        'f' => OpCode::CallMayForceF,
        'v' => OpCode::CallMayForceN,
        _ => unreachable!("dst_bank validated above"),
    };
    // pyjitpl.py `do_residual_call` step 2: the forces branch
    // concrete-executes the helper via `executor.execute_varargs(opnum1,
    // allboxes, descr)` BEFORE step 4 selects which CALL_* op to record.
    // The release-gil sub-case (`pyjitpl.py`) sits *inside* that
    // branch, so it is executed identically to a `CALL_MAY_FORCE_*` — on
    // the **original** `allboxes` (`allboxes[0]` is the wrapper funcbox;
    // the recorded op above used the re-shaped `[savebox, funcbox_real,
    // …]`).  This removes the asymmetry where release-gil was the only
    // forces sub-case that recorded without executing — the same
    // un-executed-side-effect SIGBUS class already closed for the
    // may-force branch.
    // `try_execute_residual_call_via_executor` self-gates (authoritative-
    // executor flag, const-funcbox, fnaddr ≥47-bit sanity) and degrades to
    // recording-only on decline; on success it stamps `recorded` with the
    // concrete result and brackets the active virtualizable token for the
    // duration of the call (its may-force arm).
    let resid_exec = try_execute_residual_call_via_executor(
        ctx,
        mayforce_opnum,
        allboxes,
        call_descr,
        recorded,
        pc,
        None,
    )?;
    // A decline leaves the call recorded symbolically WITHOUT running it, so
    // the walk-end no-replay commit must stay off for this trace.
    let resid_raised = match resid_exec {
        ResidualExecOutcome::Executed(result) => result.is_err(),
        ResidualExecOutcome::Declined(cause) => {
            fbw_abort_nested_unjournaled_residual(ctx, pc)?;
            fbw_mark_unjournaled_effect(cause);
            false
        }
    };
    debug_assert!(
        !resid_raised || ei.check_can_raise(false),
        "{caller}: release-gil helper raised on a `!can_raise` EI — \
         EffectInfo claim/reality mismatch"
    );
    // pyjitpl.py `heapcache.invalidate_caches_varargs(opnum1, descr,
    // allboxes)` — forces-branch invalidation uses `opnum1`
    // (`CALL_MAY_FORCE_*`) on the **original** `allboxes` so heapcache's
    // `mark_escaped_varargs` sees the same operand identities upstream does.
    ctx.trace_ctx
        .heapcache_invalidate_caches_varargs(mayforce_opnum, Some(ei), allboxes);
    // pyjitpl.py execute_varargs: `make_result_of_lastop(op)` runs
    // BEFORE `handle_possible_exception()` precisely "because we need the box
    // to show up in get_list_of_active_boxes()".  Write the recorded OpRef
    // into `registers_*[dst]` for every non-void result REGARDLESS of whether
    // the helper raised — the GUARD_NOT_FORCED resume snapshot captured below
    // must see the result box in its slot, exactly as upstream's
    // unconditional (non-void) make_result_of_lastop does.  `'v'` (void) is a
    // no-op in `write_residual_call_result_to_dst`; on a raised call the OpRef
    // carries a Null concrete shadow, never read on the exception path.
    if dst_bank != 'v' {
        write_residual_call_result_to_dst(ctx, pc, dst, dst_bank, recorded)?;
    }
    // pyjitpl.py GUARD_NOT_FORCED — unconditional on the outer
    // forces-virtual-or-virtualizable branch (the release-gil sub-case is
    // inside that branch).  A force inside the concrete-executed helper is
    // observed by the executor's vable token bracket above, surfacing as
    // `VableEscapedDuringResidualCall` (aborting the walk) before reaching
    // this guard.
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    // pyjitpl.py handle_possible_exception — emits GUARD_EXCEPTION
    // when the recording-time helper raised (then finishframe_exception:
    // surface `SubRaise` so the dead arm tail is not recorded onto the
    // exception path), else GUARD_NO_EXCEPTION.  The capture ports
    // `capture_resumedata(after_residual_call=True)` so the optimizer's
    // `store_final_boxes_in_guard` finds a populated `rd_resume_position`.
    if ei.check_can_raise(false) {
        if resid_raised {
            walker_record_guard_exception(ctx, pc);
            let exc = ctx
                .last_exc_value
                .expect("resid_raised implies last_exc_value seeded by the Err branch");
            let exc_concrete = ctx.last_exc_value_concrete;
            return Ok(Some(DispatchOutcome::SubRaise { exc, exc_concrete }));
        }
        ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
        walker_capture_snapshot_for_last_guard(ctx, pc)?;
    }
    Ok(None)
}

/// Parse `"0x<hex>"` or `"<decimal>"` into a `usize` address for
/// env-var-driven function-pointer gates.
fn parse_hex_or_decimal_usize(s: &str) -> Option<usize> {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        usize::from_str_radix(rest, 16).ok()
    } else {
        s.parse::<usize>().ok()
    }
}

/// Runtime resolution of the `bh_store_subscr_fn` address via
/// `pyre_interpreter::jit_trace_fnaddrs()` linear scan (cached in a
/// `OnceLock`).  Returns
/// `None` if the symbol is unregistered (which would indicate a
/// jit_fnaddr.rs regression — the path is registered at
/// `pyre-interpreter/src/jit_fnaddr.rs`).
///
/// Cached on first call.  Linear scan is acceptable because the table
/// has ~150 entries and the cache miss happens once per process.
pub(crate) fn bh_store_subscr_fn_addr_cached() -> Option<usize> {
    static CACHE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        const PATH: &str = "pyre_interpreter::opcode_ops::bh_store_subscr_fn";
        for (name, addr) in pyre_interpreter::jit_trace_fnaddrs() {
            if name == PATH {
                return Some(addr as usize);
            }
        }
        None
    })
}

/// Resolve a concrete callable pointer to `(w_code, arg_count, has_closure)`,
/// validating the `Function -> PyCode -> CodeObject` type chain at every
/// hop.  Returns `None` — decline to the orthodox residual call — when any link
/// fails to type-check.
///
/// The `callable` comes from `ctx.concrete_registers_r`, a best-effort shadow
/// that is NOT a GC root: a collection during the walk can leave the shadow
/// pointing at freed/relocated memory whose first word still happens to read
/// `FUNCTION_TYPE`.  Reading `(*callable).code` then yields a non-`PyCode`
/// (a host-allocated code wrapper never lives in the GC heap), so the
/// `CODE_TYPE` tag check rejects the stale shadow before `code_ptr` is read —
/// degrading to a residual call instead of dereferencing garbage.  Both walker
/// call levers (inline-at-residual and self-recursive `CALL_ASSEMBLER`) share
/// this resolver so the staleness guard is applied uniformly.
///
/// # Safety
/// `callable` must be a non-null pointer obtained from a `ConcreteValue::Ref`.
pub(crate) unsafe fn resolve_inlinable_callee(
    callable: pyre_object::PyObjectRef,
) -> Option<(*const (), usize, bool)> {
    unsafe {
        let function_type_addr = &pyre_interpreter::FUNCTION_TYPE as *const _ as usize;
        if !pyre_interpreter::is_function(callable)
            || (*callable).ob_type as *const () as usize != function_type_addr
        {
            return None;
        }
        let w_code = pyre_interpreter::function_get_code(callable);
        if w_code.is_null() {
            return None;
        }
        let code_type_addr = &pyre_interpreter::pycode::CODE_TYPE as *const _ as usize;
        if (*(w_code as *const pyre_object::pyobject::PyObject)).ob_type as *const () as usize
            != code_type_addr
        {
            return None;
        }
        let raw = pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject;
        if raw.is_null() {
            return None;
        }
        // Calling a generator / coroutine / async-generator function does not
        // run its body — it builds a suspended frame object and returns it.
        // Its jitcode therefore starts at `RETURN_GENERATOR`, which the
        // codewriter lowers to an `abort_permanent` marker: inlining the body
        // at the call site walks code the call never reaches, so the attempt
        // can only end in an abort that discards the whole trace.  Decline to
        // the residual call instead — the creation is an ordinary allocating
        // call the walk steps over, and the body's suspension never enters the
        // trace.  `generator.py:614-634` `should_not_inline` (consumed at
        // `:63`) is upstream's decision point for the same boundary; it governs
        // the *body* at `send_ex`, which is a separate question from the
        // creation call.
        if pyre_interpreter::pyframe::code_flags_make_generator((*raw).flags) {
            return None;
        }
        let closure = pyre_interpreter::function_get_closure(callable);
        Some((w_code, (*raw).arg_count as usize, !closure.is_null()))
    }
}

type ExceptionInlineReceiverGuard = (
    OpRef,
    pyre_object::PyObjectRef,
    pyre_object::PyObjectRef,
    u64,
);

/// `(arg, concrete_arg, w_type)` for an inlined operator's non-receiver
/// argument: the operand op to guard, its record-time concrete object (for the
/// physical `ob_type`), and the Python `W_TypeObject` the reflected-op decline
/// was computed against (pinned via the live `w_class`).
type ArgClassGuard = (OpRef, pyre_object::PyObjectRef, pyre_object::PyObjectRef);

/// `residual_call` shape `iIRd>X` dispatcher — `_ir_*` arglist with
/// both an int-bank list and a ref-bank list before the descr. RPython
/// parity: `pyjitpl.py _opimpl_residual_call2` (`@arguments`
/// argspec `"box", "boxes2", "descr", "orgpc"`) → same
/// `do_residual_or_indirect_call` body as `_call1`. The `boxes2`
/// argcode (`pyjitpl.py`) decodes two adjacent
/// count-prefixed lists into a single concatenated `argboxes` array
/// `[i_args..., r_args...]`. `_build_allboxes` (`pyjitpl.py`,
/// ported to [`build_allboxes`]) then permutes those to match
/// `descr.get_arg_types()` ABI ordering, so a callee whose `arg_types`
/// is `[REF, INT, REF, INT]` ends up with allboxes
/// `[funcbox, r_args[0], i_args[0], r_args[1], i_args[1]]`.
///
/// Operand layout `iIRd>X`:
///   1B funcptr (i) + 1B i-list count + N×1B i-regs + 1B r-list count
///   + M×1B r-regs + 2B descr + 1B `>X` dst.
///
/// EffectInfo classification + guard emission match
/// `dispatch_residual_call_iRd_kind` via [`select_residual_call_opcode`],
/// and the same release-gil ([`direct_call_release_gil`]) +
/// loop-invariant heapcache ([`loopinvariant_lookup`] /
/// [`loopinvariant_now_known`]) sub-cases route through dedicated
/// helpers ahead of the selector.
///
/// Heapcache invalidation matches `iRd_kind`:
/// `invalidate_caches_varargs(call_opcode, ei, allboxes)` after every
/// recorded call op (`pyjitpl.py _record_helper_varargs`); the
/// release-gil helper invalidates with `CALL_MAY_FORCE_*` per
/// `pyjitpl.py`. `OS_NOT_IN_TRACE` is fail-loud-guarded up front
/// via [`do_not_in_trace_call_result`] (matches `iRd_kind`). Pre-call
/// vable IR bookkeeping (`vable_and_vrefs_before_residual_call`
/// IR-only portion at `pyjitpl.py`: FORCE_TOKEN +
/// SETFIELD_GC) is wired identically to `iRd_kind` via
/// [`maybe_walker_vable_and_vrefs_before_residual_call`]; the runtime
/// heap mutations and the after-call helpers are handled by the
/// residual-call execution path.
///
/// Still missing relative to upstream — same set as `iRd_kind` and
/// blocked on the same infrastructure: `OS_JIT_FORCE_VIRTUAL`
/// short-circuit, `direct_libffi_call` / `direct_assembler_call`
/// specialization, KEEPALIVE for vablebox, `num_live`-aware
/// `capture_resumedata(after_residual_call=True)` on the guards. See
/// `dispatch_residual_call_iRd_kind`'s docstring for the per-item
/// blocking rationale.
/// Shared #57 gate for the BINARY_OP / COMPARE_OP int specialization.
/// Validates that both ref-list operands are concrete `W_IntObject` and
/// obtains the authentic boxed result via the same `execute_residual_call`
/// path the generic leg uses (so the concrete shadow holds the runtime
/// object incl. small-int caching / identity).
///
/// Returns `Some((lhs, rhs, lhs_val, rhs_val, boxed_result_i64))` when the
/// specialization may proceed, or `None` to fall through to the generic
/// `CallMayForce` record (non-int operands, non-const funcptr/args, or a
/// helper that raised).
/// Shared tail of the int/float specialization gates: pull the helper
/// `func_ptr` (the `allboxes[0]` constant) and the concrete arg words out
/// of `allboxes`, then run the helper concretely via
/// `execute_residual_call`.  Returns the authentic boxed-result pointer,
/// or `None` when an arg is non-concrete (vable sentinel / unstamped) or
/// the helper raises — both gates then defer to the generic record so the
/// Python-level `__op__` semantics are preserved.
fn walker_execute_may_force_boxed<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
) -> Option<i64> {
    if allboxes.is_empty() || !allboxes[0].is_constant() {
        return None;
    }
    let func_ptr = match ctx.trace_ctx.box_value(allboxes[0]) {
        Some(majit_ir::Value::Int(addr)) => addr,
        _ => return None,
    };
    let mut args = Vec::with_capacity(allboxes.len() - 1);
    for &arg in &allboxes[1..] {
        let v = match ctx.trace_ctx.box_value(arg) {
            Some(majit_ir::Value::Int(n)) => n,
            Some(majit_ir::Value::Ref(r)) => {
                if r == majit_ir::GcRef::NO_CONCRETE {
                    return None;
                }
                r.as_usize() as i64
            }
            Some(majit_ir::Value::Float(f)) => f.to_bits() as i64,
            Some(majit_ir::Value::Void) => 0,
            None => return None,
        };
        args.push(v);
    }
    // Execute the helper concretely for the authentic boxed result
    // (small-int caching / identity).  A raised helper (`Err`) or a NULL
    // result defers to the generic `CallMayForce` record so the
    // Python-level `__op__` semantics are preserved.
    let boxed_result_i64 =
        match majit_metainterp::executor::execute_residual_call(call_descr, func_ptr, &args) {
            Ok(result) if result != 0 => result,
            _ => return None,
        };
    Some(boxed_result_i64)
}

/// Resolve the concrete `PyObjectRef` carried by a Ref-bank operand's
/// recorded concrete, or `None` when it is the vable sentinel / null.
fn walker_concrete_ref_object<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opref: OpRef,
) -> Option<pyre_object::PyObjectRef> {
    match ctx.trace_ctx.concrete_of_opref(opref) {
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE => {
            let obj = r.as_usize() as pyre_object::PyObjectRef;
            if obj.is_null() { None } else { Some(obj) }
        }
        _ => None,
    }
}

/// B3 piece 3: resolve the walker's execution-context OpRef from the outer
/// portal `sym.frame` (via [`fbw_mode.snapshot_sym`]), recovering it off
/// the frame with `GetfieldGcR(frame, execution_context)` when
/// `sym.execution_context` is unseeded.  Mirrors the inline-frame EC
/// recovery (jitcode_dispatch.rs `try_walker_inline_self_recursive`) and
/// `MIFrame::ensure_execution_context` (`trace_opcode.rs`).
///
/// `None` outside a production full-body walk (no materialized portal sym)
/// or when the portal frame OpRef is unset — the PUSH_EXC_INFO / POP_EXCEPT
/// exc-info lowering then declines to the residual (SAFE).
fn walker_ensure_execution_context<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Option<OpRef> {
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    let sym = unsafe { &*sym_ptr };
    if !sym.execution_context().is_none() {
        return Some(sym.execution_context());
    }
    let frame = sym.frame();
    if frame.is_none() {
        return None;
    }
    let ec = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[frame],
        crate::descr::pyframe_execution_context_descr(),
    );
    Some(ec)
}

/// Eagerly recover the portal EC red before the full-body walk records its
/// first guard, caching it into `sym.execution_context`.
///
/// The portal `[frame, ec]` reds (`interp_jit.py reds = ['frame', 'ec']`)
/// are force-alived in every `-live-` op's R-bank, so every guard's resume
/// snapshot lists the EC color.  Loop / function-entry syms seed
/// `sym.execution_context = InputArgRef(1)` at `create_sym`, but a
/// bridge-from-guard sym whose ec color collides with a real frame slot is
/// left `OpRef::NONE` by `setup_bridge_sym` (state.rs), which defers
/// the recovery to `ensure_execution_context`.
///
/// The walker's snapshot-capture path
/// (`walker_capture_snapshot_for_last_guard` → `collect_outer_active_boxes`)
/// runs AFTER the guard, so recording the recovery there would place the
/// getfield after the guard that references it (a use-before-def; the resume
/// position would also stamp onto the getfield rather than the guard, leaving
/// the guard with `resume_pos = -1`).  Recover here instead — at walk entry,
/// before any opcode is dispatched and thus before any guard. When the EC is
/// already seeded, or the frame
/// itself is unset, this is a no-op.
pub(crate) fn seed_execution_context_for_walk<Sym: WalkSym>(
    sym: &mut Sym,
    trace_ctx: &mut TraceCtx,
) {
    if !sym.execution_context().is_none() || sym.frame().is_none() {
        return;
    }
    let ec = trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[sym.frame()],
        crate::descr::pyframe_execution_context_descr(),
    );
    sym.set_execution_context(ec);
}

fn walker_int_specialization_operands<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
) -> Option<(
    OpRef,
    OpRef,
    pyre_object::PyObjectRef,
    pyre_object::PyObjectRef,
    i64,
    i64,
    i64,
)> {
    if r_args.len() != 2 {
        return None;
    }
    let lhs = r_args[0];
    let rhs = r_args[1];
    let lhs_obj = walker_concrete_ref_object(ctx, lhs)?;
    let rhs_obj = walker_concrete_ref_object(ctx, rhs)?;
    let (lhs_val, rhs_val) = unsafe {
        // `bool` is a `W_IntObject` subclass sharing the 8-byte `intval` at
        // offset 16; the consumer unboxes it through its own `&BOOL_TYPE`
        // guard, so it stays on the int path.  Returns the concrete objects
        // so the consumer can pick the per-operand class/descr.
        if !pyre_object::is_int(lhs_obj) || !pyre_object::is_int(rhs_obj) {
            return None;
        }
        // A numeric subclass keeps the builtin `ob_type` layout while its
        // Python-visible class lives in `w_class`.  The raw int specialization
        // bypasses special-method dispatch, so only exact builtin ints/bools
        // may enter it; subclasses continue through the residual BINARY_OP.
        if !pyre_object::is_exact_builtin_instance(lhs_obj)
            || !pyre_object::is_exact_builtin_instance(rhs_obj)
        {
            return None;
        }
        (
            pyre_object::w_int_get_value(lhs_obj),
            pyre_object::w_int_get_value(rhs_obj),
        )
    };
    let boxed_result_i64 = walker_execute_may_force_boxed(ctx, allboxes, call_descr)?;
    Some((
        lhs,
        rhs,
        lhs_obj,
        rhs_obj,
        lhs_val,
        rhs_val,
        boxed_result_i64,
    ))
}

/// Float counterpart of [`walker_int_specialization_operands`].  Each
/// operand must be a concrete int or float (long → `None`: the long→float
/// cast can lose precision, matching the former float fast path's long
/// fallback).  Two ints → `None` so they route through the int
/// specialization (int `__op__`, not float).  Returns the per-operand
/// `is_int` flag + coerced `f64` value alongside the authentic boxed
/// result.
fn walker_float_specialization_operands<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
) -> Option<(
    OpRef,
    OpRef,
    pyre_object::PyObjectRef,
    pyre_object::PyObjectRef,
    bool,
    bool,
    f64,
    f64,
    i64,
)> {
    if r_args.len() != 2 {
        return None;
    }
    let lhs = r_args[0];
    let rhs = r_args[1];
    let lhs_obj = walker_concrete_ref_object(ctx, lhs)?;
    let rhs_obj = walker_concrete_ref_object(ctx, rhs)?;
    let coerce = |obj: pyre_object::PyObjectRef| -> Option<(bool, f64)> {
        unsafe {
            // `bool` is a `W_IntObject` subclass sharing `intval`; it coerces
            // through the int arm via its own &BOOL_TYPE guard at the consumer.
            if pyre_object::is_int(obj) {
                Some((true, pyre_object::w_int_get_value(obj) as f64))
            } else if pyre_object::is_float(obj) {
                Some((false, pyre_object::w_float_get_value(obj)))
            } else {
                None
            }
        }
    };
    let (lhs_is_int, lhs_f64) = coerce(lhs_obj)?;
    let (rhs_is_int, rhs_f64) = coerce(rhs_obj)?;
    // A numeric subclass keeps the builtin `ob_type` layout while its
    // Python-visible class lives in `w_class`.  The raw float specialization
    // bypasses special-method dispatch — and the `guard_class` it emits reads
    // `ob_type`, so it would not catch the subclass at runtime either — so only
    // exact builtin floats/ints/bools may enter it; subclasses continue through
    // the residual BINARY_OP / COMPARE_OP.  Mirrors the same check in
    // `walker_int_specialization_operands`.
    unsafe {
        if !pyre_object::is_exact_builtin_instance(lhs_obj)
            || !pyre_object::is_exact_builtin_instance(rhs_obj)
        {
            return None;
        }
    }
    if lhs_is_int && rhs_is_int {
        return None;
    }
    let boxed_result_i64 = walker_execute_may_force_boxed(ctx, allboxes, call_descr)?;
    Some((
        lhs,
        rhs,
        lhs_obj,
        rhs_obj,
        lhs_is_int,
        rhs_is_int,
        lhs_f64,
        rhs_f64,
        boxed_result_i64,
    ))
}

/// Walker-native unbox of a boxed `W_IntObject` operand: `GUARD_CLASS`
/// (with the walker snapshot) when the operand's class is not yet known,
/// then the ctx-only `trace_unbox_int` getfield.  Mirrors
/// `trace_unbox_int_with_resume` (`state.rs`) with the guard emitted
/// walker-native (`record_guard` + `walker_capture_snapshot_for_last_guard`)
/// instead of via `MIFrame::generate_guard` — the full-body walk has
/// decomposed `MIFrame` into the reborrowed sym slices held by
/// `WalkContext`, so reconstructing an `MIFrame` here would alias them.
fn walker_unbox_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    int_type_addr: i64,
) -> Result<OpRef, DispatchError> {
    walker_unbox_int_typed(
        ctx,
        op_pc,
        obj,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )
}

/// True when `obj` is an InputArg the concrete boundary GUARANTEES was
/// converted from a tagged immediate to a heap `W_IntObject` — i.e. a
/// frame LOCALS-region array-item InputArg (`raw() - vable_array_base <
/// nlocals`) of the active root/loop trace.
///
/// The boundary only converts the locals region: `untag_tagged_frame_locals`
/// (`eval.rs`) loops `0..frame.nlocals()`, and the record seed (`state.rs`
/// `init_symbolic` array-item pass) converts only `item_idx < nlocals`. The
/// `locals_cells_stack_w` array laid out behind those InputArgs also holds the
/// VALUE-STACK tail (`nlocals..live_prefix`), which the boundary does NOT
/// touch — so a value-stack-slot InputArg can still arrive tagged on a later
/// entry and MUST keep its defensive tag guard.
///
/// Returns `false` for:
///   * a scalar InputArg (`frame`/`ec`/`last_instr`/... at `raw() <
///     vable_array_base`) — never a boxed int operand, but classified
///     conservatively;
///   * a value-stack-slot array-item InputArg (`slot >= nlocals`);
///   * a bridge trace's InputArg (no `vable_array_base`), whose boxes come
///     from the guard-fail resume stream, not the converted portal frame;
///   * a non-InputArg (op result / const / temp).
///
/// Reads the active portal sym via `fbw_mode.snapshot_sym` (the same
/// source `reseed_vstack_from_shadow` uses for `nlocals`); `false` when the
/// pointer is null (no materialized portal, as in tests), which
/// keeps the guard (correct-but-conservative).
fn walker_inputarg_is_converted_local<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    obj: OpRef,
) -> bool {
    if !obj.is_input_arg() {
        return false;
    }
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return false;
    }
    // SAFETY: pointer live for the full-body walk; read-only nlocals /
    // vable_array_base.
    let (nlocals, base) = unsafe { ((*sym_ptr).nlocals(), (*sym_ptr).vable_array_base()) };
    let Some(base) = base else {
        return false;
    };
    let raw = obj.raw();
    raw >= base && ((raw - base) as usize) < nlocals
}

/// [`walker_unbox_int`] with an explicit `intval` descr so a `bool` operand
/// can guard its own `&BOOL_TYPE` and read through `bool_intval_descr`
/// (`bool` shares `W_IntObject`'s `intval` field).
fn walker_unbox_int_typed<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    type_addr: i64,
    intval_descr: majit_ir::DescrRef,
) -> Result<OpRef, DispatchError> {
    // A known-class operand is provably a heap `W_IntObject` — a prior
    // `GuardClass` derefed its `ob_type`, impossible on a tagged immediate — so
    // it needs neither the tag test nor a repeated `GuardClass`.  Skipping the
    // tag block for it is also what keeps a JIT-made heap box (e.g. a `wrapint`
    // result) from emitting a `CastPtrToInt` lowbit test that would force the
    // box out of virtuality in the loop.
    if pyre_object::tagged_int::CAN_BE_TAGGED && !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        if let Some(o) = walker_concrete_ref_object(ctx, obj) {
            if pyre_object::tagged_int::is_tagged_int(o) {
                let lowbit = crate::helpers::emit_tag_lowbit_test(ctx.trace_ctx, obj, true);
                walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[lowbit])?;
                return Ok(crate::helpers::emit_untag_int(
                    ctx.trace_ctx,
                    obj,
                    pyre_object::tagged_int::untag_int(o),
                ));
            } else if !walker_inputarg_is_converted_local(ctx, obj) {
                // Emit the defensive `GuardFalse(lowbit)` for every heap-concrete
                // operand EXCEPT a frame LOCALS-region InputArg (handled below):
                //   * a NON-InputArg (a residual-call result box) can still
                //     receive a tagged immediate on a future arrival;
                //   * a value-stack-slot InputArg (`slot >= nlocals`) and a
                //     bridge InputArg (no vable layout) are NOT covered by the
                //     concrete boundary, which converts only the LOCALS region
                //     (`untag_tagged_frame_locals` loops `0..frame.nlocals()`;
                //     the record seed converts `item_idx < nlocals`), so a later
                //     entry could deliver a tagged immediate to that slot.
                // With the guard the boxed leg then deopts (not faults) on a
                // tagged arrival.
                let lowbit = crate::helpers::emit_tag_lowbit_test(ctx.trace_ctx, obj, false);
                walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[lowbit])?;
            }
            // A heap-concrete LOCALS-region InputArg is an entry/loop-carried
            // frame local that the concrete boundary
            // (`untag_tagged_frame_locals` at compiled-loop entry + the
            // trace-record seed) already converted to a heap `W_IntObject`, and
            // re-converts on every subsequent entry, so no tagged immediate ever
            // reaches it. Emit NO tag test for it: the `CastPtrToInt`+`IntAnd`+
            // `GuardFalse` does not forward through the loop-close `SetfieldGc`,
            // so it would survive the unroll and pin a per-iteration rebox in the
            // steady loop. Skipping it yields the flag-false `GuardClass`+
            // immutable `GetfieldGc` shape (raw carry).
        }
    }
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, type_addr);
    }
    Ok(crate::trace_unbox_int(
        ctx.trace_ctx,
        obj,
        type_addr,
        crate::descr::ob_type_descr(),
        intval_descr,
    ))
}

/// Walker-native unbox of a fits-int `W_LongObject` operand into a raw i64 —
/// the long analogue of [`walker_unbox_int`], mirroring
/// [`crate::state::trace_unbox_long_with_resume`] but emitting the walker
/// snapshot guards.  `is_plain_int1` accepts a `W_LongObject` whose BigInt
/// fits a machine word (`type(w_obj) is W_LongObject and w_obj._fits_int()`,
/// listobject.py), and `plain_int_w` unwraps it via `_int_w`:
///
/// 1. `GUARD_CLASS(obj, LONG_TYPE)` when the class is not already known.
/// 2. `residual_call(jit_w_long_fits_int, obj)` + `GUARD_TRUE` — a later
///    execution may see the BigInt grown out of i64 range, so re-probe and
///    deopt to the residual on a non-fitting arrival.
/// 3. `residual_call(jit_w_long_toint, obj)` — `rbigint.toint()`, statically
///    non-raising after the fits guard.
///
/// A `W_LongObject` is a distinct 8-aligned heap struct and is never a tagged
/// immediate, so the `GUARD_CLASS` deref needs no lowbit pre-test.
fn walker_unbox_long<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    long_type_addr: i64,
) -> Result<OpRef, DispatchError> {
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(long_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, long_type_addr);
    }
    let obj_concrete = walker_concrete_ref_object(ctx, obj);
    let fits_fn = pyre_object::longobject::jit_w_long_fits_int as *const ();
    let fits = ctx.trace_ctx.call_typed_with_effect(
        OpCode::CallI,
        fits_fn,
        &[obj],
        &[majit_ir::Type::Ref],
        majit_ir::Type::Int,
        majit_metainterp::cannot_raise_effect_info(),
    );
    if let Some(o) = obj_concrete {
        let fits_concrete = pyre_object::longobject::jit_w_long_fits_int(o as usize as i64);
        ctx.trace_ctx
            .set_opref_concrete(fits, majit_ir::Value::Int(fits_concrete));
    }
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[fits])?;
    let toint_fn = pyre_object::longobject::jit_w_long_toint as *const ();
    let raw = ctx.trace_ctx.call_typed_with_effect(
        OpCode::CallI,
        toint_fn,
        &[obj],
        &[majit_ir::Type::Ref],
        majit_ir::Type::Int,
        majit_metainterp::cannot_raise_effect_info(),
    );
    if let Some(o) = obj_concrete {
        let v = pyre_object::longobject::jit_w_long_toint(o as usize as i64);
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(v));
    }
    Ok(raw)
}

/// Walker-native unbox of a boxed `W_FloatObject` operand: the float
/// analogue of [`walker_unbox_int`].  `GUARD_CLASS` (with the walker
/// snapshot) when the operand's class is not yet known, then the ctx-only
/// `trace_unbox_float` getfield (its own `is_class_known` check is then a
/// no-op because `class_now_known` was just set).
fn walker_unbox_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    float_type_addr: i64,
) -> Result<OpRef, DispatchError> {
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(float_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, float_type_addr);
    }
    Ok(crate::trace_unbox_float(
        ctx.trace_ctx,
        obj,
        float_type_addr,
        crate::descr::ob_type_descr(),
        crate::descr::float_floatval_descr(),
    ))
}

/// floatobject.py:139 `int_between(-1, i2 >> 48, 1)` — true while a double
/// represents `value` exactly, doubles carrying at least 48 bits of precision.
/// A wider int must be compared through its bigint, not through a rounded
/// double.
pub(crate) fn int_is_exact_as_float(value: i64) -> bool {
    (-1..1).contains(&(value >> 48))
}

/// Emit [`int_is_exact_as_float`] as a trace precondition, lowered exactly as
/// `int_between` is (`INT_SUB` + `UINT_LT`), so a re-entered trace carrying a
/// wider int bails out to the exact comparison instead of comparing rounded
/// doubles.
fn walker_guard_int_exact_as_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw_int: OpRef,
    value: i64,
) -> Result<(), DispatchError> {
    let shift = ctx.trace_ctx.const_int(48);
    let top = ctx
        .trace_ctx
        .record_op(OpCode::IntRshift, &[raw_int, shift]);
    ctx.trace_ctx
        .set_opref_concrete(top, majit_ir::Value::Int(value >> 48));
    let low = ctx.trace_ctx.const_int(-1);
    let offset = ctx.trace_ctx.record_op(OpCode::IntSub, &[top, low]);
    ctx.trace_ctx
        .set_opref_concrete(offset, majit_ir::Value::Int((value >> 48) + 1));
    let width = ctx.trace_ctx.const_int(2);
    let fits = ctx.trace_ctx.record_op(OpCode::UintLt, &[offset, width]);
    ctx.trace_ctx
        .set_opref_concrete(fits, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[fits])
}

/// Coerce a boxed operand to a raw `f64` OpRef for a float op: float →
/// `walker_unbox_float`; int → `walker_unbox_int` then `cast_int_to_float`
/// (`space.float_w` dispatches int through int2float).  Stamps the result
/// with the already-known concrete `val` so downstream `box_value` sees it.
/// Shared by the float-binary and float-compare specializations.
///
/// `exact_int` additionally guards that an int operand converts losslessly.
/// A float arithmetic operand legitimately rounds (`space.float_w`), but a
/// comparison is decided against the int's exact value (`_compare`), so only
/// the compare specialization sets it.
fn walker_coerce_operand_to_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    is_int: bool,
    val: f64,
    exact_int: bool,
) -> Result<OpRef, DispatchError> {
    let raw = if is_int {
        // bool shares int's `intval`; guard its own &BOOL_TYPE before the cast.
        let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(concrete_obj);
        let raw_int = walker_unbox_int_typed(ctx, op_pc, obj, type_addr, descr)?;
        if exact_int {
            let value = unsafe { pyre_object::w_int_get_value(concrete_obj) };
            walker_guard_int_exact_as_float(ctx, op_pc, raw_int, value)?;
        }
        ctx.trace_ctx.record_op(OpCode::CastIntToFloat, &[raw_int])
    } else {
        let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
        walker_unbox_float(ctx, op_pc, obj, float_type_addr)?
    };
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Float(val));
    Ok(raw)
}

/// Emit a walker-native guard (`record_guard` + the walker snapshot for
/// the just-recorded guard).  Mirrors `MIFrame::generate_guard` for the
/// full-body walk.
fn walker_emit_guard_with_snapshot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    opcode: OpCode,
    args: &[OpRef],
) -> Result<(), DispatchError> {
    ctx.trace_ctx.record_guard(opcode, args, 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)
}

/// Fold-specific guard snapshot: records the guard and delegates to the
/// standard `walker_capture_snapshot_for_last_guard` which handles both the
/// FBW path (fresh `collect_outer_active_boxes` from `fbw_mode.snapshot_sym`)
/// and the per-opcode arm path (`ctx.outer_active_boxes`).
///
/// Previous attempt used `ctx.outer_active_boxes` directly, which is correct
/// for the per-opcode arm entry but empty (`Vec::new()`) in the main FBW
/// walk (`dispatch_via_miframe`).  The FBW path in
/// `walker_capture_snapshot_for_last_guard_impl` re-derives `py_pc` from
/// `op_pc` and computes a fresh active-box set per guard, matching the
/// decoder's liveness query.
fn walker_emit_fold_guard_with_snapshot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    opcode: OpCode,
    args: &[OpRef],
) -> Result<(), DispatchError> {
    ctx.trace_ctx.record_guard(opcode, args, 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)
}

fn walker_flush_guard_not_invalidated<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    if ctx.trace_ctx.pending_guard_not_invalidated_pc().is_some() {
        ctx.trace_ctx.set_pending_guard_not_invalidated(None);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNotInvalidated, &[])?;
    }
    Ok(())
}

/// Record `int_eq(raw, const k)` and stamp its already-known concrete
/// truth.  Used to build the div/mod precondition guards walker-native.
fn walker_int_eq_const<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    raw: OpRef,
    k: i64,
    concrete_truth: i64,
) -> OpRef {
    let k_const = ctx.trace_ctx.const_int(k);
    let r = ctx.trace_ctx.record_op(OpCode::IntEq, &[raw, k_const]);
    ctx.trace_ctx
        .set_opref_concrete(r, majit_ir::Value::Int(concrete_truth));
    r
}

/// Record `uint_lt(raw, const k)` and stamp its already-known concrete truth.
/// Used to guard a machine shift count into `[0, k)`: the x86 SHL/SAR encoding
/// masks the count mod 64, so a reused trace whose shift count leaves the range
/// must bail to the generic (bignum-capable) leg rather than shift by `count &
/// 63`. `uint_lt` folds the negative case in (a negative count reads as a huge
/// unsigned value `>= k`).
fn walker_uint_lt_const<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    raw: OpRef,
    k: i64,
    concrete_truth: i64,
) -> OpRef {
    let k_const = ctx.trace_ctx.const_int(k);
    let r = ctx.trace_ctx.record_op(OpCode::UintLt, &[raw, k_const]);
    ctx.trace_ctx
        .set_opref_concrete(r, majit_ir::Value::Int(concrete_truth));
    r
}

/// Record `float_eq(raw, const k)` and stamp its already-known concrete
/// truth.  Used to build the float-div zero-divisor precondition guard
/// walker-native (the JIT representation of `floatobject.py _floatdiv`'s
/// `if y == 0.0: raise ZeroDivisionError`).
fn walker_float_eq_const<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    raw: OpRef,
    k: f64,
    concrete_truth: i64,
) -> OpRef {
    let k_const = ctx.trace_ctx.const_float(k.to_bits() as i64);
    let r = ctx.trace_ctx.record_op(OpCode::FloatEq, &[raw, k_const]);
    ctx.trace_ctx
        .set_opref_concrete(r, majit_ir::Value::Int(concrete_truth));
    r
}

/// ll_math.py `VERY_LARGE_FLOAT`: the smallest power of 64 whose
/// `* 100.0` overflows to infinity.  `ll_math_isinf` (ll_math.py)
/// tests `(y + VERY_LARGE_FLOAT) == y` when jitted — one add plus one
/// compare instead of two ±inf equality checks.
fn very_large_float() -> f64 {
    let mut f = 1.0f64;
    while f * 100.0 != f64::INFINITY {
        f *= 64.0;
    }
    f
}

/// Record a float comparison with its already-known concrete truth, then
/// pin the observed direction with `GuardTrue`/`GuardFalse` (walker
/// snapshot at `op_pc`).
fn walker_float_cmp_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    opcode: OpCode,
    args: &[OpRef],
    truth: bool,
) -> Result<(), DispatchError> {
    let c = ctx.trace_ctx.record_op(opcode, args);
    ctx.trace_ctx
        .set_opref_concrete(c, majit_ir::Value::Int(truth as i64));
    let guard = if truth {
        OpCode::GuardTrue
    } else {
        OpCode::GuardFalse
    };
    walker_emit_guard_with_snapshot(ctx, op_pc, guard, &[c])
}

/// Inline trace of `_pow` (floatobject.py, ported as
/// `float_pow_inner`) for its fast paths: `y == 2.0` (`float_mul`),
/// `y == 0.0` / `bx == 1.0` (constant result), and the mainstream
/// finite `x >= 0` case.  Each `if` on the way is recorded as a float
/// comparison pinned by a guard in the concretely-observed direction —
/// the shape the meta-tracer produces upstream, where the y-dependent
/// checks const-fold when `y` comes from LOAD_CONST — and only the raw
/// libm pow remains as a residual `call_f(ccall_pow, x, y)`
/// (ll_math.py `math_pow`, EF_CANNOT_RAISE: no `guard_no_exception`).
///
/// Cold branches return `Ok(None)` before emitting anything and stay on
/// the opaque `float_pow_jit` leg: taken isnan/isinf arms and the
/// negative-base arm need fmod/floor/copysign lowerings (lloperation has
/// no float_mod llop — those are residual calls upstream too).
///
/// Guard soundness: pow is pure, so a guard failing even after the
/// residual call deopts to `op_pc` and re-executes the whole BINARY_OP in
/// the interpreter.  The raising cases are all behind such guards:
/// `0.0 ** negative` (ZeroDivisionError) fails the `x == 0.0` or
/// `y < 0.0` guard, negative-base-fractional (PowDomainError → complex)
/// fails the `x < 0.0` guard, and overflow (OverflowError,
/// floatobject.py `isinf(z) and not isinf(bx)`) fails the
/// trailing isfinite guard — `bx` is already pinned finite, so the check
/// reduces to `isinf(z)`, emitted as `ll_math_isfinite`'s jitted form
/// `(z - z) == 0.0` (ll_math.py).
fn walker_emit_float_pow_inline<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    x: OpRef,
    y: OpRef,
    lx: f64,
    ly: f64,
    result_val: f64,
) -> Result<Option<OpRef>, DispatchError> {
    // Decline the cold paths before any emission so the generic leg
    // starts from a clean trace.  (`lx >= 0.0` is false for NaN, but the
    // isnan case is spelled out to mirror the branch list above.)
    let tame = !lx.is_nan() && !ly.is_nan() && !lx.is_infinite() && !ly.is_infinite() && lx >= 0.0;
    if ly != 2.0 && ly != 0.0 && !tame {
        return Ok(None);
    }

    // floatobject.py  if y == 2.0: return x * x
    let two = ctx.trace_ctx.const_float(2.0f64.to_bits() as i64);
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatEq, &[y, two], ly == 2.0)?;
    if ly == 2.0 {
        let r = ctx.trace_ctx.record_op(OpCode::FloatMul, &[x, x]);
        ctx.trace_ctx
            .set_opref_concrete(r, majit_ir::Value::Float(result_val));
        return Ok(Some(r));
    }
    // floatobject.py  if y == 0.0: return 1.0
    let zero = ctx.trace_ctx.const_float(0.0f64.to_bits() as i64);
    let one = ctx.trace_ctx.const_float(1.0f64.to_bits() as i64);
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatEq, &[y, zero], ly == 0.0)?;
    if ly == 0.0 {
        return Ok(Some(one));
    }
    // floatobject.py  if isnan(x)  (ll_math_isnan: x != x)
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatNe, &[x, x], false)?;
    // floatobject.py  if isnan(y)
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatNe, &[y, y], false)?;
    // floatobject.py  if isinf(y)
    let vlf_val = very_large_float();
    let vlf = ctx.trace_ctx.const_float(vlf_val.to_bits() as i64);
    let t = ctx.trace_ctx.record_op(OpCode::FloatAdd, &[y, vlf]);
    ctx.trace_ctx
        .set_opref_concrete(t, majit_ir::Value::Float(ly + vlf_val));
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatEq, &[t, y], false)?;
    // floatobject.py  if isinf(x)
    let t = ctx.trace_ctx.record_op(OpCode::FloatAdd, &[x, vlf]);
    ctx.trace_ctx
        .set_opref_concrete(t, majit_ir::Value::Float(lx + vlf_val));
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatEq, &[t, x], false)?;
    // floatobject.py  if x == 0.0 and y < 0.0: ZeroDivisionError
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatEq, &[x, zero], lx == 0.0)?;
    if lx == 0.0 {
        // The raising direction never reaches here: the concrete helper
        // execution would have raised and declined the specialization.
        debug_assert!(ly >= 0.0);
        walker_float_cmp_guard(ctx, op_pc, OpCode::FloatLt, &[y, zero], false)?;
    }
    // floatobject.py  if bx < 0.0  (cold: declined above)
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatLt, &[x, zero], false)?;
    // floatobject.py  if bx == 1.0 (negate_result is false here)
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatEq, &[x, one], lx == 1.0)?;
    if lx == 1.0 {
        return Ok(Some(one));
    }
    // floatobject.py  z = math.pow(bx, y) — the residual raw libm call
    let z = ctx.trace_ctx.call_float_typed_with_effect(
        crate::trace_opcode::ccall_pow as *const (),
        &[x, y],
        &[majit_ir::Type::Float, majit_ir::Type::Float],
        majit_metainterp::CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
    );
    ctx.trace_ctx
        .set_opref_concrete(z, majit_ir::Value::Float(result_val));
    // floatobject.py  if isinf(z) and not isinf(bx): OverflowError
    let d = ctx.trace_ctx.record_op(OpCode::FloatSub, &[z, z]);
    ctx.trace_ctx
        .set_opref_concrete(d, majit_ir::Value::Float(result_val - result_val));
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatEq, &[d, zero], true)?;
    // floatobject.py  negate_result is false on this path.
    Ok(Some(z))
}

/// Box a raw i64 int result. Under `CAN_BE_TAGGED`, when the recorded
/// concrete `value` fits the tagged range, emit an immediate
/// `(value<<1)|1` (rtagged.py ll_int_to_unboxed) guarded by
/// `IntAddOvf(raw,raw)` + `GuardNoOverflow` — the doubling overflow IS the
/// fits check, and a runtime value that does not fit deopts instead of
/// producing a wrong box. Otherwise (or flag off) fall back to the heap
/// `wrapint` box. The caller stamps the Ref concrete.
fn walker_box_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw: OpRef,
    value: i64,
) -> Result<OpRef, DispatchError> {
    let _ = (op_pc, value);
    Ok(crate::state::wrapint(ctx.trace_ctx, raw))
}

/// Concrete counterpart to [`walker_box_int`]. The op is now a heap
/// `NewWithVtable`, so the stamped concrete must ALSO be a heap ptr. When the
/// residual result is tagged (flag-true), materialize a real heap
/// `W_IntObject` via `w_int_new_unique` for the concrete so
/// `op(heap) == concrete(heap)`.
fn box_int_concrete(value: i64, runtime_ptr: i64) -> majit_ir::Value {
    let ptr = if pyre_object::tagged_int::CAN_BE_TAGGED
        && pyre_object::tagged_int::is_tagged_int(runtime_ptr as pyre_object::PyObjectRef)
    {
        pyre_object::intobject::w_int_new_unique(value) as i64
    } else {
        runtime_ptr
    };
    majit_ir::Value::Ref(majit_ir::GcRef(ptr as usize))
}

/// Emit a walker-native `GUARD_CLASS(obj, type_addr)` (with the walker
/// snapshot) when `obj`'s class is not yet known, then record it as known.
/// The guard-only counterpart of [`walker_unbox_int_typed`] for operands
/// that are passed to a residual call by reference rather than unboxed.
fn walker_guard_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    type_addr: i64,
) -> Result<(), DispatchError> {
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        // `GuardClass` reads `ob_type` off `obj` (rpython/jit/backend/x86/
        // assembler.py `_cmp_guard_class` derefs the pointer with no tag
        // test), so the frontend must not hand it a tagged immediate. `obj`
        // here is a concrete heap box at record time (callers gate on
        // `is_long`), but the trace is reused for operands that arrive tagged
        // on a later entry. Emit the low-bit `GuardFalse` first — mirroring
        // `walker_unbox_int_typed`'s boxed leg — so a tagged arrival deopts
        // instead of faulting on the class deref.
        if pyre_object::tagged_int::CAN_BE_TAGGED
            && walker_concrete_ref_object(ctx, obj)
                .is_some_and(|o| !pyre_object::tagged_int::is_tagged_int(o))
        {
            let lowbit = crate::helpers::emit_tag_lowbit_test(ctx.trace_ctx, obj, false);
            walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[lowbit])?;
        }
        let type_const = ctx.trace_ctx.const_int(type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, type_addr);
    }
    Ok(())
}

/// Guard the fixed-layout instance representation, the receiver type's live
/// version tag, and the exact map shape used by the mapdict attribute folds.
/// `INSTANCE_TYPE` proves the receiver has `W_ObjectObject` fields; the
/// promoted map identity pins its class and storage coordinates
/// (mapdict.py).
fn walker_guard_mapdict_instance_shape<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_type: pyre_object::PyObjectRef,
    version_tag: u64,
    map: pyre_interpreter::objspace::std::mapdict::MapRef,
) -> Result<(), DispatchError> {
    let instance_type_addr = &pyre_object::pyobject::INSTANCE_TYPE as *const _ as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(instance_type_addr);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, instance_type_addr);
    }

    // The instance map pins the storage layout, but class mutation can change
    // lookup precedence without changing that map. Re-read the receiver type's
    // live version tag at every trace entry and deopt before the folded storage
    // access when it changes.
    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    let version_op = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        w_type_const,
        crate::descr::type_version_tag_descr(),
    );
    let version_const = ctx.trace_ctx.const_int(version_tag as i64);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardValue,
        &[version_op, version_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(version_op, version_const);

    // guard_value(getfield_gc_i(obj, map), C_map): `jit.promote(self.map)`
    // (`mapdict.py`).  The map nodes are interned + immortal, so the
    // pointer is a stable identity guarded as an opaque word (object_map_descr
    // is Int-typed).
    //
    // The guard may only be elided when the map read is ALREADY a compile-time
    // constant — i.e. a prior promotion in this trace pinned it via
    // `replace_box`.  It must NOT be elided merely because `box_value(map_op)`
    // reports the concrete map: every traced getfield op carries its live value
    // (`opimpl_getfield_gc_i` -> `set_opref_concrete`, `history.py`
    // FrontendOp), so `box_value == map` holds for the very first read and
    // would drop the guard on the trace's entry.  A trace whose map guard is
    // dropped reads `storage[storageindex]` off any same-class receiver whose
    // map differs (GuardClass alone does not pin the layout), returning a wild
    // slot value.  Pin the map with `replace_box` after guarding so a later
    // fold on the same receiver correctly elides (matching the trait
    // `implement_guard_value`).
    let map_op =
        crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, obj, crate::descr::object_map_descr());
    if !map_op.is_constant() {
        let map_const = ctx.trace_ctx.const_int(map as i64);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[map_op, map_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(map_op, map_const);
    }
    Ok(())
}

/// Pin every branch input preceding a raw typed exception-slot arm.
/// `GuardClass` fixes the kind-specific `W_BaseException` layout and kind tag;
/// `GuardValue(getfield(w_class))` distinguishes heap subclasses sharing that
/// layout; and the class `version_tag` guard pins the preceding type-dict miss.
/// Exception `w_dict` lookup follows these arms in `baseobjspace.rs`, so it is
/// intentionally not part of the guard set.  This is the same promoted-class
/// lookup shape used by the mapdict folds (`mapdict.py`).
fn walker_guard_exception_attr_slot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    w_type: pyre_object::PyObjectRef,
    version_tag: u64,
) -> Result<(), DispatchError> {
    let physical_type = unsafe { (*concrete_obj).ob_type } as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(physical_type);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, physical_type);
    }
    let live_w_class =
        crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, crate::descr::w_class_descr());
    let w_class_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardValue,
        &[live_w_class, w_class_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(live_w_class, w_class_const);

    let type_const = ctx.trace_ctx.const_ref(w_type as i64);
    let live_version = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        type_const,
        crate::descr::type_version_tag_descr(),
    );
    let version_const = ctx.trace_ctx.const_int(version_tag as i64);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardValue,
        &[live_version, version_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(live_version, version_const);
    Ok(())
}

fn walker_load_name_from_code(w_code_ptr: usize, name_idx: usize) -> Option<String> {
    unsafe {
        let code_ptr = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef);
        if code_ptr.is_null() {
            return None;
        }
        let code = &*(code_ptr as *const pyre_interpreter::CodeObject);
        pyre_interpreter::pyframe::load_name_from_code(code, name_idx).map(ToString::to_string)
    }
}

fn walker_record_getfield_gc_i_uncached<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    obj: OpRef,
    descr: DescrRef,
) -> OpRef {
    // Plain GETFIELD_GC_I regardless of purity; OptHeap re-derives
    // purity from the descr. There is no pure getfield opnum.
    ctx.trace_ctx
        .record_op_with_descr(OpCode::GetfieldGcI, &[obj], descr)
}

fn walker_record_getfield_gc_r_uncached<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    obj: OpRef,
    descr: DescrRef,
) -> OpRef {
    // Plain GETFIELD_GC_R regardless of purity; OptHeap re-derives
    // purity from the descr. There is no pure getfield opnum.
    ctx.trace_ctx
        .record_op_with_descr(OpCode::GetfieldGcR, &[obj], descr)
}

/// True only when this `LoadAttr` residual is immediately consumed by the
/// paired `load_method_self(obj, attr, code, name_idx)` residual emitted for
/// `LOAD_METHOD`.  Plain `LOAD_ATTR` of a function descriptor must keep the
/// normal bound-method semantics and cannot be rewritten to the raw function.
fn next_op_is_load_method_self_for_attr<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &WalkContext<'_, '_, Sym>,
    attr_dst_reg: usize,
) -> bool {
    let Some(mut next) = crate::jitcode_runtime::decode_op_at(code, op.next_pc) else {
        return false;
    };
    while next.opname == "live"
        || next.opname.starts_with("setarrayitem_vable")
        || next.opname.starts_with("setfield_vable")
    {
        let Some(after_live) = crate::jitcode_runtime::decode_op_at(code, next.next_pc) else {
            return false;
        };
        next = after_live;
    }
    if next.key != "residual_call_ir_r/iIRd>r" {
        return false;
    }
    let i_len_pc = next.pc + 2;
    let Some(&i_len_byte) = code.get(i_len_pc) else {
        return false;
    };
    let i_width = 1 + i_len_byte as usize;
    let r_len_pc = next.pc + 1 + 1 + i_width;
    let Some(&r_len_byte) = code.get(r_len_pc) else {
        return false;
    };
    let r_len = r_len_byte as usize;
    if r_len < 2 {
        return false;
    }
    let Some(&attr_reg_byte) = code.get(r_len_pc + 1 + 1) else {
        return false;
    };
    if attr_reg_byte as usize != attr_dst_reg {
        return false;
    }
    let r_width = 1 + r_len;
    let descr_offset = 1 + i_width + r_width;
    let descr_index = decode_descr_index(code, &next, descr_offset);
    ctx.descr_refs
        .get(descr_index)
        .and_then(|descr| descr.as_call_descr())
        .is_some_and(|cd| {
            cd.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::LoadMethodSelf
        })
}

/// STORE_ATTR mirror of [`try_walker_specialize_load_attr`] for an existing
/// unboxed integer or float slot.  Recognition proves the plain mapdict write
/// cannot invoke Python or raise; the returned descriptor/arglist replaces only
/// the residual helper and effect.  The caller deliberately continues through
/// the generic residual recorder/executor so concrete execution, body-effect
/// tracking, and rollback semantics remain identical to the generic setattr
/// path (mapdict.py).
enum WalkerStoreAttrSpecialization {
    Residual(DescrRef, Vec<OpRef>),
    Direct,
}

/// The canonical Python class of an exact builtin operand, or `None` when it
/// is a subclass instance or carries no `w_class` to pin.
///
/// A specialization that bypasses special-method lookup needs this at RECORD
/// time to pick the class object its replay guard compares against.  PyPy has
/// no equivalent step: `space.allocate_instance` gives an `int` subclass its
/// own RPython class, so one `guard_class` already separates it from
/// `W_IntObject`.  pyre shares the `ob_type` layout between the two and carries
/// the Python-visible class in `w_class`, so `GuardClass` alone leaves a
/// subclass free to reuse the trace and skip its `__floordiv__` / `__pow__` /
/// `__lt__` override.
///
/// `is_exact_builtin_instance` also accepts a NULL `w_class`; that operand
/// cannot be pinned by [`walker_guard_exact_w_class`], so it declines here
/// rather than emitting a guard that would fail on its own recorded operand.
///
/// # Safety
/// `obj` must be a non-null, untagged heap object.
unsafe fn walker_exact_builtin_class(
    obj: pyre_object::PyObjectRef,
) -> Option<pyre_object::PyObjectRef> {
    unsafe {
        let w_class = (*obj).w_class;
        if w_class.is_null() {
            return None;
        }
        let canonical = pyre_object::pyobject::get_instantiate(&*(*obj).ob_type);
        std::ptr::eq(w_class, canonical).then_some(canonical)
    }
}

/// Walker-native mirror of the trait `trace_guard_exact_w_class`
/// (`trace_opcode.rs`): emit `getfield_gc_r(w_class)` → `ptr_eq(expected)`
/// → `guard_true` so the spec_ii fast path only stays live for an element
/// whose Python-level `w_class` is the canonical type object — a later
/// subclass sharing the payload `ob_type` side-exits rather than being
/// silently rewrapped as a plain int.  Skipped for an unescaped element (a
/// fresh `wrapint` box is provably plain, and reading its `w_class` would
/// force the box OptVirtualize removes).
fn walker_guard_exact_w_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    expected_typeobj: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    if expected_typeobj.is_null() || ctx.trace_ctx.heap_cache().is_unescaped(obj) {
        return Ok(());
    }
    let actual =
        crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, crate::descr::w_class_descr());
    let expected = ctx.trace_ctx.const_ref(expected_typeobj as i64);
    let eq = ctx.trace_ctx.record_op(OpCode::PtrEq, &[actual, expected]);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[eq])
}

/// #62 dead-`box_bool` proof for [`try_walker_specialize_compare_op_int`] /
/// `_float`.  Returns `true` only when a forward JitCode lookahead proves
/// the compare's boxed Ref dst register (`dst_reg`) is consumed *solely*
/// by the immediately-following `is_true` residual (the `POP_JUMP_IF_*`
/// truth read), so eliding the box is sound.
///
/// The proof requires, scanning forward from the compare until the first
/// `goto_if_not` / `goto`:
///   1. exactly one op reads `dst_reg` as a Ref operand, and it is an
///      `is_true`-shaped residual (`residual_call_r_i`, single Ref arg →
///      Int result);
///   2. no op overwrites `dst_reg` (`>r`) before that read;
///   3. the scan terminates at a `goto_if_not` (the branch the `is_true`
///      result feeds);
///   4. `dst_reg`'s color is NOT live at the `goto_if_not` resume target,
///      so the guard snapshot (`collect_outer_active_boxes`) cannot pick
///      up the Int marker that replaces the box.
///
/// Any deviation (escape to a local store, arithmetic use, second reader,
/// register reuse, kept-on-stack short-circuit, missing branch) returns
/// `false` → the caller emits the real box (current behaviour).  FBW-only
/// (returns `false` when `fbw_mode.snapshot_sym` is null).
fn compare_box_provably_dead<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    compare_pc: usize,
    dst_reg: u8,
) -> bool {
    let full_body_sym = ctx.fbw_mode.snapshot_sym;
    if full_body_sym.is_null() {
        return false;
    }
    // SAFETY: same contract as walker_capture_snapshot_for_last_guard_impl —
    // pointer live for the full-body walk, immutable layout fields only.
    let (code, jitcode_index, payload): (&[u8], i32, &crate::PyJitCode) = unsafe {
        let sym = &*full_body_sym;
        if sym.jitcode().is_null() {
            return false;
        }
        let jc = &*sym.jitcode();
        if jc.payload.code_ptr.is_null() {
            return false;
        }
        (
            jc.payload.jitcode.code.as_slice(),
            jc.index as i32,
            &jc.payload,
        )
    };
    let Some(start) = crate::jitcode_runtime::decode_op_at(code, compare_pc) else {
        return false;
    };
    let mut pc = start.next_pc;
    let mut readers = 0u32;
    let mut reader_is_is_true = false;
    let mut goto_if_not_pc: Option<usize> = None;
    for _ in 0..64 {
        let Some(op) = crate::jitcode_runtime::decode_op_at(code, pc) else {
            return false;
        };
        // Decode this op's operands, tracking reads/writes of dst_reg.
        let mut cursor = op.pc + 1;
        let mut chars = op.argcodes.chars();
        let mut reads = false;
        let mut writes = false;
        while let Some(c) = chars.next() {
            match c {
                'i' | 'c' | 'f' => cursor += 1,
                'r' => {
                    if *code.get(cursor).unwrap_or(&0) == dst_reg {
                        reads = true;
                    }
                    cursor += 1;
                }
                'L' | 'd' | 'j' => cursor += 2,
                'I' | 'F' => {
                    let n = *code.get(cursor).unwrap_or(&0) as usize;
                    cursor += 1 + n;
                }
                'R' => {
                    let n = *code.get(cursor).unwrap_or(&0) as usize;
                    for k in 0..n {
                        if *code.get(cursor + 1 + k).unwrap_or(&0) == dst_reg {
                            reads = true;
                        }
                    }
                    cursor += 1 + n;
                }
                '>' => {
                    let rt = chars.next();
                    if rt == Some('r') && *code.get(cursor).unwrap_or(&0) == dst_reg {
                        writes = true;
                    }
                    cursor += 1;
                }
                _ => return false,
            }
        }
        if writes {
            // dst overwritten before/at a read — give up (the value we'd
            // elide is not the one this op produces; stay conservative).
            return false;
        }
        if reads {
            readers += 1;
            // is_true shape: single Ref arg, Int result (`iRd>i`).
            reader_is_is_true = op.key == "residual_call_r_i/iRd>i";
        }
        if op.opname == "goto_if_not" {
            goto_if_not_pc = Some(op.pc);
            break;
        }
        if op.opname == "goto" || op.opname == "raise" || op.opname == "ref_return" {
            return false;
        }
        pc = op.next_pc;
    }
    // Conditions 1–3.
    if readers != 1 || !reader_is_is_true {
        return false;
    }
    let Some(gin_pc) = goto_if_not_pc else {
        return false;
    };
    let Some(gin_op) = crate::jitcode_runtime::decode_op_at(code, gin_pc) else {
        return false;
    };
    // Condition 4: `dst_reg`'s color must be dead at BOTH branch arms (the
    // POP_JUMP pops the tested bool regardless of direction, so the
    // guard's resume — whichever arm is not-taken — must not carry it).
    // Checking both arms also defends against register-color reuse: even
    // though the transient bool shares `dst_reg` with a later local, that
    // local is not yet live at the post-pop arm.  A `dst_reg` that IS live
    // at an arm means the snapshot would capture the Int marker → bail.
    // (`goto_if_not/iL`: operand 0 = `i` truth reg byte, operand 1 = `L`
    // 2-byte target label.)
    let fallthrough_jc = gin_op.next_pc;
    let target_jc = read_label(code, &gin_op, 1);
    // The `jc_pc` here is a branch ARM's op-start (`gin_op.next_pc` /
    // `read_label`), NOT the guard's resume marker, so the RAW offset does
    // not satisfy the carried-coordinate contract (`can_decode_live_vars`
    // may hold at an interior arm offset whose liveness window differs from
    // the py opcode's representative resume window — verified: querying the
    // raw offset directly drops live colors, e.g. `ref_ [0,1,4]` vs `[0,1]`).
    // The normalization this reader wants — the containing py opcode's
    // resume `-live-` — IS the resume-marker twin at `jc_pc`
    // (`resume_marker_for_jitcode_pc`, the invert→trivia-skip→resolve
    // composition built at codewrite time), so liveness is keyed on the twin
    // alone.
    let arm_dst_live = |jc_pc: usize| -> bool {
        // No twin ⇒ cannot prove the color dead ⇒ treat as live. Treating a
        // missing twin's liveness as empty would wrongly prove the register
        // dead and drop a live box.
        let Some(marker) = payload.resume_marker_for_jitcode_pc(jc_pc) else {
            return true;
        };
        let banks =
            crate::state::frame_liveness_reg_indices_by_bank_from_pc(jitcode_index, marker as i32);
        banks.ref_.iter().any(|&c| c as u8 == dst_reg)
    };
    if arm_dst_live(fallthrough_jc) || arm_dst_live(target_jc) {
        return false;
    }
    true
}

/// Global descr-pool sub-jitcode lookup (resolves a global jitcode index
/// through `ALL_JITCODES`, mirroring the shadow walker's lookup).  A
/// build-time canonical sub-body (`w_list_append`) carries no per-fn descr
/// pool (`JitCodeBody` has no `descrs` field), so it resolves its `d`/`j`
/// descr operands through the global pool (`all_descr_refs()` /
/// `RawDescrPool::Global`), and its inline-call descrs through this lookup.
static GLOBAL_SUB_JITCODE_LOOKUP_FN: fn(usize) -> Option<SubJitCodeBody> =
    sub_jitcode_body_by_index;

/// Return the driver green key for this top-level FOR_ITER residual.
///
/// `op_pc` belongs to the current per-CodeObject JitCode, while the driver
/// key is `(W_Code, python FOR_ITER pc)`.  The full-body snapshot root owns
/// the outer JitCode metadata needed for that inversion.  Inlined sub-walks
/// have a distinct callee JitCode but deliberately share the root snapshot;
/// decline their range fold rather than associating a callee guard with the
/// caller's key.
fn walker_foriter_green_key<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Option<u64> {
    if ctx.fbw_mode.inline_subwalk || ctx.fbw_mode.snapshot_sym.is_null() {
        return None;
    }
    // SAFETY: snapshot-root mode keeps the outer `PyreSym` live throughout
    // the full-body walk.  This reads immutable code metadata only.
    let sym = unsafe { &*ctx.fbw_mode.snapshot_sym };
    if sym.jitcode().is_null() {
        return None;
    }
    let jitcode = unsafe { &*sym.jitcode() };
    let raw_code = unsafe { jitcode.raw_code() };
    let w_code = pyre_interpreter::live_code_wrapper(raw_code as *const ()) as *const ();
    if w_code.is_null() {
        return None;
    }
    let foriter_start_pc = python_pc_for_jitcode_pc(&jitcode.payload.metadata, op_pc) as usize;
    Some(crate::driver::make_green_key(w_code, foriter_start_pc))
}

fn mark_trace_reads_module_global(
    tc: &mut TraceCtx,
    w_globals: pyre_object::PyObjectRef,
    name: &str,
) {
    if !w_globals.is_null() && crate::state::module_dict_cell_slot_direct(w_globals, name).is_some()
    {
        tc.reads_module_global = true;
    }
}

fn mark_trace_reads_module_global_from_code(
    tc: &mut TraceCtx,
    w_globals: pyre_object::PyObjectRef,
    w_code_ptr: usize,
    name_idx: usize,
) -> bool {
    // A null globals dict cannot be probed for membership; report unresolved so
    // the caller takes the conservative fallback, matching
    // `mark_trace_reads_module_global_from_frame_name`.
    if w_globals.is_null() {
        return false;
    }
    let Some(name) = (unsafe {
        let raw = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject;
        if raw.is_null() {
            None
        } else {
            pyre_interpreter::pyframe::load_name_from_code(&*raw, name_idx).map(|n| n.to_string())
        }
    }) else {
        return false;
    };
    mark_trace_reads_module_global(tc, w_globals, &name);
    true
}

fn mark_trace_reads_module_global_from_frame_name(
    tc: &mut TraceCtx,
    frame_ptr: usize,
    w_name_ptr: usize,
) -> bool {
    if frame_ptr == 0 || w_name_ptr == 0 {
        return false;
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let w_globals = frame.get_w_globals();
    if w_globals.is_null() {
        return false;
    }
    let name = unsafe {
        pyre_object::unicodeobject::w_str_get_value(w_name_ptr as pyre_object::PyObjectRef)
    };
    mark_trace_reads_module_global(tc, w_globals, name);
    true
}

/// Shared module-dict cell fast path for the LOAD_GLOBAL / LOAD_NAME folds:
/// const-fold a module-global name read to a `QuasiimmutField` version guard
/// + elidable `jit_namespace_cell_lookup`, reading an `ObjectMutableCell`'s
/// `w_value` live.  Returns `false` (fall through to the residual) for a
/// missing / `IntMutableCell` / still-movable slot.
fn emit_module_dict_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    w_globals: pyre_object::PyObjectRef,
    name: &str,
) -> Result<bool, DispatchError> {
    // Cell fast path applies only to a module dict still in strategy mode
    // whose slot holds a raw value, an `ObjectMutableCell`, or an
    // `IntMutableCell`; null/absent keys fall through to the residual.
    if let Some(slot) = crate::state::module_dict_cell_slot_direct(w_globals, name) {
        if let Some(stored) = crate::state::module_dict_cell_value_direct(w_globals, slot) {
            if !stored.is_null() {
                // The fast path const-folds `stored` (the slot's raw value, or
                // the `ObjectMutableCell` / `IntMutableCell`) as the elidable
                // `jit_namespace_cell_lookup` result.  That bakes its address
                // into the trace / guard resume data.  The collector is
                // moving, so a `stored` still in the nursery at trace time
                // relocates nursery->oldgen afterwards and the baked address
                // dangles (a `memo` dict grown in the loop is the canonical
                // case).  Fall through to the residual live lookup, which
                // re-reads the slot each call and follows the relocation, when
                // `stored` can still move.  Mutable cells are `malloc_typed`
                // (never nursery), so `can_move` is false and a hot int/object
                // global folds; a raw movable value does not.
                if !majit_gc::can_move(majit_ir::GcRef(stored as usize)) {
                    return emit_namespace_cell_fold(
                        ctx, op_pc, dst, dst_bank, w_globals, slot, stored, true,
                    );
                }
            }
        }
        // The name is present in the module dict but unfoldable
        // (null / movable raw value / strategy-switched); keep the residual
        // rather than misreaching into the builtins fallback (the residual
        // reads the live globals slot, which is correct).
        return Ok(false);
    }
    // Name absent from this module dict.
    Ok(false)
}

/// Emit the `QUASIIMMUT_FIELD(ns, slot)` + elidable `jit_namespace_cell_lookup`
/// + (for a mutable cell) live field read that the LOAD_GLOBAL cell fold lowers
/// to, seeding the dst's concrete with the unwrapped value.  `stored` is the
/// raw value-or-cell at `slot` of `ns` (a module dict in strategy mode); the
/// caller has already proven it foldable (non-null, immovable).  An
/// `ObjectMutableCell` reads `cell.w_value` (`getfield_gc_r`); an
/// `IntMutableCell` reads `cell.intvalue` (`getfield_gc_i`) and re-boxes it
/// (the box is elided by the optimizer when the sole consumer unboxes).
fn emit_namespace_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    ns: pyre_object::PyObjectRef,
    slot: usize,
    stored: pyre_object::PyObjectRef,
    guard_frame_globals: bool,
) -> Result<bool, DispatchError> {
    let is_obj_cell = unsafe { pyre_object::celldict::is_object_mutable_cell(stored) };
    let is_int_cell = unsafe { pyre_object::celldict::is_int_mutable_cell(stored) };
    let result_obj = unsafe { pyre_object::celldict::unwrap_cell(stored) };

    if guard_frame_globals && !guard_current_frame_globals_identity(ctx, op_pc, ns)? {
        return Ok(false);
    }
    let ns_const = ctx.trace_ctx.const_ref(ns as i64);
    let slot_const = ctx.trace_ctx.const_int(slot as i64);
    crate::state::record_namespace_quasiimmut_field(
        ctx.trace_ctx,
        ns_const,
        slot_const,
        slot as u32,
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)?;
    // Bake the immovable cell as a `ConstPtr` (pypy `ConstPtr(cell)`).  The
    // `QuasiimmutField(ns, slot)` guard above invalidates the loop on a
    // rebind / strategy-version bump (`optimize_QUASIIMMUT_FIELD` watches the
    // `(dict, slot)` pair, not the cell), and the caller's `can_move` check
    // guarantees the address is stable — the optimizer already folds the
    // equivalent elidable `jit_namespace_cell_lookup` down to this same const
    // ptr.  A genuine constant (not the elidable call's `RefOp` result, which
    // is not `is_constant()`) is what lets the trace-time heapcache's
    // `_unique_const_heuristic` canonicalise the LOAD's `getfield_gc_i` and
    // the STORE fold's `setfield_gc_i` onto one cache slot; without it a hot
    // int global's cached field goes stale.
    let cell_opref = ctx.trace_ctx.const_ref(stored as i64);
    // An `ObjectMutableCell` needs `cell.w_value` read LIVE so a same-key
    // reassign (in-place `write_cell`, no version bump) is observed each
    // iteration; an `IntMutableCell` reads `cell.intvalue` LIVE for the same
    // reason (`write_cell` mutates `intvalue` in place for an int->int
    // reassign) then re-boxes the raw int; a raw stored value is its own
    // result.
    let default_concrete = majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize));
    let (result_opref, result_concrete) = if is_obj_cell {
        (
            crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                cell_opref,
                crate::descr::object_mutable_cell_value_descr(),
            ),
            default_concrete,
        )
    } else if is_int_cell {
        let raw_int = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            cell_opref,
            crate::descr::int_mutable_cell_value_descr(),
        );
        let intval = unsafe { pyre_object::w_int_get_value(result_obj) };
        (
            walker_box_int(ctx, op_pc, raw_int, intval)?,
            box_int_concrete(intval, result_obj as i64),
        )
    } else {
        (cell_opref, default_concrete)
    };
    // Seed the dst's concrete with the unwrapped LOAD result so chained
    // walker handlers see the resolved value instead of `Null`.
    ctx.trace_ctx
        .set_opref_concrete(result_opref, result_concrete);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result_opref)?;
    // `pyjitpl.py _opimpl_residual_call*` finishes its no-raise
    // tail with `metainterp.clear_exception()`.  The fold replaces a
    // SUCCESSFUL (non-raising) `load_global_fn` residual, so it must mirror
    // that clear: in a handler body the except-side `LOAD_GLOBAL` runs with a
    // standing `last_exc_value` (the just-raised exception being matched), and
    // the residual success arm (`ctx.last_exc_value = None` at the
    // `exec_result` Ok leg) is what drains it before the handler's trailing
    // `catch_exception/L`.  Without this clear the elided residual leaves
    // `last_exc_value` set and the walk aborts `CatchExceptionWithActiveException`.
    ctx.last_exc_value = None;
    ctx.last_exc_value_concrete = ConcreteValue::Null;
    Ok(true)
}

/// STORE dual of [`emit_namespace_cell_fold`]: `QUASIIMMUT_FIELD(ns, slot)` +
/// elidable `jit_namespace_cell_lookup` (const-fold the cell ptr) +
/// `setfield_gc_i(cell, raw_int)` writing `IntMutableCell.intvalue` in place.
/// Mirrors pypy's inlined `write_cell` int arm (`typeobject.py`, the
/// `isinstance(w_cell, IntMutableCell) and is_plain_int1(w_value)` branch):
/// `setfield_gc(ConstPtr(cell), i_new, IntMutableCell.inst_intvalue)`.  No
/// runtime guard on `raw_int` — the caller recovered it from a
/// provably-plain-int JIT box (heapcache), so `is_plain_int1` folds away as it
/// does in the optimized pypy trace.  The version watcher (the
/// `QUASIIMMUT_FIELD` guard) still protects cell IDENTITY: reassigning the
/// global to a non-int replaces the cell + bumps the strategy version,
/// invalidating this loop.
fn emit_namespace_cell_store_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    ns: pyre_object::PyObjectRef,
    slot: usize,
    stored: pyre_object::PyObjectRef,
    raw_int: OpRef,
    new_int: i64,
) -> Result<(), DispatchError> {
    let ns_const = ctx.trace_ctx.const_ref(ns as i64);
    let slot_const = ctx.trace_ctx.const_int(slot as i64);
    crate::state::record_namespace_quasiimmut_field(
        ctx.trace_ctx,
        ns_const,
        slot_const,
        slot as u32,
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)?;
    // Bake the immovable cell as a `ConstPtr`, identical to the LOAD fold
    // (`emit_namespace_cell_fold`), so this `setfield_gc_i` and the LOAD's
    // `getfield_gc_i` canonicalise onto one trace-heapcache slot via
    // `_unique_const_heuristic` (both `ConstPtr(cell)`, matched by
    // `same_constant`).  An elidable-call `RefOp` cell would not be
    // `is_constant()`, leaving the store's cache write unreachable from the
    // load — the hot int global's cached field would go stale.
    let cell_opref = ctx.trace_ctx.const_ref(stored as i64);
    // `setfield_gc(cell, raw_int, IntMutableCell.intvalue)` with the same
    // heapcache-redundancy skip + write-through as `setfield_gc_via_heapcache`
    // (`pyjitpl.py _opimpl_setfield_gc_any`).
    let descr = crate::descr::int_mutable_cell_value_descr();
    let descr_index = descr.index();
    let is_redundant = ctx
        .trace_ctx
        .heapcache_getfield_cached(cell_opref, descr_index)
        == Some(raw_int);
    if is_redundant {
        ctx.trace_ctx.profiler().count_ops(
            majit_ir::OpCode::SetfieldGc,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
    } else {
        // pyjitpl.py:974-988 `_opimpl_setfield_gc_any` record leg.
        ctx.trace_ctx.profiler().count_ops(
            majit_ir::OpCode::SetfieldGc,
            majit_metainterp::counters::OPS,
        );
        ctx.trace_ctx.profiler().count_ops(
            majit_ir::OpCode::SetfieldGc,
            majit_metainterp::counters::RECORDED_OPS,
        );
        ctx.trace_ctx.record_op_with_descr(
            majit_ir::OpCode::SetfieldGc,
            &[cell_opref, raw_int],
            descr,
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(cell_opref, descr_index, raw_int);
        // Authoritative-executor eager store: the elided residual would have
        // run `write_cell` concretely (`try_execute_residual_call_via_executor`),
        // so apply the in-place `cell.intvalue` write now and journal the
        // displaced value for the non-commit rollback
        // ([`FBW_CELL_STORE_JOURNAL`]).  Without it the live cell keeps its
        // pre-store value while the trace heapcache carries the new box —
        // the next LOAD fold's cache-hit sanity check (pyjitpl.py)
        // trips on the divergence, and the walk's remaining concrete
        // execution reads the stale global.  The redundant arm above skips
        // the write: `cached == raw_int` means the cell already holds this
        // box's value (the cache is seeded from — and kept in step with —
        // the live cell).
        let cell = stored as *mut pyre_object::celldict::IntMutableCell;
        fbw_cell_store_journal_push(stored, unsafe { (*cell).intvalue });
        unsafe { (*cell).intvalue = new_int };
    }
    // `store_name_fn` is `CallFlavor::Plain` (can-raise); the fold replaces a
    // SUCCESSFUL non-raising store, so mirror the residual success arm's
    // exception clear exactly as [`emit_namespace_cell_fold`] does.
    ctx.last_exc_value = None;
    ctx.last_exc_value_concrete = ConcreteValue::Null;
    Ok(())
}

/// #67 shape fix: append virtualizable data boxes so the walker merge-point
/// `live_arg_boxes` matches the JUMP `close_loop_args_at` records.
fn append_virtualizable_boxes(ctx: &TraceCtx, mut reds: Vec<OpRef>) -> Vec<OpRef> {
    if let Some(total) = ctx.virtualizable_boxes_len() {
        for i in 0..total.saturating_sub(1) {
            if let Some(b) = ctx.virtualizable_box_at(i) {
                reds.push(b);
            }
        }
    }
    reds
}

fn write_branch_result<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    result_write: Option<(usize, OpRef)>,
) -> Result<(), DispatchError> {
    if let Some((dst, resbox)) = result_write {
        let concrete = match ctx.trace_ctx.concrete_of_opref(resbox) {
            Some(Value::Int(value)) => ConcreteValue::Int(value),
            _ => ConcreteValue::Null,
        };
        write_int_reg(ctx, op_pc, dst, resbox, concrete)?;
    }
    Ok(())
}

fn branch_without_guard<Sym: WalkSym>(
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    taken_pc: usize,
    result_write: Option<(usize, OpRef)>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    write_branch_result(ctx, op.pc, result_write)?;
    Ok((DispatchOutcome::Continue, taken_pc))
}

fn guarded_branch_core<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    guard_opcode: OpCode,
    guard_operands: &[OpRef],
    taken_pc: usize,
    other_pc: usize,
    result_write: Option<(usize, OpRef)>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Resume-data capture (`capture_resumedata(resumepc=orgpc)` at
    // `pyjitpl.py`) is threaded via
    // `walker_capture_snapshot_for_last_guard(other_target)`.
    // Branch guards resume at the runtime jump destination — the
    // branch NOT taken in the trace — not the `goto_if_not` opcode
    // itself (`trace_opcode.rs`, `resume_pc =
    // other_target`).  The trace took the `switchcase` direction;
    // a guard failure flips to the other arm, where the Python
    // interpreter has already popped the comparison truth, so the
    // blackhole must re-enter past `POP_JUMP_IF_*`.  GuardTrue
    // (trace fell through to `op.next_pc`) → resume at `target`;
    // GuardFalse (trace jumped to `target`) → resume at `op.next_pc`.
    let other_target = resolve_branch_target_through_trampoline(code, other_pc);
    {
        // #124/#281: a branch guard whose resume target still holds a
        // live operand-stack temp (short-circuit `and`/`or`, the
        // conditional expression, chained comparison) keeps that temp
        // on the not-taken arm the single-frame snapshot does not model
        // by itself.  The kept temp(s) are recovered from the not-taken
        // edge's `ref_copy` parallel-move trampoline (decoded below),
        // exact for any kept-stack depth.  Plain `while` / `if`
        // branches resume at depth 0 and carry no kept temp.
        //
        // #171 sub-walk single-frame collapse: inside an inlined-callee
        // sub-walk that resumes at the caller's CALL boundary,
        // `other_target` is a *callee* coordinate absent from the outer
        // jitcode's py_pc→jitcode tables, so `branch_resume_target_stack_depth`
        // (which maps through `fbw_mode.snapshot_sym`) would read a
        // meaningless outer depth at the coincidental offset.  The
        // collapse guard does not resume at a callee coordinate at all:
        // like every other single-frame sub-walk guard it collapses to
        // the caller's CALL boundary (`entry_py_pc` / `outer_active_boxes`,
        // `walker_capture_snapshot_for_last_guard_impl`), re-executing
        // the whole call on deopt — so there is no callee kept-stack
        // slot to recover.  Treat it as depth 0.
        //
        // Scope this to the collapse case ONLY: the #68 multiframe
        // inline path (`n_parents == n_callees`, both > 0) resumes the callee at its
        // OWN pc through `GuardCaptureScope::branch_guard_jitcode_pc`
        // (`walker_capture_multi_frame_inline_snapshot`), so its
        // kept-stack branches still need the real depth/recovery.
        //
        // The non-collapse read resolves `other_target` through an
        // `ActiveResumeFrame`.  `current()` selects the innermost inlined
        // callee when a sub-walk is active (else the portal frame), so a
        // callee branch guard's `other_target` is inverted through the
        // callee's own tables rather than the outer frame's — the frame
        // whose box collection (`collect_callee_active_boxes`) already
        // keys off the same framestack top. The #68
        // multiframe path reaches this `else`; the single-frame collapse
        // case short-circuits to `None` above.
        let single_frame_collapse = ctx.fbw_mode.inline_subwalk && {
            let session = ctx.session.borrow();
            let n_parents = session
                .framestack
                .iter()
                .filter(|frame| frame.parent.is_some())
                .count();
            let n_callees = session.framestack.len();
            !(n_parents > 0 && n_parents == n_callees)
        };
        let gate_frame = ActiveResumeFrame::current(ctx.session, ctx.fbw_mode.snapshot_sym);
        let resume_depth = if single_frame_collapse {
            None
        } else {
            gate_frame
                .as_ref()
                .and_then(|f| branch_resume_target_stack_depth(f, other_target))
        };
        let kept_stack = resume_depth.is_some_and(|d| d > 0);
        let depth_gt_1 = resume_depth.is_some_and(|d| d > 1);
        // Mirror-sourced kept-stack compile: a kept-stack guard
        // COMPILES whenever the walk-level operand-stack mirror
        // (`ctx.vstack_boxes`) covers EVERY kept resume slot
        // `0..resume_depth` with a non-NONE, non-NULL box — i.e. the
        // snapshot is 100% mirror-sourced with no legacy fallback.  This
        // bypasses the kept-stack declines below (whose purpose is
        // precisely the unreliable legacy resume the mirror replaces).
        // When the mirror does NOT cover a kept slot,
        // `mirror_covers_kept` is `false` and the hazard checks below
        // still run — but with the flat maps deleted they decline only
        // for an unrestorable-Ref arm (Hazard 1) or an INVALID mirror
        // (undermodeled walk); a VALID mirror with a NONE hole (an
        // edge-materialized merge temp) compiles, its slot sourced from
        // the decoded trampoline recovery (`resolved_recovered`).
        let mirror_covers_kept = ctx.vstack_valid
            && resume_depth.is_some_and(|d| {
                (0..d).all(|s| {
                    ctx.vstack_boxes
                        .get(s as usize)
                        .copied()
                        .is_some_and(|b| b != OpRef::NONE && !opref_is_null_const_ptr(b))
                })
            });
        // `branch_resume_target_stack_depth` reads
        // `fbw_mode.snapshot_sym`. Probe the jitcode-store depth for
        // the unrestorable-arm decline below as well.
        // `kept_stack_any_leg` and the kept-stack hazard checks below
        // all read `fbw_mode.snapshot_sym`, which models the top-level
        // traced jitcode's register file. In an inlined-callee sub-walk
        // (`is_top_level == false`) the current `concrete_registers_r`
        // is the callee's, so an outer stack-slot color indexes a
        // foreign callee register — `kept_boxed_int` below would then
        // dereference an unrelated `Ref`, a dangling pointer
        // (KERN_INVALID_ADDRESS / SIGSEGV). A callee branch collapses to
        // the caller's CALL boundary on deopt (the #171 single-frame
        // collapse), so it has no top-level kept-stack slot to recover;
        // gate on `is_top_level` and treat it as no kept stack, matching
        // the `resume_depth` collapse handling above.
        //
        // The depth lookup also keys off `ctx.outer_jitcode_index`,
        // which is the FBW sym's `(*sym.jitcode).index` and uniformly 0
        // (the canonical core's `index` is never stamped with the
        // runtime `MetaInterpStaticData.jitcodes` position), so for the
        // second and later distinct functions compiled in a program it
        // resolves to `jitcodes[0]` — the FIRST function — and reads its
        // metadata at this function's jitcode pc, yielding a wrong depth.
        // The FBW-leg `kept_stack` reads the depth through `gate_frame`
        // (`ActiveResumeFrame`, resolved per-function via
        // `ensure_jitcode_index`), so it is correct for every function;
        // `kept_stack_any_leg` is retained only as the fallback for the
        // `gate_frame == None` case.
        let kept_stack_any_leg = ctx.is_top_level
            && branch_resume_target_stack_depth_any_leg(other_target, ctx.outer_jitcode_index)
                .is_some_and(|d| d > 0);
        // A kept-stack guard's not-taken arm keeps one or more
        // operand-stack temps live across the guard.  The not-taken
        // edge resolves the merge through inline `ref_copy(dst <- src)`
        // moves (`flatten.rs insert_renamings`): the live kept
        // value sits at the guard-pc color `src`, which the walk has
        // written, while the resume merge color `dst` is unwritten at
        // the guard point.  Decode that trampoline into `(dst, src)`
        // pairs (`#420`); the snapshot / vable recovery then reads
        // `registers_r[src]` for each kept slot — exact for any kept-
        // stack depth, superseding the positional depth-1 heuristic.
        let raw_branch_target = other_pc;
        let kept_recovered = if kept_stack {
            decode_branch_trampoline_ref_moves(code, raw_branch_target)
        } else {
            Some(Vec::new())
        };
        if std::env::var_os("PYRE_DIAG124C").is_some() && kept_stack {
            eprintln!(
                "[edge124] pc={} depth={resume_depth:?} raw_target={raw_branch_target} \
                 moves(dst<-src)={kept_recovered:?}",
                op.pc,
            );
        }
        // Resolve each `(dst, src)` move to `(dst, live guard value)`
        // against the guard-state register file NOW, before recording
        // the guard.  A move whose `src` is out of range or holds
        // `OpRef::NONE` recovers no live kept value and is dropped so the
        // snapshot encoder never records a dead kept slot.  A const-source
        // `ref_copy` patches `src` into the constants window of
        // `registers_r`, so this one read covers register and const
        // sources alike.  This resolved set is the single source of truth
        // the snapshot encoder reads
        // (`GuardCaptureScope::branch_guard_kept_recovered` below);
        // `record_guard` records into the trace history only and
        // does not mutate `registers_r`, so reading it here is identical
        // to reading it post-guard.
        let resolved_recovered: Option<Vec<(u16, OpRef)>> = kept_recovered.as_ref().map(|mv| {
            mv.iter()
                .filter_map(|&(dst, src)| {
                    let v = ctx.registers_r.get(src as usize).copied()?;
                    (v != OpRef::NONE).then_some((dst, v))
                })
                .collect()
        });
        // PARITY DEVIATION (converges at the symbolic-valuestack
        // capture, #73/#423): `opimpl_goto_if_not` (pyjitpl.py)
        // always records GUARD_TRUE/FALSE for a non-constant condition
        // and captures resumedata (pyjitpl.py), never declining —
        // its resume reconstruction is complete.  pyre's kept-stack
        // resume reconstruction is NOT yet complete (the walker
        // snapshot is partial), so recording the guard and
        // resuming would rebuild a kept slot as NULL / a wrong value:
        // the #416/#420 boxed-int short-circuit / conditional-expression
        // SIGSEGV + silent miscompile.  Until that capture lands pyre
        // deviates by declining here.  A kept-stack guard's not-taken
        // arm is only safe to compile when the blackhole can reconstruct
        // every value the arm reads on resume.  Three resume hazards make
        // a kept-stack arm unsafe; each is described at its check below.
        // Decline → interpreter (correct).  Applies to depth-1 and
        // depth > 1.
        //
        // Gate on `kept_stack` (per-function-correct via `gate_frame`)
        // OR the jitcode-store `kept_stack_any_leg` fallback.
        // `kept_stack_any_leg` alone is unsound for the second and later
        // distinct functions in a program: its `outer_jitcode_index`
        // resolves to `jitcodes[0]` (the first function) and reports a
        // wrong depth, so a kept-stack arm in a later function would skip
        // the decline and silently miscompile.  `kept_stack` reads the
        // correct per-function depth and closes that gap.
        if (kept_stack || kept_stack_any_leg) && !mirror_covers_kept {
            let liveness = branch_arm_resume_ref_liveness(ctx.fbw_mode, other_target);
            // Hazard (1): the not-taken arm reads a regular Ref register
            // the blackhole resumes as NULL (the conditional-expression
            // boxed-int NULL-deref crash).
            //
            // Scoped to the undermodeled invalid-mirror walk
            // (`!ctx.vstack_valid`), exactly like Hazards (2)/(3) below.
            // A VALID walk mirror sources every on-stack kept slot from
            // `ctx.vstack_boxes` and every kept local from the vable
            // shadow (`collect_outer_active_boxes`), so on resume the
            // not-taken arm's Ref reads are reconstructed per-slot even
            // when the guard's snapshot liveness (`live_ref`) does not
            // name that register color — the same per-slot recovery that
            // makes the short-circuit / conditional-expression kept-stack
            // reads (#416/#420) restorable for Hazards (2)/(3).  The
            // register-file scan only proves a hazard for the INVALID
            // mirror, where those per-slot sources are unavailable.
            let reads_null_ref = !ctx.vstack_valid
                && match &liveness {
                    Some((live_ref, num_regs_r)) => {
                        branch_arm_reads_unrestorable_ref(code, other_target, live_ref, *num_regs_r)
                    }
                    // Liveness unavailable (`fbw_mode.snapshot_sym` is
                    // null, or the coordinate is unresolved) — cannot
                    // prove restorable, so decline.
                    None => true,
                };
            // Hazard (2): the not-taken edge carries `ref_copy` renames
            // (`kept_recovered` non-empty) — the #416/#420 short-circuit
            // / chained-comparison kept-stack recovery.  Historically the
            // recovery read the flat merge-color register file, which
            // could return a stale reused box for a hoisted heap-int
            // constant (the `((i & 1) and 1000000)` silent miscompile).
            // With the flat maps deleted the capture is per-slot: a
            // VALID walk mirror sources every on-stack kept slot, and an
            // edge-materialized merge slot resolves through the decoded
            // `(dst, src)` trampoline against the guard-pc register file
            // (`resolved_recovered`) — verified byte-exact against the
            // declined-interpreter oracle across the 599-program
            // adversarial corpus (kept-stack census, incl. the
            // heap-int short-circuit / conditional-expression repros).
            // The hazard remains only for an UNDERMODELED walk (invalid
            // mirror: inline sub-walk / Unmodeled opcode), where those
            // per-slot sources are unavailable and forcing the recovery
            // through miscompiles (nested and/or under an inline
            // sub-walk).
            let uses_edge_recovery = !ctx.vstack_valid
                && kept_recovered
                    .as_deref()
                    .is_some_and(|moves| !moves.is_empty());
            // Hazard (3): a kept operand-stack slot itself holds a heap
            // int outside the 1-byte immediate range `[0, 256)` (the
            // accumulator in `acc += (x if c else y)`).  Historically the
            // guard's resume snapshot rebuilt such a slot through the
            // flat merge-color maps and could deliver a WRONG / NULL box
            // (the conditional-expression boxed-int crash).  As with
            // Hazard (2), the per-slot capture that replaced the flat
            // maps restores boxed-int kept slots faithfully whenever the
            // walk mirror is VALID (corpus-verified against the declined
            // oracle), so the hazard is scoped to the undermodeled
            // invalid-mirror walk.
            let kept_boxed_int = !ctx.vstack_valid
                && gate_frame.as_ref().is_some_and(|f| {
                    kept_stack_has_boxed_int_hazard(f, other_target, ctx.concrete_registers_r)
                });
            // A not-taken arm resuming at an exception-handler-protected
            // PC carries the kept exception operand (`PUSH_EXC_INFO`'s
            // Ref) on its operand stack; the handler-entry mirror reseed
            // reconstructs it directly (so `mirror_covers_kept` gates this
            // whole block out), and where the mirror still does not cover,
            // that kept Ref always also trips Hazard (1)/(2)/(3) — so the
            // exc-region case needs no decline of its own.
            if reads_null_ref || uses_edge_recovery || kept_boxed_int {
                // Attribute the kept-stack decline to the hazard that
                // fired and the mirror state behind it, so a corpus run
                // (`PYRE_FBW_DEBUG_ABORT`) can separate the distinct
                // decline causes without re-instrumenting: an inline
                // sub-walk (`subwalk`/`!vstack_valid`), a genuinely
                // unrestorable regular Ref (`reads_null_ref` with a
                // valid mirror), or an undermodeled-mirror boxed-int /
                // edge-recovery slot.
                if fbw_debug_abort_enabled() {
                    eprintln!(
                        "[decline-why] PERMANENT pc={} other_target={} vstack_valid={} \
                         subwalk={} mirror_covers_kept={} depth_gt_1={} kept_stack={} \
                         kept_stack_any_leg={} reads_null_ref={} uses_edge_recovery={} \
                         kept_boxed_int={} kept_recovered_nonempty={}",
                        op.pc,
                        other_target,
                        ctx.vstack_valid,
                        ctx.fbw_mode.inline_subwalk,
                        mirror_covers_kept,
                        depth_gt_1,
                        kept_stack,
                        kept_stack_any_leg,
                        reads_null_ref,
                        uses_edge_recovery,
                        kept_boxed_int,
                        kept_recovered.as_deref().is_some_and(|m| !m.is_empty()),
                    );
                }
                // Stamp the abort coordinate at the raise point so the
                // driver gate cannot observe an unrelated prior abort.
                ctx.session.borrow_mut().abort_in_subwalk = ctx.fbw_mode.inline_subwalk;
                return Err(DispatchError::BranchGuardUnrestorableKeptStackPermanent { pc: op.pc });
            }
        }
        // A depth > 1 kept operand stack is recoverable on resume from
        // the per-slot sources of a VALID walk mirror: every on-stack
        // kept slot from `ctx.vstack_boxes`, and an edge-materialized
        // merge slot (a NONE hole in an otherwise valid mirror) from the
        // decoded trampoline recovery (`resolved_recovered`).  Only an
        // INVALID mirror (an undermodeled walk: inline sub-walk /
        // Unmodeled opcode) leaves the kept slots without a reliable
        // per-slot source, so decline → interpreter (correct).
        if depth_gt_1 && !ctx.vstack_valid {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[decline-why] UNSUPPORTED pc={} other_target={} vstack_valid={} \
                     subwalk={} depth_gt_1={} kept_stack={} kept_stack_any_leg={}",
                    op.pc,
                    other_target,
                    ctx.vstack_valid,
                    ctx.fbw_mode.inline_subwalk,
                    depth_gt_1,
                    kept_stack,
                    kept_stack_any_leg,
                );
            }
            // Stamp the abort coordinate at the raise point so the
            // walk-end branch-flush gate cannot flush the outer frame
            // from a callee-coordinate abort.
            ctx.session.borrow_mut().abort_in_subwalk = ctx.fbw_mode.inline_subwalk;
            return Err(DispatchError::BranchGuardKeptStackUnsupported { pc: op.pc });
        }
        ctx.trace_ctx.record_guard(guard_opcode, guard_operands, 0);
        // Publish the guard's own jitcode coordinate ONLY for the
        // kept-stack case so the snapshot encoder recovers the kept
        // operand-stack values from the guard-pc register file (the
        // resume coordinate `other_target` names a merge point whose
        // live colors the walk has not written at the guard point).
        // A depth-0 branch resumes losslessly at `other_target` via
        // the baseline `py_pc → jitcode` resume-translation path;
        // routing it through the
        // guard-pc carrier would resume one opcode early (re-running
        // `goto_if_not`) and desync the decoded box layout.
        // Feed the snapshot encoder the SAME resolved set the gate
        // checked (resolved above, before `record_guard`): each
        // `(dst, live guard value)`, sources already filtered to live
        // in-range values. Only a kept-stack guard publishes these
        // inputs.
        let kept = if kept_stack {
            resolved_recovered.unwrap_or_default()
        } else {
            Vec::new()
        };
        walker_capture_snapshot_for_last_guard_scoped(
            ctx,
            other_target,
            GuardCaptureScope {
                branch_guard_jitcode_pc: kept_stack.then_some(op.pc),
                branch_guard_kept_recovered: &kept,
                ..GuardCaptureScope::default()
            },
        )?;
    }
    write_branch_result(ctx, op.pc, result_write)?;
    Ok((DispatchOutcome::Continue, taken_pc))
}

fn goto_if_not_branch_on<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    condbox: OpRef,
    switchcase: i64,
    target: usize,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // pyjitpl.py `opimpl_goto_if_not` requires a boolean switchcase.
    assert!(
        switchcase == 0 || switchcase == 1,
        "opimpl_goto_if_not: switchcase must be 0 or 1, got {} (pc={})",
        switchcase,
        op.pc
    );
    let (guard_opcode, taken_pc, other_pc) = if switchcase != 0 {
        (OpCode::GuardTrue, op.next_pc, target)
    } else {
        (OpCode::GuardFalse, target, op.next_pc)
    };

    // `generate_guard` in `pyjitpl.py opimpl_goto_if_not` skips Const boxes.
    // No register replacement occurs here; fused comparisons pass
    // `replace=False`, preserving loop-variant conditions.
    if condbox.is_constant() {
        branch_without_guard(op, ctx, taken_pc, None)
    } else {
        guarded_branch_core(
            code,
            op,
            ctx,
            guard_opcode,
            &[condbox],
            taken_pc,
            other_pc,
            None,
        )
    }
}

fn int_ovf_jump<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let target = read_label(code, op, 0);
    let b1 = read_int_reg(code, op, 2, ctx)?;
    let b2 = read_int_reg(code, op, 3, ctx)?;
    let dst = code[op.pc + 5] as usize;
    let (resbox, overflow) = record_int_ovf(ctx, op.pc, opcode, b1, b2)?;

    // `pyjitpl.py opimpl_int_add_jump_if_ovf` and
    // `pyjitpl.py handle_possible_overflow_error`: Const operands branch
    // without a guard; symbolic operands emit an operand-less overflow guard.
    let (guard_opcode, taken_pc, result_write) = if overflow {
        (OpCode::GuardOverflow, target, None)
    } else {
        (OpCode::GuardNoOverflow, op.next_pc, Some((dst, resbox)))
    };
    if resbox.is_constant() {
        branch_without_guard(op, ctx, taken_pc, result_write)
    } else {
        ctx.trace_ctx.record_guard(guard_opcode, &[], 0);
        // `handle_possible_overflow_error` resumes at `orgpc`: unlike
        // `goto_if_not`, blackhole must re-execute the arithmetic opcode so
        // the no-overflow path writes its result register.
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        write_branch_result(ctx, op.pc, result_write)?;
        Ok((DispatchOutcome::Continue, taken_pc))
    }
}

fn fused_goto_if_not_int<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let b = read_int_reg(code, op, 1, ctx)?;
    let condbox = record_int_cmp(ctx, opcode, a, b);
    let switchcase = match ctx.trace_ctx.concrete_of_opref(condbox) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::GotoIfNotValueNotConcrete {
                pc: op.pc,
                value: condbox,
            });
        }
    };
    let target = read_label(code, op, 2);
    goto_if_not_branch_on(code, op, ctx, condbox, switchcase, target)
}

/// RPython `pyjitpl.py _establish_nullity(box, orgpc)` — the shared body
/// behind `opimpl_goto_if_not_ptr_nonzero` / `opimpl_goto_if_not_ptr_iszero`:
///
///   value = box.nonnull()
///   if heapcache.is_nullity_known(box):
///       profiler.count_ops(rop.GUARD_NONNULL, HEAPCACHED_OPS)
///       return value
///   if value:
///       if not heapcache.is_class_known(box):
///           generate_guard(rop.GUARD_NONNULL, box, resumepc=orgpc)
///   else:
///       if not isinstance(box, Const):
///           generate_guard(rop.GUARD_ISNULL, box, resumepc=orgpc)
///           promoted_box = executor.constant_from_op(box)
///           replace_box(box, promoted_box)
///   heapcache.nullity_now_known(box)
///   return value
///
/// Unlike the compare-fused gotos this records no condition and emits no
/// GuardTrue/GuardFalse: the nullity guard *is* the guard, and the branch
/// direction comes from the pointer the walk observed. The guard emission
/// stays walker-side because resume snapshots are —
/// `walker_emit_guard_with_snapshot` is the walk's `generate_guard`.
///
/// The heapcache check runs before `box.nonnull()`: upstream reads the
/// concrete pointer unconditionally, but the walker knows it only through
/// the concrete shadow, which may be absent even when a prior guard already
/// proved the box's nullity. Consulting the cache first keeps the known
/// answer from being lost to a missing shadow (the value is read lazily only
/// on a cache miss).
///
/// Returns `box.nonnull()`.
fn establish_nullity<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    boxref: OpRef,
) -> Result<bool, DispatchError> {
    // Pyre's `is_nullity_known` splits upstream's boolean into which side is
    // known (`Some(true)` non-null, `Some(false)` null, `None` unknown). When
    // the nullity is already known, the stored side *is* `box.nonnull()` — an
    // SSA box's nullity is immutable — so the heapcache answers the branch
    // without reading the pointer. This is consulted before the concrete
    // shadow so a box whose nullity a prior guard proved still resolves even
    // when its Ref shadow is absent.
    let known = ctx.trace_ctx.heap_cache().is_nullity_known(boxref, |op| {
        op.inline_const_to_value().and_then(|v| match v {
            Value::Int(n) => Some(n),
            Value::Ref(gc) => Some(gc.0 as i64),
            _ => None,
        })
    });
    if let Some(value) = known {
        ctx.trace_ctx.profiler().count_ops(
            OpCode::GuardNonnull,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return Ok(value);
    }
    // Nullity not yet known: `box.nonnull()` reads the pointer itself. The
    // walker knows it only through the concrete shadow, and an absent shadow
    // is not evidence of either nullity, so decline rather than guess the
    // branch direction.
    let ConcreteValue::Ref(ptr) = read_ref_reg_concrete(code, op, 0, ctx) else {
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: op.key,
        });
    };
    let value = !ptr.is_null();
    if value {
        // A known class already implies non-null, so the guard is redundant.
        if !ctx.trace_ctx.heap_cache().is_class_known(boxref) {
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[boxref])?;
        }
    } else if !boxref.is_constant() {
        walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardIsnull, &[boxref])?;
        // `constant_from_op` of a proven-null ref is the null constant.
        let promoted = ctx.trace_ctx.const_ref(0);
        ctx.trace_ctx.replace_box(boxref, promoted);
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .nullity_now_known(boxref, value);
    Ok(value)
}

/// The unary member of the fused-goto family. `pyjitpl.py`
/// `opimpl_goto_if_not_int_is_zero` records the condition and hands it to
/// `opimpl_goto_if_not` with `replace=False`:
///
///   condbox = self.execute(rop.INT_IS_ZERO, box)
///   self.opimpl_goto_if_not(condbox, target, orgpc, replace=False)
///
/// Operand layout `iL`: 1B int reg + 2B label.
fn fused_goto_if_not_int_unary<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let condbox = record_int_unary(ctx, opcode, a);
    let switchcase = match ctx.trace_ctx.concrete_of_opref(condbox) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::GotoIfNotValueNotConcrete {
                pc: op.pc,
                value: condbox,
            });
        }
    };
    let target = read_label(code, op, 1);
    goto_if_not_branch_on(code, op, ctx, condbox, switchcase, target)
}

fn fused_goto_if_not_float<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_float_reg(code, op, 0, ctx)?;
    let b = read_float_reg(code, op, 1, ctx)?;
    let condbox = record_float_cmp(ctx, opcode, a, b);
    let switchcase = match ctx.trace_ctx.concrete_of_opref(condbox) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::GotoIfNotValueNotConcrete {
                pc: op.pc,
                value: condbox,
            });
        }
    };
    let target = read_label(code, op, 2);
    goto_if_not_branch_on(code, op, ctx, condbox, switchcase, target)
}

fn fused_goto_if_not_ptr<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_ref_reg(code, op, 0, ctx)?;
    let b = read_ref_reg(code, op, 1, ctx)?;
    let a_concrete = read_ref_reg_concrete(code, op, 0, ctx);
    let b_concrete = read_ref_reg_concrete(code, op, 1, ctx);
    let condbox = record_ptr_cmp(ctx, opcode, a, b, a_concrete, b_concrete);
    let switchcase = match ctx.trace_ctx.concrete_of_opref(condbox) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::GotoIfNotValueNotConcrete {
                pc: op.pc,
                value: condbox,
            });
        }
    };
    let target = read_label(code, op, 2);
    goto_if_not_branch_on(code, op, ctx, condbox, switchcase, target)
}

/// Per-opname dispatch table. Returning `(outcome, next_pc)` lets
/// branching handlers (`goto/L`) override the linear `op.next_pc`
/// advance; non-branching handlers return `op.next_pc` unchanged.
fn handle<Sym: WalkSym>(
    op: &DecodedOp,
    code: &[u8],
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // The `int_*` / `float_*` / `ptr_*` families whose arm is a uniform
    // `HELPER(code, op, ctx, OpCode::VARIANT)` call are dispatched from the
    // `regular_record_table!`-generated table (see `arith.rs`); a `Some`
    // there stands in for the arm that used to live in this `match`.
    if let Some(res) = dispatch_regular_record(op, code, ctx) {
        return res;
    }
    match op.key {
        "live/" => Ok((DispatchOutcome::Continue, op.next_pc)),
        "loop_header/i" => {
            // pyjitpl.py `opimpl_loop_header(jdindex, orgpc)`:
            // pure flag setter — stamps `seen_loop_header_for_jdindex` so
            // the following `jit_merge_point` treats the arrival as a
            // loop crossing (the lowered `can_enter_jit` at a backward
            // jump, jtransform.py). The close/register decision
            // happens at the merge point (pyjitpl.py). Mirrors
            // majit's `BC_LOOP_HEADER` arm (`pyjitpl/dispatch.rs`).
            let jd_opref = read_int_reg(code, op, 0, ctx)?;
            let jdindex = match ctx.trace_ctx.concrete_of_opref(jd_opref) {
                Some(Value::Int(v)) => v,
                _ => return Err(DispatchError::LoopHeaderJdIndexUnresolved { pc: op.pc }),
            };
            ctx.trace_ctx.seen_loop_header_for_jdindex = jdindex as i32;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // RPython parity: `pyjitpl.py _opimpl_inline_call*`
        // pushes a fresh `MIFrame(jitcode)` populated with caller args,
        // raises `ChangeFrame()` so the metainterp loop dispatches the
        // next op on the new frame, and on `*_return` pops back via
        // `metainterp.finishframe(value)` — writing `value` into the
        // caller's `>X` slot. Walker simulates the same shape with
        // synchronous recursion through `dispatch_inline_call_dr_kind`.
        //
        // The `_r_r` (Ref result) and `_r_i` (Int result) variants share
        // the same `dR` arglist shape; only the dst bank differs.
        "inline_call_r_r/dR>r" => dispatch_inline_call_dr_kind(code, op, ctx, 'r'),
        "inline_call_r_i/dR>i" => dispatch_inline_call_dr_kind(code, op, ctx, 'i'),
        // `_r_v/dR` — void-return variant per `bhimpl_inline_call_r_v`
        // (`blackhole.py`).  Same recursion + arglist as `_r_*`;
        // callee exits via `void_return/`, no SubReturn writeback.
        "inline_call_r_v/dR" => dispatch_inline_call_dr_kind(code, op, ctx, 'v'),
        // `_ir_*` variants extend the arglist to a (I-list, R-list) pair.
        // RPython's `setup_call(argboxes_i, argboxes_r, argboxes_f)` populates
        // both kind banks. The dst bank still selects the SubReturn write
        // target (Ref bank for `_ir_r/dIR>r`, Int bank for `_ir_i/dIR>i`,
        // void no-write for `_ir_v/dIR`).
        "inline_call_ir_r/dIR>r" => dispatch_inline_call_dir_kind(code, op, ctx, 'r'),
        "inline_call_ir_i/dIR>i" => dispatch_inline_call_dir_kind(code, op, ctx, 'i'),
        "inline_call_ir_v/dIR" => dispatch_inline_call_dir_kind(code, op, ctx, 'v'),
        // `_irf_*` variants extend the arglist with a float list (I-list,
        // R-list, F-list). Same `setup_call(argboxes_i, argboxes_r,
        // argboxes_f)` distribution; dst bank chooses Int / Ref / Float
        // for the SubReturn writeback or void no-write for `_irf_v/dIRF`.
        "inline_call_irf_i/dIRF>i" => dispatch_inline_call_dirf_kind(code, op, ctx, 'i'),
        "inline_call_irf_r/dIRF>r" => dispatch_inline_call_dirf_kind(code, op, ctx, 'r'),
        "inline_call_irf_f/dIRF>f" => dispatch_inline_call_dirf_kind(code, op, ctx, 'f'),
        "inline_call_irf_v/dIRF" => dispatch_inline_call_dirf_kind(code, op, ctx, 'v'),
        "goto/L" => {
            // RPython `blackhole.py bhimpl_goto(target): return
            // target`. The 2-byte LE label was resolved by
            // `assembler.fix_labels` to a direct pc; pyre + RPython
            // agree that goto records nothing (pure control flow).
            let target = read_label(code, op, 0);
            Ok((DispatchOutcome::Continue, target))
        }
        "goto_if_not/iL" => {
            // RPython `pyjitpl.py opimpl_goto_if_not`:
            //
            //   @arguments("box", "label", "orgpc")
            //   def opimpl_goto_if_not(self, box, target, orgpc, replace=True):
            //       switchcase = box.getint()
            //       if switchcase:
            //           assert switchcase == 1
            //           opnum = rop.GUARD_TRUE
            //           promoted_box = CONST_1
            //       else:
            //           opnum = rop.GUARD_FALSE
            //           promoted_box = CONST_0
            //       self.metainterp.generate_guard(opnum, box, resumepc=orgpc)
            //       if not switchcase:
            //           self.pc = target
            //       if isinstance(box, Const):
            //           return
            //       if replace:
            //           self.metainterp.replace_box(box, promoted_box)
            //
            // Operand layout `iL`: 1B Int register + 2B LE label.
            // Concrete branch value comes from `TraceCtx::concrete_of_opref`
            // (same path `switch/id` uses); non-concrete OpRefs surface
            // `GotoIfNotValueNotConcrete` rather than guess a direction.
            let valuebox = read_int_reg(code, op, 0, ctx)?;
            let target = read_label(code, op, 1);
            let switchcase = match ctx.trace_ctx.concrete_of_opref(valuebox) {
                Some(Value::Int(v)) => v,
                _ => {
                    if std::env::var("PYRE_DIAG_GIN").is_ok() {
                        let tb = read_int_reg_concrete(code, op, 0, ctx);
                        let reg = code[op.pc + 1] as usize;
                        eprintln!(
                            "[diag-gin] pc={} valuebox={:?} reg={} concrete_of_opref=NON-INT typed_bank={:?}",
                            op.pc, valuebox, reg, tb
                        );
                    }
                    return Err(DispatchError::GotoIfNotValueNotConcrete {
                        pc: op.pc,
                        value: valuebox,
                    });
                }
            };
            goto_if_not_branch_on(code, op, ctx, valuebox, switchcase, target)
        }
        // `pyjitpl.py`'s exec-generated `opimpl_goto_if_not_<cmp>` loop:
        // `condbox = self.execute(rop.<CMP>, b1, b2); self.opimpl_goto_if_not(
        // condbox, target, orgpc, replace=False)`. The fused arm records the
        // same compare and reuses `goto_if_not_branch_on`.
        // The pointer pair reads its branch direction straight out of
        // `_establish_nullity` — no condition op, no GuardTrue/GuardFalse:
        //
        //   def opimpl_goto_if_not_ptr_nonzero(self, box, target, orgpc):
        //       if not self._establish_nullity(box, orgpc):
        //           self.pc = target
        //
        //   def opimpl_goto_if_not_ptr_iszero(self, box, target, orgpc):
        //       if self._establish_nullity(box, orgpc):
        //           self.pc = target
        //
        // Operand layout `rL`: 1B ref reg + 2B label.
        "goto_if_not_ptr_nonzero/rL" | "goto_if_not_ptr_iszero/rL" => {
            let boxref = read_ref_reg(code, op, 0, ctx)?;
            let nonnull = establish_nullity(code, op, ctx, boxref)?;
            let jump = if op.key == "goto_if_not_ptr_nonzero/rL" {
                !nonnull
            } else {
                nonnull
            };
            let target = read_label(code, op, 1);
            Ok((
                DispatchOutcome::Continue,
                if jump { target } else { op.next_pc },
            ))
        }
        // Same shape with one operand; `goto_if_not_int_is_true` has no arm
        // because the assembler spells that condition as plain `goto_if_not`,
        // matching the class-attribute alias upstream gives it.
        "goto_if_not_int_is_zero/iL" => {
            fused_goto_if_not_int_unary(code, op, ctx, OpCode::IntIsZero)
        }
        "goto_if_not_int_lt/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntLt),
        "goto_if_not_int_le/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntLe),
        "goto_if_not_int_eq/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntEq),
        "goto_if_not_int_ne/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntNe),
        "goto_if_not_int_gt/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntGt),
        "goto_if_not_int_ge/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntGe),
        "goto_if_not_float_lt/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatLt),
        "goto_if_not_float_le/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatLe),
        "goto_if_not_float_eq/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatEq),
        "goto_if_not_float_ne/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatNe),
        "goto_if_not_float_gt/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatGt),
        "goto_if_not_float_ge/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatGe),
        "goto_if_not_ptr_eq/rrL" => fused_goto_if_not_ptr(code, op, ctx, OpCode::PtrEq),
        "goto_if_not_ptr_ne/rrL" => fused_goto_if_not_ptr(code, op, ctx, OpCode::PtrNe),
        "int_add_jump_if_ovf/Lii>i" => int_ovf_jump(code, op, ctx, OpCode::IntAddOvf),
        "int_sub_jump_if_ovf/Lii>i" => int_ovf_jump(code, op, ctx, OpCode::IntSubOvf),
        "int_mul_jump_if_ovf/Lii>i" => int_ovf_jump(code, op, ctx, OpCode::IntMulOvf),
        "catch_exception/L" => {
            // RPython `blackhole.py bhimpl_catch_exception(target)` —
            // "no-op when run normally" — and `pyjitpl.py
            // opimpl_catch_exception`:
            //
            //   def opimpl_catch_exception(self, target):
            //       """This is a no-op when run normally.  We can check that
            //       last_exc_value is a null ptr; it should have been set to None
            //       by the previous instruction.  If the previous instruction
            //       raised instead, finishframe_exception() should have been
            //       called and we would not be there."""
            //       assert not self.metainterp.last_exc_value
            //
            // The 2-byte target is metadata: when a `raise` fires on the
            // previous instruction, `handle_exception_in_frame`
            // (`blackhole.py`) reads it to redirect the unwinder
            // (consumed by `try_catch_exception_at` from the inline_call
            // SubRaise arm). Linear walk advances past the operand
            // without using the target.
            //
            // The RPython assert turns into a typed error here:
            // reaching `catch_exception/L` with `ctx.last_exc_value =
            // Some(_)` means either (a) the codewriter emitted a
            // catch_exception/L outside an exception-table position,
            // or (b) a previous catch handler didn't clear
            // last_exc_value after handling the raise. Either is a
            // codewriter-pass invariant violation.
            if ctx.last_exc_value.is_some() {
                return Err(DispatchError::CatchExceptionWithActiveException { pc: op.pc });
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "switch/id" => dispatch_switch_id(code, op, ctx),
        "residual_call_r_r/iRd>r" => dispatch_residual_call_iRd_kind(code, op, ctx, 'r'),
        // `_r_i/iRd>i` mirrors `_r_r/iRd>r` with the dst kind flipped to
        // int. RPython `pyjitpl.py _opimpl_residual_call1` is
        // exec-generated for `_callR` and `_callI` (Type::Ref vs
        // Type::Int return) — see resoperation.py `Type::Int =>
        // CallI`. EffectInfo classification + GUARD_NO_EXCEPTION
        // emission stay identical; only the result OpCode (CallI /
        // CallPureI) and dst writeback bank (registers_i) differ.
        "residual_call_r_i/iRd>i" => dispatch_residual_call_iRd_kind(code, op, ctx, 'i'),
        // `_ir_*/iIRd>X` extends the arglist with an i-bank list. RPython
        // `pyjitpl.py:_opimpl_residual_call*` exec-generates this for
        // callees taking both int + ref args (setup_call distributes
        // (argboxes_i, argboxes_r, argboxes_f=[])). Same EffectInfo
        // classification path; the only operand-shape change is the
        // I-list prefix between funcptr and the R-list.
        "residual_call_ir_r/iIRd>r" => dispatch_residual_call_iIRd_kind(code, op, ctx, 'r'),
        "residual_call_ir_i/iIRd>i" => dispatch_residual_call_iIRd_kind(code, op, ctx, 'i'),
        // `_irf_*/iIRFd>X` extends `_ir_*` with an f-bank list before the
        // descr (`pyjitpl.py _opimpl_residual_call3`, `boxes3`
        // argcode `pyjitpl.py`). EffectInfo classification +
        // guard emission identical; only the operand layout adds the F
        // suffix list.
        "residual_call_irf_r/iIRFd>r" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'r'),
        "residual_call_irf_i/iIRFd>i" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'i'),
        "residual_call_irf_f/iIRFd>f" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'f'),
        // `_*_v/iRd|iIRd|iIRFd` void variants — `_opimpl_residual_call*`
        // bodies discard the call result for void return kinds
        // (`pyjitpl.py opimpl_residual_call_r_v = _opimpl_residual_call1`,
        // `:1351 opimpl_residual_call_ir_v = _opimpl_residual_call2`,
        // `:1355 opimpl_residual_call_irf_v = _opimpl_residual_call3`;
        // `blackhole.py bhimpl_residual_call_*_v`).
        // EffectInfo classification + guard emission match the result-typed
        // siblings; only the operand layout drops the `>X` dst byte and
        // the writeback no-ops via `write_residual_call_result_to_dst`'s
        // `'v'` arm. `select_residual_call_opcode`'s `'v'` arm maps to the
        // `CallN` / `CallPureN` / `CallMayForceN` / `CallLoopinvariantN`
        // family per `resoperation.py Type::Void => CallN`.
        "residual_call_r_v/iRd" => dispatch_residual_call_iRd_kind(code, op, ctx, 'v'),
        "residual_call_ir_v/iIRd" => dispatch_residual_call_iIRd_kind(code, op, ctx, 'v'),
        "residual_call_irf_v/iIRFd" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'v'),
        // The `int_*` / `float_*` / `ptr_*` record families are routed
        // through `dispatch_regular_record` (see `arith.rs`) before this
        // match, so their arms no longer appear here.
        // `int_between/iii>i` decomposes a 3-arg range check at record
        // time per `pyjitpl.py opimpl_int_between` into
        // `INT_SUB + (INT_EQ on ConstInt(1) fast path | INT_SUB +
        // UINT_LT generic)`. Surfaces in `make_ll_isinstance`'s
        // range-covering branch (`rclass.py`).
        "int_between/iii>i" => int_between_record(code, op, ctx),
        // `int_floordiv/ii>i` and `int_mod/ii>i` intentionally absent:
        // `jtransform.py` rewrites both to
        // `direct_call(ll_int_py_*)` before jitcode emission.  The
        // trace-front lowering at `majit-translate/src/codegen.rs`
        // mirrors that rewrite for code reaching the JIT trace, so
        // this walker is never asked to dispatch the bare ops on a
        // traceable path.  Build-time helper graphs that still emit
        // the bare ops (e.g. `pyre/pyre-interpreter/src/baseobjspace.rs`
        // long_mod / long_div until the build-pipeline jtransform
        // port lands) get a `setdefault`-allocated dynamic byte and
        // resolve through BH dispatch only.
        // `cast_int_to_float` / `cast_int_to_ptr` / `cast_ptr_to_int`
        // route through `dispatch_regular_record` (see `arith.rs`
        // `unop_cast_record`) — part of the `pyjitpl.py`
        // exec-generated unary family.
        "ptr_nonzero/r>i" => ptr_nullity_record(code, op, ctx, true),
        "ptr_iszero/r>i" => ptr_nullity_record(code, op, ctx, false),
        "int_guard_value/i" => guard_value_record(code, op, ctx, GuardValueBank::Int),
        "ref_guard_value/r" => guard_value_record(code, op, ctx, GuardValueBank::Ref),
        "float_guard_value/f" => guard_value_record(code, op, ctx, GuardValueBank::Float),
        "abort/>r" => {
            // pyre-only result marker: `Assembler::encode_op`'s default
            // branch emits this when an untranslatable op's result is
            // classified `Ref` by `infer_concrete_from_op`'s
            // Abort→GcRef fallback.  Blackhole counterpart
            // (`handler_abort_result_marker_r`, `blackhole.rs`) is
            // a pure PC bump — no operand read, no register write, no
            // IR op recorded.  The actual abort signal is `abort/`
            // (BC_ABORT = 13), not this; reaching `abort/>r` in normal
            // flow is upstream-only an artefact of result-kind
            // classification and the dst slot is never read in a
            // post-abort code path.
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "abort/>i" => {
            // Int-result twin of `abort/>r`. Blackhole counterpart
            // `handler_abort_result_marker_i` (`blackhole.rs`) is a
            // pure PC bump — `infer_concrete_from_op`'s Abort→Int fallback
            // classifies the untranslatable op's result as Int, so the
            // dst byte is decoded (accounted for in `op.next_pc`) but
            // never written. Same rationale as `abort/>r`: the real abort
            // signal is `abort/` / `abort_permanent/`, not this marker.
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "abort/" => {
            // pyre-only `BC_ABORT` marker (`handler_abort_marker_pyre`,
            // `blackhole.rs`): the front-end emitted a graph node
            // with no dedicated `OpKind`, so reaching it means the body
            // carries an untranslatable op. The trace cannot be recorded;
            // surface a recoverable abort (production driver →
            // `TraceAction::Abort`).
            Err(DispatchError::AbortMarkerReached { pc: op.pc })
        }
        "abort_permanent/" => {
            // pyre-only `BC_ABORT_PERMANENT` fail-path
            // (`bhimpl_abort_permanent`, `blackhole.rs`): emitted for
            // paths that must always terminate the frame (BigInt-overflow
            // / unported-op fallbacks). Surface a permanent abort
            // (production driver → `TraceAction::AbortPermanent`) so the
            // location is never traced again.
            //
            // Capture the inline-sub-walk state here: an `op.pc` reached
            // inside an inlined callee is a callee coordinate the outer
            // walk's py_pc→jitcode tables cannot resolve, so the abort-point
            // flush must
            // decline. Latch the value at the marker because the sub-walk's
            // context is gone when the top-level driver reads the out-channel.
            ctx.session.borrow_mut().abort_in_subwalk = ctx.fbw_mode.inline_subwalk;
            Err(DispatchError::AbortPermanentMarkerReached { pc: op.pc })
        }
        // Heapcache-aware getfield reads. RPython
        // `pyjitpl.py opimpl_getfield_gc_<i|r|f>` →
        // `_opimpl_getfield_gc_any_pureornot` (`pyjitpl.py`)
        // dispatches the same way through `heapcache.get_field_updater`.
        // Walker handles the canonical `rd>X` shapes (Ref source);
        // pyre-specific `id>X` variants where the source is an int
        // register holding an unwrapped pointer are kind-flow kind-flow
        // territory and stay unsupported here.
        "getfield_gc_i/rd>i" => getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcI, 'i'),
        "getfield_gc_r/rd>r" => getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcR, 'r'),
        "getfield_gc_f/rd>f" => getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcF, 'f'),
        // RPython aliases the `_pure` jitcode spelling to the plain
        // handler (`pyjitpl.py:884 opimpl_getfield_gc_i_pure =
        // opimpl_getfield_gc_i`), so the recorded opcode stays plain
        // and purity is re-derived from `descr.is_always_pure()` by
        // the optimizer (`heap.py:641` const fold + invalidation-exempt
        // field cache).
        "getfield_gc_i_pure/rd>i" => {
            getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcI, 'i')
        }
        "getfield_gc_r_pure/rd>r" => {
            getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcR, 'r')
        }
        "getfield_gc_f_pure/rd>f" => {
            getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcF, 'f')
        }
        // Virtualizable getfield reads. RPython
        // `pyjitpl.py opimpl_getfield_vable_{i,r,f}` —
        // walker delegates to `TraceCtx::vable_getfield_{int,ref,float}`
        // which already implements `_nonstandard_virtualizable` fallback
        // to GETFIELD_GC + standard-vable `virtualizable_boxes[index]`
        // cache read (`majit-metainterp/src/trace_ctx.rs`).
        // Same `rd>X` operand shape as `getfield_gc_*`; only the
        // semantic handler routes through the vable mirror.
        "getfield_vable_i/rd>i" => getfield_vable_via_metainterp(code, op, ctx, 'i'),
        "getfield_vable_r/rd>r" => getfield_vable_via_metainterp(code, op, ctx, 'r'),
        "getfield_vable_f/rd>f" => getfield_vable_via_metainterp(code, op, ctx, 'f'),
        // Virtualizable setfield writes. RPython
        // `pyjitpl.py _opimpl_setfield_vable` — walker
        // delegates to `TraceCtx::vable_setfield`
        // (`majit-metainterp/src/trace_ctx.rs`) which handles the
        // `_nonstandard_virtualizable` fallback to SETFIELD_GC + the
        // standard-vable `virtualizable_boxes[index] = valuebox` +
        // `synchronize_virtualizable` mirror.  Operand shapes:
        // `setfield_vable_i/rid`, `setfield_vable_r/rrd`,
        // `setfield_vable_f/rfd` — value bank differs, no dst byte.
        "setfield_vable_i/rid" => setfield_vable_via_metainterp(code, op, ctx, 'i'),
        "setfield_vable_r/rrd" => setfield_vable_via_metainterp(code, op, ctx, 'r'),
        "setfield_vable_f/rfd" => setfield_vable_via_metainterp(code, op, ctx, 'f'),
        // Virtualizable array reads/writes + length. RPython
        // `pyjitpl.py _opimpl_{get,set}arrayitem_vable` /
        // `opimpl_arraylen_vable` — walker delegates to the
        // `TraceCtx::vable_{get,set}arrayitem_*` / `vable_arraylen_vable`
        // ports which already implement the `_nonstandard_virtualizable`
        // GC fallback and the standard-vable `virtualizable_boxes[index]`
        // cache path.  The `(VableArray, Array)` descr pair is resolved to
        // the vinfo's identity-keyed `(fdescr, adescr)` via
        // `vable_array_descrs_from_jitcode`.
        "getarrayitem_vable_i/ridd>i" => getarrayitem_vable_via_metainterp(code, op, ctx, 'i'),
        "getarrayitem_vable_r/ridd>r" => getarrayitem_vable_via_metainterp(code, op, ctx, 'r'),
        "getarrayitem_vable_f/ridd>f" => getarrayitem_vable_via_metainterp(code, op, ctx, 'f'),
        "setarrayitem_vable_i/riidd" => setarrayitem_vable_via_metainterp(code, op, ctx, 'i'),
        "setarrayitem_vable_r/rirdd" => setarrayitem_vable_via_metainterp(code, op, ctx, 'r'),
        "setarrayitem_vable_f/rifdd" => setarrayitem_vable_via_metainterp(code, op, ctx, 'f'),
        "arraylen_vable/rdd>i" => arraylen_vable_via_metainterp(code, op, ctx),
        // RPython `pyjitpl.py opimpl_hint_force_virtualizable(box)` is a thin
        // forward:
        //
        //   self.metainterp.gen_store_back_in_vable(box)
        //
        // `TraceCtx::gen_store_back_in_vable` hosts the whole body, including
        // the nonstandard-virtualizable and already-forced gates, so the arm
        // reads its one operand and hands it over. Operand layout `r`.
        "hint_force_virtualizable/r" => {
            let vable = read_ref_reg(code, op, 0, ctx)?;
            ctx.trace_ctx.gen_store_back_in_vable(vable);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // setfield_gc canonical shapes. `iid` / `ird` (int box)
        // shapes are pyre kind-flow kind-flow territory and stay
        // unsupported.
        "setfield_gc_i/rid" => setfield_gc_via_heapcache(code, op, ctx, 'i'),
        // USE_C_FORM short value (`assembler.py`): the stored
        // int is an inline signed byte read as a `ConstInt` box.
        "setfield_gc_i/rcd" => setfield_gc_via_heapcache(code, op, ctx, 'c'),
        "setfield_gc_r/rrd" => setfield_gc_via_heapcache(code, op, ctx, 'r'),
        "setfield_gc_f/rfd" => setfield_gc_via_heapcache(code, op, ctx, 'f'),
        // Raw memory carries no heapcache bookkeeping: where the `_gc_`
        // family consults and updates the cache, `pyjitpl.py`'s raw pair
        // only records.
        //
        //   def opimpl_raw_load_i(self, addrbox, offsetbox, arraydescr):
        //       return self.execute_with_descr(rop.RAW_LOAD_I, arraydescr,
        //                                      addrbox, offsetbox)
        //
        // Operand layout `iid>i`: 1B base + 1B offset + 2B descr + 1B dst.
        "raw_load_i/iid>i" => {
            let base = read_int_reg(code, op, 0, ctx)?;
            let offset = read_int_reg(code, op, 1, ctx)?;
            let descr = read_descr(code, op, 2, ctx)?;
            // pyjitpl.py:1025-1027 `opimpl_raw_load_i`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawLoadI, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawLoadI, majit_metainterp::counters::RECORDED_OPS);
            let result =
                ctx.trace_ctx
                    .record_op_with_descr(OpCode::RawLoadI, &[base, offset], descr);
            let dst = code[op.pc + 5] as usize;
            // The walk never performed the load, so the result carries no
            // recording-time concrete and readers decline instead of
            // consuming a value the walker did not observe.
            write_int_reg(ctx, op.pc, dst, result, ConcreteValue::Null)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // `_opimpl_raw_store` delegates to `execute_raw_store`:
        //
        //   self.execute_and_record(rop.RAW_STORE, arraydescr,
        //                           addrbox, offsetbox, valuebox)
        //
        // Records and nothing else — no cache update, no result register.
        // Operand layout `iiid`: 1B base + 1B offset + 1B value + 2B descr.
        "raw_store_i/iiid" => {
            let base = read_int_reg(code, op, 0, ctx)?;
            let offset = read_int_reg(code, op, 1, ctx)?;
            let value = read_int_reg(code, op, 2, ctx)?;
            let descr = read_descr(code, op, 3, ctx)?;
            // pyjitpl.py:1018-1022 `_opimpl_raw_store`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawStore, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawStore, majit_metainterp::counters::RECORDED_OPS);
            ctx.trace_ctx
                .record_op_with_descr(OpCode::RawStore, &[base, offset, value], descr);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // Heapcache-aware array reads/writes (canonical `rid>X` /
        // `ri{i,r,f}d` shapes).  Array indices are always int-classified,
        // so the index operand is always decoded from the `i` register
        // bank; non-canonical Ref-index shapes (`rrd>r`/`rrrd`/`rrfd`)
        // never arise.
        "getarrayitem_gc_i/rid>i" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcI, 'i')
        }
        "getarrayitem_gc_r/rid>r" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcR, 'r')
        }
        "getarrayitem_gc_f/rid>f" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcF, 'f')
        }
        // RPython `opimpl_getarrayitem_gc_{i,f,r}_pure`
        // — distinct opimpls (NOT aliased to the non-pure form,
        // unlike `getfield_gc_*_pure`).
        // Records `rop.GETARRAYITEM_GC_PURE_{I,F,R}` directly through
        // `_do_getarrayitem_gc_any(rop.GETARRAYITEM_GC_PURE_*, ...)`.
        // The ConstPtr+ConstInt constant-fold fast path requires BOTH the
        // array box and the index box to be `Const`; on the walker they
        // are generally runtime OpRefs, so the fold precondition is unmet
        // and the miss branch records the Pure rop.
        "getarrayitem_gc_i_pure/rid>i" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcPureI, 'i')
        }
        "getarrayitem_gc_r_pure/rid>r" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcPureR, 'r')
        }
        "getarrayitem_gc_f_pure/rid>f" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcPureF, 'f')
        }
        "setarrayitem_gc_i/riid" => setarrayitem_gc_via_heapcache(code, op, ctx, 'i'),
        // const-VALUE `c`-argcode form (USE_C_FORM `assembler.py`):
        // the store value is one inline signed byte → ConstInt, decoded
        // inside the handler.  Mirror of `setfield_gc_i/rcd`.
        "setarrayitem_gc_i/ricd" => setarrayitem_gc_via_heapcache(code, op, ctx, 'c'),
        "setarrayitem_gc_r/rird" => setarrayitem_gc_via_heapcache(code, op, ctx, 'r'),
        // `c`-argcode form (`assembler.py emit_const(allow_short
        // =True)`, USE_C_FORM `assembler.py`): the index is one
        // inline signed byte → ConstInt.  Same body as
        // [`setarrayitem_gc_via_heapcache`] with the index read from
        // the bytecode instead of an i-bank register.
        "setarrayitem_gc_r/rcrd" => {
            let array = read_ref_reg(code, op, 0, ctx)?;
            let index = OpRef::ConstInt(code[op.pc + 2] as i8 as i64);
            let value = read_ref_reg(code, op, 2, ctx)?;
            let descr = read_descr(code, op, 3, ctx)?;
            let descr_index = descr.index();
            // pyjitpl.py:736-744 `_opimpl_setarrayitem_gc_any`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::SetarrayitemGc, majit_metainterp::counters::OPS);
            ctx.trace_ctx.profiler().count_ops(
                OpCode::SetarrayitemGc,
                majit_metainterp::counters::RECORDED_OPS,
            );
            ctx.trace_ctx.record_op_with_descr(
                OpCode::SetarrayitemGc,
                &[array, index, value],
                descr,
            );
            ctx.trace_ctx
                .heapcache_setarrayitem(array, index, descr_index, value);
            walker_fill_materialized_array(ctx, array, index, value);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "setarrayitem_gc_f/rifd" => setarrayitem_gc_via_heapcache(code, op, ctx, 'f'),
        // `arraylen_gc` — `blackhole.py bhimpl_arraylen_gc`
        // (@arguments("cpu","r","d", returns="i")).  Operand layout `rd>i`:
        // 1B r-reg(array) + 2B descr + 1B i-reg(dst).  Delegates to
        // `state::opimpl_arraylen_gc` (heapcache-aware length tracking,
        // pyjitpl.py).
        "arraylen_gc/rd>i" => {
            let array = read_ref_reg(code, op, 0, ctx)?;
            let descr = read_descr(code, op, 1, ctx)?;
            let result = crate::state::opimpl_arraylen_gc(&mut ctx.trace_ctx, array, descr);
            let dst = code[op.pc + 4] as usize;
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // RPython `pyjitpl.py opimpl_assert_not_none` and
        // `opimpl_record_exact_class` are heapcache hints: consult the
        // cache, record once, then stamp what the record proved. Both
        // bodies already live on `TraceCtx` — the walker reads the
        // operands and delegates, so the two tracers cannot drift.
        //
        // Operand layout `r`: 1B ref reg.
        "assert_not_none/r" => {
            let opref = read_ref_reg(code, op, 0, ctx)?;
            let known_nonnull = ctx.trace_ctx.heap_cache().is_nullity_known(opref, |op| {
                op.inline_const_to_value().and_then(|value| match value {
                    Value::Int(value) => Some(value),
                    Value::Ref(value) => Some(value.0 as i64),
                    _ => None,
                })
            }) == Some(true);
            let concrete = if known_nonnull {
                0
            } else {
                // `trace_assert_not_none` reaches `executor.py
                // do_assert_not_none` on a cache miss, so only that path needs
                // the pointer value. An absent shadow cannot prove non-nullness.
                let ConcreteValue::Ref(ptr) = read_ref_reg_concrete(code, op, 0, ctx) else {
                    return Err(DispatchError::UnsupportedOpname {
                        pc: op.pc,
                        key: op.key,
                    });
                };
                ptr as i64
            };
            ctx.trace_ctx.trace_assert_not_none(opref, concrete);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // Operand layout `ri`: 1B ref reg + 1B int reg holding the class
        // vtable address. A non-constant class operand is skipped inside
        // the helper, matching the `isinstance(clsbox, Const)` gate.
        "record_exact_class/ri" => {
            let opref = read_ref_reg(code, op, 0, ctx)?;
            let cls = read_int_reg(code, op, 1, ctx)?;
            ctx.trace_ctx.trace_record_exact_class(opref, cls);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // RPython `pyjitpl.py opimpl_new` delegates to `execute_new`:
        //
        //   resbox = self.execute_and_record(rop.NEW, typedescr)
        //   self.heapcache.new(resbox)
        //   return resbox
        //
        // Same `d>r` layout and same posture as `new_with_vtable` below,
        // minus the class stamp — a plain struct allocation carries no
        // vtable word, so nothing is known about its class.
        "new/d>r" => {
            let descr = read_descr(code, op, 0, ctx)?;
            // pyjitpl.py:624-629 `execute_new`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::New, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::New, majit_metainterp::counters::RECORDED_OPS);
            let resbox = ctx.trace_ctx.record_op_with_descr(OpCode::New, &[], descr);
            ctx.trace_ctx.heap_cache_mut().new_object(resbox);
            let dst = code[op.pc + 3] as usize;
            write_ref_reg(ctx, op.pc, dst, resbox, ConcreteValue::Null)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // RPython `pyjitpl.py opimpl_new_with_vtable` delegates straight to
        // `execute_new_with_vtable`:
        //
        //   resbox = self.execute_and_record(rop.NEW_WITH_VTABLE, descr)
        //   self.heapcache.new(resbox)
        //   self.heapcache.class_now_known(resbox)
        //   return resbox
        //
        // Operand layout `d>r`: 2B descr + 1B dst (`assembler.rs`
        // `new_with_vtable` pushes the descr u16 then the result reg).
        "new_with_vtable/d>r" => {
            let descr = read_descr(code, op, 0, ctx)?;
            // `class_now_known` takes the vtable address: pyre tracks the
            // concrete class pointer where upstream only raises HF_KNOWN_CLASS.
            let known_class = descr.as_size_descr().map(|size| size.vtable() as i64);
            // pyjitpl.py:624-629 `execute_new_with_vtable`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::NewWithVtable, majit_metainterp::counters::OPS);
            ctx.trace_ctx.profiler().count_ops(
                OpCode::NewWithVtable,
                majit_metainterp::counters::RECORDED_OPS,
            );
            let resbox = ctx
                .trace_ctx
                .record_op_with_descr(OpCode::NewWithVtable, &[], descr);
            ctx.trace_ctx.heap_cache_mut().new_object(resbox);
            if let Some(class) = known_class {
                ctx.trace_ctx
                    .heap_cache_mut()
                    .class_now_known(resbox, class);
            }
            let dst = code[op.pc + 3] as usize;
            // No recording-time concrete: the walk never allocated a real
            // object, so readers of the result decline instead of consuming a
            // stale value — the posture `new_array_clear` keeps whenever it
            // cannot materialize a backing block.
            write_ref_reg(ctx, op.pc, dst, resbox, ConcreteValue::Null)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // RPython `pyjitpl.py opimpl_new_array_clear` —
        // `_opimpl_new_array(rop.NEW_ARRAY_CLEAR, lengthbox,
        // arraydescr)` records the op and seeds the heapcache via
        // `heapcache.new_array(resbox, lengthbox)`.  Operand layout
        // per `bhimpl_new_array_clear @arguments("cpu","i","d",
        // returns="r")`: 1B i-reg(length) + 2B descr + 1B r-reg(dst).
        // The `cd>r` arm is the `c`-argcode form (inline signed-byte
        // length, USE_C_FORM): identical byte layout, length decoded
        // as ConstInt.
        "new_array_clear/id>r" | "new_array_clear/cd>r" => {
            let length = if op.key == "new_array_clear/cd>r" {
                OpRef::ConstInt(code[op.pc + 1] as i8 as i64)
            } else {
                read_int_reg(code, op, 0, ctx)?
            };
            let descr = read_descr(code, op, 1, ctx)?;
            let is_ref_array = descr
                .as_array_descr()
                .map_or(false, |a| a.is_array_of_pointers());
            // pyjitpl.py:631-637 `_opimpl_new_array`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::NewArrayClear, majit_metainterp::counters::OPS);
            ctx.trace_ctx.profiler().count_ops(
                OpCode::NewArrayClear,
                majit_metainterp::counters::RECORDED_OPS,
            );
            let resbox =
                ctx.trace_ctx
                    .record_op_with_descr(OpCode::NewArrayClear, &[length], descr);
            // heapcache.py `new_array(box, lengthbox)` adds
            // the virtual/unescaped flags only when `lengthbox` is a
            // Const ("only constant-length arrays are virtuals").
            ctx.trace_ctx
                .heap_cache_mut()
                .new_array(resbox, length, length.is_constant());
            let dst = code[op.pc + 4] as usize;
            // Walker virtual-force (module-global fresh-container off-by-one
            // fix): for a constant-length array of refs on the items-block GC
            // path, eagerly allocate a real GC-traced block and stamp it as the
            // recording-time concrete VALUE (`Op.value` — GC-rooted via
            // `walk_op_const_ptr_refs`, NOT `make_constant`, so the compiled
            // trace still allocates fresh per iteration; same posture as an
            // executed residual's observed result). A later BUILD_LIST /
            // BUILD_TUPLE residual then resolves a concrete array arg and runs
            // during the walk, committing a container stored into an escaping
            // slot (e.g. a module-global cell) for the recorded iteration
            // instead of forcing the void-store abort. The companion
            // `walker_fill_materialized_array` fills slots at `setarrayitem_gc`;
            // if any element/index is non-concrete it reverts the array to the
            // no-concrete sentinel so the residual declines (abort, as before).
            // Non-ref / non-constant arrays and the gate-off fallback keep the
            // Null posture.
            let mut concrete = ConcreteValue::Null;
            if is_ref_array && length.is_constant() {
                if let Some(majit_ir::Value::Int(n)) = ctx.trace_ctx.box_value(length) {
                    if let Ok(cap) = usize::try_from(n) {
                        if let Some(block) = unsafe {
                            pyre_object::object_array::alloc_cleared_ref_items_block_gc(cap)
                        } {
                            if ctx.trace_ctx.try_set_opref_concrete(
                                resbox,
                                majit_ir::Value::Ref(majit_ir::GcRef(block as usize)),
                            ) {
                                concrete = ConcreteValue::Ref(block as pyre_object::PyObjectRef);
                            }
                        }
                    }
                }
            }
            write_ref_reg(ctx, op.pc, dst, resbox, concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "int_copy/i>i" => {
            // RPython `pyjitpl.py _opimpl_any_copy(self, box) → box`
            // + `@arguments("box")` + `>i` result coding: read src
            // register, write the same OpRef into the dst slot. Pypy
            // records *no* IR op for a copy — pure SSA-level rename.
            // Operand layout `i>i`: 1B src + 1B dst.
            //
            // Int-bank concrete shadow: propagate the source slot's Int-bank concrete
            // shadow alongside the symbolic OpRef, mirroring the
            // `ref_copy/r>r` concrete-shadow chain.  Without this, a
            // `goto_if_not/iL` reading the dst slot wouldn't see the
            // concrete and would surface `GotoIfNotValueNotConcrete`
            // even when the source had a known concrete (e.g. a
            // constant Int seeded by `allocate_callee_register_banks`).
            let src_val = read_int_reg(code, op, 0, ctx)?;
            let src_concrete = read_int_reg_concrete(code, op, 0, ctx);
            let dst = code[op.pc + 2] as usize;
            write_int_reg(ctx, op.pc, dst, src_val, src_concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "int_copy/c>i" => {
            // `int_copy/c>i` — USE_C_FORM short source (`assembler.py`):
            // the small ConstInt is one inline signed byte (`signedord`,
            // `blackhole.py`), not a `registers_i` slot. Like `int_copy/i>i`
            // this records no IR op (pure SSA copy); the dst is seeded with the
            // constant box plus its concrete shadow so a downstream
            // `goto_if_not/iL` / `switch/id` can fold. Operand layout `c>i`:
            // 1B signed const + 1B dst.
            let value = code[op.pc + 1] as i8 as i64;
            let dst = code[op.pc + 2] as usize;
            write_int_reg(
                ctx,
                op.pc,
                dst,
                OpRef::ConstInt(value),
                ConcreteValue::Int(value),
            )?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "float_copy/f>f" => {
            // Float-bank sibling of `int_copy/i>i` — pure SSA-level
            // rename, no IR op recorded. Operand layout `f>f`: 1B src
            // + 1B dst.
            let src_val = read_float_reg(code, op, 0, ctx)?;
            let dst = code[op.pc + 2] as usize;
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = src_val;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_push/r" => {
            // Blackhole `bhimpl_ref_push` (`blackhole.py`):
            // `self.tmpreg_r = a`.  `insert_renamings` (`flatten.py`)
            // emits `*_push`/`*_pop` around a cyclic parallel move — the
            // swap `r_a <-> r_b` lowers to `push r_b; copy r_b<-r_a;
            // pop r_a` so the overwritten value survives in the tmpreg.
            // Pure SSA-level scratch move, no IR op recorded.  Operand
            // layout `r`: 1B src.
            let src_val = read_ref_reg(code, op, 0, ctx)?;
            let src_concrete = read_ref_reg_concrete(code, op, 0, ctx);
            {
                let mut sess = ctx.session.borrow_mut();
                sess.tmpreg_r = src_val;
                sess.tmpreg_r_concrete = src_concrete;
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_pop/>r" => {
            // Blackhole `bhimpl_ref_pop` (`blackhole.py`):
            // `return self.get_tmpreg_r()`.  Reads the value stashed by
            // the matching `ref_push/r` back into a dst register, in
            // lock-step with its concrete shadow.  Operand layout `>r`:
            // 1B dst.
            let (val, concrete) = {
                let sess = ctx.session.borrow();
                (sess.tmpreg_r, sess.tmpreg_r_concrete)
            };
            let dst = code[op.pc + 1] as usize;
            write_ref_reg(ctx, op.pc, dst, val, concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "int_push/i" => {
            // Blackhole `bhimpl_int_push` (`blackhole.py`):
            // `self.tmpreg_i = a`.  Int-bank sibling of `ref_push/r`,
            // stashing the concrete shadow alongside the OpRef.
            // Operand layout `i`: 1B src.
            let src_val = read_int_reg(code, op, 0, ctx)?;
            let src_concrete = read_int_reg_concrete(code, op, 0, ctx);
            {
                let mut sess = ctx.session.borrow_mut();
                sess.tmpreg_i = src_val;
                sess.tmpreg_i_concrete = src_concrete;
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "int_pop/>i" => {
            // Blackhole `bhimpl_int_pop` (`blackhole.py`).  Writes through
            // `write_int_reg` so `concrete_registers_i[dst]` tracks the value
            // the move restores instead of the one it overwrote.
            // Operand layout `>i`: 1B dst.
            let (val, concrete) = {
                let sess = ctx.session.borrow();
                (sess.tmpreg_i, sess.tmpreg_i_concrete)
            };
            let dst = code[op.pc + 1] as usize;
            write_int_reg(ctx, op.pc, dst, val, concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "float_push/f" => {
            // Blackhole `bhimpl_float_push` (`blackhole.py`).
            // Operand layout `f`: 1B src.
            let src_val = read_float_reg(code, op, 0, ctx)?;
            ctx.session.borrow_mut().tmpreg_f = src_val;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "float_pop/>f" => {
            // Blackhole `bhimpl_float_pop` (`blackhole.py`).  The Float bank
            // resolves concretes from the OpRef itself
            // (`read_float_reg_concrete`) rather than a side table, so moving
            // the OpRef carries the concrete with it.
            // Operand layout `>f`: 1B dst.
            let val = ctx.session.borrow().tmpreg_f;
            let dst = code[op.pc + 1] as usize;
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = val;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_copy/r>r" => {
            // Ref-bank sibling of `int_copy/i>i`. Same RPython
            // `_opimpl_any_copy` body — the `>r` suffix only changes
            // which register bank the writeback lands in. Const-source
            // variants (codewriter `emit_ref_copy!` with `ConstRef`)
            // resolve via the constants window of `registers_r`: the
            // assembler's `load_const_r` patches the src operand to a
            // constants-pool register index in `[num_regs_r,
            // num_regs_and_consts_r)`, which `setposition` (RPython
            // `pyjitpl.py`) pre-populates with the const OpRef.
            // No IR op recorded.
            let src_val = read_ref_reg(code, op, 0, ctx)?;
            // Propagate the source slot's concrete
            // shadow alongside the symbolic OpRef.  This is the
            // critical chain: catch_exception → seeds last_exc_value
            // / concrete → `last_exc_value/>r` writes both into
            // `registers_r[X]` and `concrete_registers_r[X]` →
            // `ref_copy/r>r` copies X to Y → a follow-on `raise/r`
            // reads Y and finds the correct concrete to emit
            // GUARD_CLASS against.  Without this propagation the
            // copy chain wipes the concrete and silently disables
            // the guard.
            let src_concrete = read_ref_reg_concrete(code, op, 0, ctx);
            let dst = code[op.pc + 2] as usize;
            write_ref_reg(ctx, op.pc, dst, src_val, src_concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_return/r" => {
            // RPython `pyjitpl.py:opimpl_ref_return(self, value)` calls
            // `metainterp.finishframe(value)`. Two branches by frame depth:
            //
            //   * Outermost frame → `compile_done_with_this_frame` (pyjitpl.py)
            //     records `rop.FINISH(value)` with
            //     `done_with_this_frame_descr_ref`. Trace ends.
            //   * Nested frame → `metainterp.popframe()` returns control to
            //     the caller's metainterp loop with `value` in hand; the
            //     caller's `_opimpl_inline_call*` lands `value` in its
            //     `>r` slot via `make_result_of_lastop`.
            //
            // Walker selects between the two via `ctx.is_top_level`.
            let result = read_ref_reg(code, op, 0, ctx)?;
            // PyPy `box.value = result` parity at the frame boundary:
            // the callee's slot-keyed concrete shadow (`concrete_registers_r`)
            // carries the live PyObject pointer; mirror it onto the
            // OpRef-keyed `opref_concrete` channel so the caller's
            // `concrete_from_recorded_opref` (in `dispatch_inline_call_*_kind`)
            // sees the stamped Box.value.  Skips constants — `TraceCtx::constants
            // .get_value` is the authoritative shadow for those.
            if !result.is_constant() {
                if let ConcreteValue::Ref(ptr) = read_ref_reg_concrete(code, op, 0, ctx) {
                    if !ptr.is_null() {
                        ctx.trace_ctx.set_opref_concrete(
                            result,
                            majit_ir::Value::Ref(majit_ir::GcRef(ptr as usize)),
                        );
                    }
                }
            }
            if ctx.is_top_level {
                // Slice b: route the loop-free portal exit through
                // `TraceAction::Finish` so the compile pipeline records
                // the FINISH from `finish_args`.  Re-box to Type::Ref +
                // store_token_in_vable, then stash the payload; do NOT
                // call `ctx.trace_ctx.finish()` here (would double-record).
                //
                // No-replay portal exit: also stash the CONCRETE return
                // so `eval.rs` returns the walk's result directly instead
                // of re-running the compiled trace against the already
                // side-effected heap (the walk consumed it).  A null /
                // unknown concrete leaves the cell `None` → degrade.
                if let ConcreteValue::Ref(ptr) = read_ref_reg_concrete(code, op, 0, ctx) {
                    if !ptr.is_null() {
                        fbw_finish_concrete_set(ConcreteValue::Ref(ptr));
                    }
                }
                fbw_terminate_with_finish(ctx, result, op.pc)?;
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "int_return/i" => {
            // RPython `pyjitpl.py opimpl_int_return = _opimpl_any_return`
            // (pyjitpl.py `_opimpl_any_return: self.metainterp.finishframe(box)`).
            // Top-level: `compile_done_with_this_frame` (pyjitpl.py)
            // records `FINISH([value], descr=done_with_this_frame_descr_int)`.
            // Sub-walk: `SubReturn { Some(value) }` — caller's
            // `inline_call_*_i` would land the int OpRef in its `>i` slot.
            // Operand layout `i`: 1B int register at op.pc+1.
            let result = read_int_reg(code, op, 0, ctx)?;
            // PyPy `box.value = result` parity at the frame boundary —
            // see `ref_return/r` comment above for rationale.
            if !result.is_constant() {
                if let ConcreteValue::Int(v) = read_int_reg_concrete(code, op, 0, ctx) {
                    ctx.trace_ctx
                        .set_opref_concrete(result, majit_ir::Value::Int(v));
                }
            }
            if ctx.is_top_level {
                // Slice b: portal-exit FINISH carries Type::Ref even for
                // an int return (the eval_loop_jit result_type is REF),
                // so `fbw_ensure_boxed_for_ca` re-boxes via wrapint.
                //
                // No-replay portal exit: stash the concrete int so
                // `eval.rs` returns the walk's result directly (re-boxed
                // via `ConcreteValue::to_pyobj`).
                if let ConcreteValue::Int(v) = read_int_reg_concrete(code, op, 0, ctx) {
                    fbw_finish_concrete_set(ConcreteValue::Int(v));
                }
                fbw_terminate_with_finish(ctx, result, op.pc)?;
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "int_return/c" => {
            // USE_C_FORM short source (`assembler.py`): the return value
            // is one inline signed byte (`signedord`, `blackhole.py`),
            // not a `registers_i` slot — so `result` is a `ConstInt` that
            // already carries its value (no `set_opref_concrete` needed).
            // Otherwise identical to `int_return/i`. Operand layout `c`:
            // 1B signed const at op.pc+1.
            let value = code[op.pc + 1] as i8 as i64;
            let result = OpRef::ConstInt(value);
            if ctx.is_top_level {
                fbw_finish_concrete_set(ConcreteValue::Int(value));
                fbw_terminate_with_finish(ctx, result, op.pc)?;
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "float_return/f" => {
            // RPython `pyjitpl.py opimpl_float_return = _opimpl_any_return`.
            // Top-level: `compile_done_with_this_frame` (pyjitpl.py)
            // records `FINISH([value], descr=done_with_this_frame_descr_float)`.
            // Sub-walk: `SubReturn { Some(value) }` carrying the float
            // OpRef — same enum variant as int/ref because the OpRef is
            // bank-agnostic; the caller's inline_call variant decides
            // which bank to write into.
            // Operand layout `f`: 1B float register at op.pc+1.
            let result = read_float_reg(code, op, 0, ctx)?;
            if ctx.is_top_level {
                // Slice b: portal-exit FINISH carries Type::Ref;
                // `fbw_ensure_boxed_for_ca` re-boxes the float via
                // wrapfloat.
                if let Some(majit_ir::Value::Float(v)) = ctx.trace_ctx.box_value(result) {
                    fbw_finish_concrete_set(ConcreteValue::Float(v));
                }
                fbw_terminate_with_finish(ctx, result, op.pc)?;
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "void_return/" => {
            // RPython `pyjitpl.py opimpl_void_return`:
            //
            //   @arguments()
            //   def opimpl_void_return(self):
            //       self.metainterp.finishframe(None)
            //
            // Top-level: `compile_done_with_this_frame` (pyjitpl.py)
            // takes the `result_type == VOID` branch — `exits = []`,
            // `token = sd.done_with_this_frame_descr_void`. The FINISH
            // carries no value.
            // Sub-walk: `SubReturn { None }` — RPython's
            // `_opimpl_inline_call_*_v` variants don't write a dst
            // register on the caller side (the codewriter emits no `>X`
            // marker for void calls).
            // No operand bytes (the `/` argcodes is empty).
            if ctx.is_top_level {
                // Slice b: route the void portal exit through
                // `TraceAction::Finish` (empty args) so the compile
                // pipeline records the FINISH(void) from `finish_args`,
                // mirroring the three value-returning arms.  Store the
                // assembler token in the vable + GUARD_NOT_FORCED_2 like
                // those arms, then stash a void-marked payload; do NOT
                // call `ctx.trace_ctx.finish()` here (would double-record).
                //
                // No-replay portal exit: stash `Null` (= void → None at
                // the consume site) so a side-effecting void function
                // returns directly instead of re-running its already
                // applied effects.
                fbw_finish_concrete_set(ConcreteValue::Null);
                fbw_terminate_void_with_finish(ctx, op.pc)?;
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((DispatchOutcome::SubReturn { result: None }, op.next_pc))
            }
        }
        "raise/r" => {
            // RPython `pyjitpl.py opimpl_raise(exc_value_box, orgpc)`:
            //   if not self.metainterp.heapcache.is_class_known(exc_value_box):
            //       clsbox = self.cls_of_box(exc_value_box)
            //       self.metainterp.generate_guard(rop.GUARD_CLASS, exc_value_box,
            //                                      clsbox, resumepc=orgpc)
            //   self.metainterp.class_of_last_exc_is_const = True
            //   self.metainterp.last_exc_value = exc_value_box.getref(...)
            //   self.metainterp.last_exc_box = exc_value_box
            //   self.metainterp.popframe()
            //   self.metainterp.finishframe_exception()
            //
            // Walker dual behaviour:
            //   * `is_top_level` → outermost FINISH (above).
            //   * sub-walk frame → propagate `SubRaise { exc }` to the
            //     caller's `inline_call_*` handler.
            //
            // GUARD_CLASS emission is enabled: `concrete_registers_r`
            // is `&mut` and every walker write goes through
            // [`write_ref_reg`], so the concrete tracks the symbolic
            // in lock-step.  Were the shadow an immutable dispatch-entry
            // snapshot, sibling handlers could rewrite `registers_r[dst]`
            // without updating it, so this read would find a stale
            // concrete and silently skip the guard.  Read-after-write
            // now returns the right concrete (or `Null` if the handler
            // didn't know, in which case the guard skips — same
            // semantics as the snapshot's tail).
            //
            // Mirrors the retired trait-side raise path.  The read at
            // `ob_header.ob_type` resolves to the per-`ExcKind` `PyType`
            // static (`interp_exceptions.rs::exc_kind_to_pytype`), so the
            // emitted `GuardClass` discriminates the actual subclass.
            // Stashes the concrete into `ctx.last_exc_value_concrete`
            // so a downstream
            // `last_exc_value/>r` can propagate it into its dst slot.
            let exc = read_ref_reg(code, op, 0, ctx)?;
            let mut concrete_exc = read_ref_reg_concrete(code, op, 0, ctx);
            if matches!(concrete_exc, ConcreteValue::Ref(p) if p.is_null())
                && let Some(Value::Ref(gc_ref)) = ctx.trace_ctx.box_value(exc)
                && gc_ref != majit_ir::GcRef::NO_CONCRETE
            {
                let ptr = gc_ref.as_usize() as pyre_object::PyObjectRef;
                if !ptr.is_null() && unsafe { pyre_object::is_exception(ptr) } {
                    concrete_exc = ConcreteValue::Ref(ptr);
                }
            }
            // `pyjitpl.py opimpl_raise` calls
            // `generate_guard(GUARD_CLASS, exc_value_box, clsbox,
            // resumepc=orgpc)`; the first line of `generate_guard`
            // (`pyjitpl.py`) is `if isinstance(box, Const):
            // return`. Const exception boxes already pin the class so
            // no guard is needed. Resume-data capture is omitted here
            // for the same reason as `goto_if_not/iL` above — see that
            // arm's comment for the trace-emitter endgame.
            if !exc.is_constant() {
                if let ConcreteValue::Ref(exc_ptr) = concrete_exc {
                    if !exc_ptr.is_null() && !ctx.trace_ctx.heap_cache().is_class_known(exc) {
                        let exc_class_ptr = unsafe {
                            (*(exc_ptr as *const pyre_object::interp_exceptions::W_BaseException))
                                .ob_header
                                .ob_type
                        };
                        let cls_const = ctx.trace_ctx.const_int(exc_class_ptr as usize as i64);
                        ctx.trace_ctx
                            .record_guard(OpCode::GuardClass, &[exc, cls_const], 0);
                        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
                        ctx.trace_ctx
                            .heap_cache_mut()
                            .class_now_known(exc, exc_class_ptr as usize as i64);
                    }
                }
            }
            ctx.last_exc_value = Some(exc);
            ctx.last_exc_value_concrete = concrete_exc;
            ctx.fbw_mode.class_of_last_exc_is_const = true;
            let freshly_normalized = fbw_built_exc_take(exc);
            if !recording_instruction_is_bare_reraise(ctx, op.pc) {
                let caught_in_frame = try_catch_exception_at(code, op.next_pc).is_some();
                if caught_in_frame {
                    // Unless this walk built the exception it may already
                    // carry callee nodes, so the general case prepends onto
                    // its chain rather than starting a fresh one.  Either
                    // recorder declining leaves the node to the opaque hook,
                    // which builds a frame from the callee's code + globals
                    // for the level the walk has no box for.  That frame
                    // carries no locals and its own identity, so it is a
                    // weaker node than the live one — still a node whose
                    // `tb_frame.f_code` answers the right code object.
                    let emit_runtime = if freshly_normalized {
                        !record_fresh_application_traceback(ctx, exc, concrete_exc, op.pc)
                    } else {
                        !record_prepend_application_traceback(ctx, exc, concrete_exc, op.pc)
                    };
                    record_inline_application_traceback(
                        ctx,
                        exc,
                        concrete_exc,
                        op.pc,
                        false,
                        emit_runtime,
                    );
                    record_top_level_application_traceback(
                        ctx,
                        exc,
                        concrete_exc,
                        op.pc,
                        false,
                        emit_runtime,
                    );
                }
            }
            // Route the top-level raise through `SubRaise` so walk()'s
            // SubRaise arm runs the in-frame
            // `catch_exception/L` lookahead (`finishframe_lookahead_at`) and
            // jumps into the handler instead of recording the top-level
            // exit-frame finish (which escapes a try/except as a no-payload
            // Terminate abort).
            Ok((
                DispatchOutcome::SubRaise {
                    exc,
                    exc_concrete: concrete_exc,
                },
                op.next_pc,
            ))
        }
        "last_exc_value/>r" => {
            // RPython parity: `pyjitpl.py opimpl_last_exc_value`:
            //
            //   @arguments()
            //   def opimpl_last_exc_value(self):
            //       exc_value = self.metainterp.last_exc_value
            //       assert exc_value
            //       return self.metainterp.last_exc_box
            //
            // Reads no operand; the `>r` decorator writes the result into
            // `registers_r[dst]`. No IR op recorded — the standing
            // `metainterp.last_exc_box` (mirrored here as
            // `ctx.last_exc_value`) is already a recorder OpRef from when
            // `raise/r` set it. This is a pure SSA-rename of the
            // exception slot into a Ref-bank dst, mirroring how
            // `int_copy/i>i` and `_opimpl_any_copy` collapse to a
            // register move without recording.
            //
            // Operand layout `>r`: 1B dst register only (the `>r` arg is
            // the writeback marker, not a separate operand byte; the dst
            // byte sits at op.pc+1).
            //
            // Forward-prep status: the opname is registered in
            // `wire_handler("last_exc_value/>r", handler_last_exc_value)`
            // (`blackhole.rs`) and `m.insert("last_exc_value/>r",
            // BC_LAST_EXC_VALUE)` (`jitcode/mod.rs`), but pyre's
            // codewriter does not currently emit `FlatOp::LastExcValue`
            // for any traced Python arm — `dump_unsupported_opnames_in_insns_table`
            // confirms the opname is absent from `OUT_DIR/insns.bin`.
            // The handler matches RPython's unconditional `setup_insns`
            // registration so it's ready when an except-handler arm
            // (e.g. `BC_LAST_EXC_VALUE` consumer in CPython 3.14
            // `LOAD_SPECIAL`/`CHECK_EXC_MATCH` lowering) lands.
            let exc = ctx
                .last_exc_value
                .ok_or(DispatchError::LastExcValueWithoutActiveException { pc: op.pc })?;
            let dst = code[op.pc + 1] as usize;
            // Propagate the standing exception's
            // concrete shadow into the dst slot.  `ctx.last_exc_value_
            // concrete` is the live `PyObjectRef` (seeded by either an
            // adapter caller or an earlier walker `raise/r`). This lets a follow-on `raise/r` reading
            // `registers_r[dst]` find a non-Null concrete and emit the
            // correct GUARD_CLASS.
            let exc_concrete = ctx.last_exc_value_concrete;
            write_ref_reg(ctx, op.pc, dst, exc, exc_concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "last_exception/>i" => {
            // RPython parity: `pyjitpl.py opimpl_last_exception`:
            //
            //   @arguments()
            //   def opimpl_last_exception(self):
            //       exc_value = self.metainterp.last_exc_value
            //       assert exc_value
            //       assert self.metainterp.class_of_last_exc_is_const
            //       exc_cls = rclass.ll_cast_to_object(exc_value).typeptr
            //       return ConstInt(ptr2int(exc_cls))
            //
            // The class pointer is the standing exception's `ob_type`.
            // `read_typeptr_from_exception` (`dispatch.rs`) routes
            // through `cpu.cls_of_box`. Walker reads the live concrete
            // exception's `W_BaseException.ob_header.ob_type` directly. The result is
            // a `ConstInt(typeptr)` — no IR op recorded, matching
            // RPython's `return ConstInt(...)`.
            //
            // Operand layout `>i`: 1B dst register only; the dst byte sits
            // at `op.pc + 1`.
            let exc = ctx
                .last_exc_value
                .ok_or(DispatchError::LastExceptionWithoutActiveException { pc: op.pc })?;
            let typeptr = if let Some(cls) = ctx.trace_ctx.heap_cache().get_known_class(exc) {
                cls
            } else {
                let exc_ptr = match ctx.last_exc_value_concrete {
                    ConcreteValue::Ref(p) if !p.is_null() => p,
                    _ => {
                        return Err(DispatchError::LastExceptionWithoutActiveException {
                            pc: op.pc,
                        });
                    }
                };
                unsafe {
                    (*(exc_ptr as *const pyre_object::interp_exceptions::W_BaseException))
                        .ob_header
                        .ob_type as i64
                }
            };
            let dst = code[op.pc + 1] as usize;
            let cls_const = ctx.trace_ctx.const_int(typeptr);
            write_int_reg(ctx, op.pc, dst, cls_const, ConcreteValue::Int(typeptr))?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "reraise/" => {
            // RPython parity: `pyjitpl.py opimpl_reraise(self)` —
            //
            //   assert self.metainterp.last_exc_value
            //   self.metainterp.popframe()
            //   self.metainterp.finishframe_exception()
            //
            // Reads no operand; uses the standing `metainterp.last_exc_value`
            // which was set either by an earlier `raise/r` in this frame
            // or — when the unwinder routed into a `catch_exception`
            // handler — by the inline_call SubRaise arm just before
            // jumping to the handler PC.
            //
            // Walker behaviour mirrors `raise/r`'s dual-frame routing:
            //   * top-level → outermost FINISH(last_exc_value,
            //     exit_frame_with_exception_descr_ref).
            //   * sub-walk → SubRaise{exc=last_exc_value}, bubbling
            //     through the parent's inline_call handler (which may
            //     itself catch via `catch_exception/L` lookahead).
            //
            // `last_exc_value == None` violates the RPython assert and
            // surfaces as `ReraiseWithoutLastExcValue` (codewriter
            // invariant: `reraise` only emits inside a `catch_exception`
            // body or after an explicit `raise`).
            let exc = ctx
                .last_exc_value
                .ok_or(DispatchError::ReraiseWithoutLastExcValue { pc: op.pc })?;
            // Symmetric with `raise/r`.
            Ok((
                DispatchOutcome::SubRaise {
                    exc,
                    exc_concrete: ctx.last_exc_value_concrete,
                },
                op.next_pc,
            ))
        }
        // The `i` spelling is what the assembler emits once the jitdriver
        // index outstrips a signed byte. Its leading byte names an Int-bank
        // slot containing the jdindex; the `c` spelling stores the jdindex
        // directly in that byte.
        "jit_merge_point/cIRFIRF" | "jit_merge_point/iIRFIRF" => {
            // RPython parity: `opimpl_jit_merge_point` →
            // `reached_loop_header`. pyre's retired
            // trait mirror was `close_loop_args`.
            //
            // The JitCode merge point carries its greens + reds inline
            // (`blackhole.rs bhimpl_jit_merge_point` decodes the same
            // six typed lists). Walking JitCode, the walker reads them
            // directly rather than recomputing live args via liveness.
            //
            // Operand layout `cIRFIRF`: jdindex (`c`, 1 signed byte) +
            // greens (`I`=gi, `R`=gr, `F`=gf) + reds (`I`=ri, `R`=rr,
            // `F`=rf). pyre's portal jitdriver greens =
            // `[next_instr, is_being_profiled, pycode]` (`eval.rs`),
            // so gi[0]=next_instr (the Python pc) and gr[0]=pycode
            // (PyCode). The green key is derivable from the op's own
            // greens, and `next_instr` is the SAME Python-pc coordinate the
            // trace-start seed uses (`trace.rs add_merge_point(make_green_key(
            // w_code, start_pc))`) — no jitcode-pc/python-pc mismatch.
            let jdindex = if op.key == "jit_merge_point/iIRFIRF" {
                let jd_opref = read_int_reg(code, op, 0, ctx)?;
                match ctx.trace_ctx.concrete_of_opref(jd_opref) {
                    Some(Value::Int(value)) => value as usize,
                    _ => return Err(DispatchError::LoopHeaderJdIndexUnresolved { pc: op.pc }),
                }
            } else {
                code[op.pc + 1] as i8 as usize
            };
            let mut off = 1usize; // skip the jdindex operand
            let (gi, n) = read_int_var_list(code, op, off, ctx)?;
            off += n;
            let (gr, n) = read_ref_var_list(code, op, off, ctx)?;
            off += n;
            let (_gf, n) = read_float_var_list(code, op, off, ctx)?;
            off += n;
            let (ri, n) = read_int_var_list(code, op, off, ctx)?;
            off += n;
            let (rr, n) = read_ref_var_list(code, op, off, ctx)?;
            off += n;
            let (rf, _n) = read_float_var_list(code, op, off, ctx)?;

            // Green key from (pycode, next_instr) = (gr[0], gi[0]) concretes.
            let (Some(&pc_green), Some(&code_green)) = (gi.first(), gr.first()) else {
                return Err(DispatchError::JitMergePointGreenKeyUnresolved { pc: op.pc });
            };
            let next_instr = match ctx.trace_ctx.concrete_of_opref(pc_green) {
                Some(Value::Int(v)) => v as usize,
                _ => return Err(DispatchError::JitMergePointGreenKeyUnresolved { pc: op.pc }),
            };
            // An inlined callee's own loop
            // header routes to a `CALL_ASSEMBLER` into its already-compiled loop
            // token EVEN WHEN its pycode green resolves.  nbody's `advance` has a
            // const-Ref code_green (resolves) plus an existing loop token, so the
            // recovery in the `code_green unresolved` arm below never fires and
            // the walk falls through to the normal loop-crossing path — which
            // closes a degenerate module-loop iteration at the inner header or
            // walks into the callee body and aborts
            // `LoopBearingCalleeInlineUnsupported`.  Firing here (only inside a
            // sub-walk — the framestack is non-empty — and only when a token
            // exists) routes the inlined loop-bearing callee to its own compiled
            // loop, the trait-parity `LoopTargetDescr`/`CALL_ASSEMBLER` shape.
            //
            // "An inlined callee's own loop" is whose loop this header is, which
            // a non-empty framestack does not establish: a bridge's outer-frame
            // continuation runs the trace ROOT forward (with a frame still on the
            // stack) and reaches the very loop it exited, so the framestack proxy
            // matches there too and routes a bridge that must close with a JUMP
            // back into that loop (`CloseLoop` -> `CloseLoopWithArgs`) into a
            // CALL_ASSEMBLER request no caller consumes, aborting it
            // (`OuterNonTerminate`). Discriminate on the header's frame vs the
            // trace root: same code = the root's own loop, so fall through to the
            // normal loop-crossing path.
            let fbw_root_code = {
                let sym_ptr = ctx.fbw_mode.snapshot_sym;
                if sym_ptr.is_null() {
                    None
                } else {
                    unsafe {
                        let sym = &*sym_ptr;
                        if sym.jitcode().is_null() {
                            None
                        } else {
                            let jc = &*sym.jitcode();
                            let raw = jc.payload.code_ptr;
                            if raw.is_null() {
                                None
                            } else {
                                Some(pyre_interpreter::live_code_wrapper(raw as *const ()))
                            }
                        }
                    }
                }
            };
            let callee_code = ctx
                .session
                .borrow()
                .framestack
                .last()
                .map(|frame| frame.w_code)
                .filter(|&cc| {
                    fbw_root_code.is_none_or(|root| root as *const () != cc as *const ())
                });
            if let Some(callee_code) = callee_code {
                let callee_key =
                    crate::driver::make_green_key(callee_code as *const (), next_instr);
                let (driver, _) = crate::driver::driver_pair();
                let greenboxes = [
                    Value::Int(next_instr as i64),
                    Value::Int(0),
                    Value::Ref(majit_ir::GcRef(callee_code)),
                ];
                let red_types = [Type::Ref, Type::Ref];
                if let Some(token) = driver.get_or_make_portal_assembler_token_arc(
                    callee_key,
                    &greenboxes,
                    &red_types,
                ) {
                    return Ok((
                        DispatchOutcome::SubLoopCalleeCallAssembler {
                            token,
                            target_pc: next_instr,
                        },
                        op.next_pc,
                    ));
                }
            }
            let code_ptr = match ctx.trace_ctx.concrete_of_opref(code_green) {
                Some(Value::Ref(gcref)) if gcref.0 != 0 => gcref.0 as *const (),
                _ => {
                    // Inside a multi-frame inline sub-walk the callee's own
                    // `jit_merge_point` (its loop header) carries a pycode green
                    // with no live Ref shadow, so this resolution fails and the
                    // enclosing trace would decline. Recover
                    // the callee code from the FBW inline stack; if a compiled
                    // loop token already exists for (callee_code, next_instr),
                    // surface a recursive CALL_ASSEMBLER request to the caller's
                    // inline return site (mirror `opimpl_recursive_call_
                    // assembler`, metainterp.rs).
                    let callee_code = ctx
                        .session
                        .borrow()
                        .framestack
                        .last()
                        .map(|frame| frame.w_code);
                    if let Some(callee_code) = callee_code {
                        let callee_key =
                            crate::driver::make_green_key(callee_code as *const (), next_instr);
                        let (driver, _) = crate::driver::driver_pair();
                        let greenboxes = [
                            Value::Int(next_instr as i64),
                            Value::Int(0),
                            Value::Ref(majit_ir::GcRef(callee_code)),
                        ];
                        let red_types = [Type::Ref, Type::Ref];
                        if let Some(token) = driver.get_or_make_portal_assembler_token_arc(
                            callee_key,
                            &greenboxes,
                            &red_types,
                        ) {
                            return Ok((
                                DispatchOutcome::SubLoopCalleeCallAssembler {
                                    token,
                                    target_pc: next_instr,
                                },
                                op.next_pc,
                            ));
                        }
                    }
                    top_level_live_code(ctx)
                        .ok_or(DispatchError::JitMergePointGreenKeyUnresolved { pc: op.pc })?
                }
            };
            let key = crate::driver::make_green_key(code_ptr, next_instr);

            // pyjitpl.py: a jit_merge_point is a loop CROSSING
            // only when a `loop_header` op (the lowered `can_enter_jit`
            // at a backward-jump site) stamped the per-trace flag just
            // before — arrival by straight-line fall-through (e.g. the
            // first check of a fresh inner `while`) records nothing and
            // walks on. Without this gate the walker closes a degenerate
            // "outer-iteration" loop at the inner header (exhausted-check
            // → outer increment → new-iterator → re-arrival), occupying
            // the inner loop's green key so its specialized retrace can
            // never compile. Mirrors majit's `BC_JIT_MERGE_POINT` auto
            // loop-header + close protocol (`pyjitpl/dispatch.rs`).
            // The `c` form's jdindex is the leading literal byte; the `i`
            // form was resolved through its Int-bank operand above.
            // pyjitpl.py:2610-2626: a guard's resume coordinate
            // (`resumepc=orgpc`) lies INSIDE the guarded opcode's
            // implementation, past the dispatch-top `jit_merge_point`, so an
            // RPython MIFrame resumed from a guard never re-crosses the
            // loop-header merge point at position zero. The walker resumes a
            // bridge at the opcode BOUNDARY, so its first crossing at the
            // resume coordinate is the same op it is resuming INTO — not a
            // loop crossing. Skip exactly once. `take()` clears on the first
            // crossing regardless of pc, so a mid-body-resume bridge whose
            // first crossing is a DIFFERENT header is unaffected.
            if ctx.is_top_level
                && ctx.trace_ctx.seen_loop_header_for_jdindex < 0
                && ctx.fbw_mode.bridge_entry_merge_pc.take() == Some(next_instr)
            {
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
            if ctx.trace_ctx.seen_loop_header_for_jdindex < 0 {
                // pyjitpl.py `if not any_operation: return`.
                if ctx.trace_ctx.num_ops() == 0 {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // pyjitpl.py `if not jitdriver_sd.no_loop_header:`
                let no_loop_header = ctx
                    .trace_ctx
                    .metainterp_sd()
                    .jitdrivers_sd
                    .get(jdindex)
                    .map(|jd| jd.no_loop_header)
                    .unwrap_or(false);
                if !no_loop_header {
                    // pyjitpl.py `if self.metainterp.portal_call_depth:
                    // return` — nested portal call waits for an explicit
                    // loop_header.
                    let depth_zero = ctx
                        .trace_ctx
                        .portal_call_depth_fn
                        .as_ref()
                        .map(|f| f() == 0)
                        .unwrap_or(false);
                    if !depth_zero || !ctx.is_top_level {
                        return Ok((DispatchOutcome::Continue, op.next_pc));
                    }
                    // pyjitpl.py: fall-through arrival counts as
                    // an automatic loop_header only when compiled targets
                    // already exist for the crossed green key.
                    let has_targets = ctx
                        .trace_ctx
                        .has_compiled_targets_fn
                        .as_ref()
                        .map(|f| f(key))
                        .unwrap_or(false);
                    if !has_targets {
                        return Ok((DispatchOutcome::Continue, op.next_pc));
                    }
                }
                // pyjitpl.py: automatically add a loop_header.
                ctx.trace_ctx.seen_loop_header_for_jdindex = jdindex as i32;
            }
            // pyjitpl.py.
            assert!(
                ctx.trace_ctx.seen_loop_header_for_jdindex == jdindex as i32,
                "found a loop_header for a JitDriver that does not match \
                 the following jit_merge_point's"
            );
            ctx.trace_ctx.seen_loop_header_for_jdindex = -1;

            // pyjitpl.py self.heapcache.reset()
            ctx.trace_ctx.heap_cache_mut().reset();

            // `close_loop_args_at` (trace_opcode.rs) runs the merge-point
            // vable `last_instr` sync BEFORE building the jump args: the
            // scalar is overridden to `merge_pc - 1` (a resume into the
            // target loop must re-enter at the header opcode). The walker
            // must do the same before `append_virtualizable_boxes` below —
            // otherwise the compile_trace arm's JUMP into the existing loop
            // carries the LAST GUARD's published `last_instr` (e.g. 104
            // instead of header-1=86 on fannkuch), resuming the interpreter
            // at the wrong bytecode (fannkuch permutation state never
            // reaches its exit condition → non-crashing infinite loop).
            sync_intermediate_merge_point_last_instr(ctx.trace_ctx, next_instr);

            // Reds = the live loop args, in bytecode bank order
            // [int.., ref.., float..]. For pyre's portal jitdriver the
            // reds are `[frame, ec]` (both Ref → rr), matching the
            // reds-only LABEL inputargs after
            // `patch_new_loop_to_load_virtualizable_fields`.
            let mut live_args: Vec<OpRef> = Vec::with_capacity(ri.len() + rr.len() + rf.len());
            live_args.extend(ri.iter().copied());
            live_args.extend(rr.iter().copied());
            live_args.extend(rf.iter().copied());
            // The loop-close path (`run_perfn_walk` CloseLoop post-processing,
            // trace.rs) rebuilds the jump args via `close_loop_args_at`, which
            // sources the reds from `sym.frame` / `sym.execution_context` — not
            // from the walk register file read above (the register slot may
            // hold a const-folded alias of the same value). The merge-point
            // registration must use the SAME box identities: `history.cut`
            // (cross-loop cut, pyjitpl.py) takes the registered
            // green_boxes as the new loop's inputargs, and a close-side red
            // absent from them escapes into an extra appended inputarg —
            // producing an entry layout `patch_new_loop_to_load_virtualizable_
            // fields` cannot reduce, so every interpreter entry aborts
            // (`extend_compiled_live_values` count mismatch).
            {
                let sym_ptr = ctx.fbw_mode.snapshot_sym;
                if !sym_ptr.is_null() && ri.is_empty() && rf.is_empty() && rr.len() == 2 {
                    let sym = unsafe { &*sym_ptr };
                    live_args[0] = sym.frame();
                    if !sym.execution_context().is_none() {
                        live_args[1] = sym.execution_context();
                    }
                }
            }
            live_args = append_virtualizable_boxes(ctx.trace_ctx, live_args);

            // pyjitpl.py remove_consts_and_duplicates over the
            // reds + virtualizable_boxes[:-1]: every live arg must be a
            // distinct non-const box before it is registered as a merge
            // point or matched for loop closure. A const or duplicate is
            // replaced with a fresh `same_as` op, written back into the
            // virtualizable shadow so subsequent reads/snapshots use the
            // new identity (the in-place boxes[i] mutation upstream).
            // Without this, the registered green_boxes carry the SAME
            // OpRef at two positions and `cut_trace_from`'s remap maps
            // both to the LAST position — every body/snapshot reference
            // to the duplicated box then reads the wrong loop-carried
            // slot (e.g. a local aliased by a dead stack entry reads the
            // entry's NULL at every deopt).
            {
                use std::collections::HashSet;
                let mut duplicates: HashSet<OpRef> = HashSet::new();
                for i in 0..live_args.len() {
                    let opref = live_args[i];
                    if opref.is_constant() || !duplicates.insert(opref) {
                        let tp = ctx
                            .trace_ctx
                            .get_opref_type(opref)
                            .unwrap_or(majit_ir::Type::Ref);
                        let same_as_op = majit_ir::OpCode::same_as_for_type(tp);
                        let new_opref = ctx.trace_ctx.record_op(same_as_op, &[opref]);
                        live_args[i] = new_opref;
                        // live_args = [frame, ec, vable_boxes[0..len-1]];
                        // frame/ec are first-occurrence runtime boxes and
                        // never wrap, so only the vable payload mirrors
                        // back (virtualizable_boxes[i] = op upstream).
                        if i >= 2 {
                            ctx.trace_ctx.set_virtualizable_box_at(i - 2, new_opref);
                        }
                    }
                }
            }

            if std::env::var("PYRE_DIAG_51C").is_ok() {
                eprintln!(
                    "[51c-redclose] pc={} ri_regs={:?} rr_regs={:?} rf_regs={:?}",
                    op.pc, ri, rr, rf
                );
                for (idx, &arg) in live_args.iter().enumerate() {
                    eprintln!(
                        "[51c-redclose]   live_arg[{idx}] {arg:?} concrete={:?}",
                        ctx.trace_ctx.concrete_of_opref(arg)
                    );
                }
            }

            // pyjitpl.py compile_trace attempt (retired trait
            // mirror `close_loop_args`): when the
            // crossed green key already has compiled targets and no
            // retrace is in progress, close the trace-so-far as a bridge
            // (guard origin) / entry bridge (interp origin,
            // `compile_trace_from_interp`) ending in a JUMP into the
            // existing loop.  Without this a func-entry trace that walks
            // the prologue and reaches the already-hot inner loop header
            // falls through to compile_loop → has_compiled_targets →
            // SwitchToBlackhole(ABORT_BAD_LOOP), so every portal call
            // re-runs the prologue interpreted and re-aborts a trace —
            // overhead scaling with call count.
            if ctx.is_top_level && ctx.is_authoritative_executor {
                let (driver, _) = crate::driver::driver_pair();
                let has_partial = driver.meta_interp().partial_trace().is_some();
                let bridge_origin = driver
                    .meta_interp()
                    .bridge_info()
                    .map(|b| (b.trace_id, b.fail_index));
                let has_targets = driver.meta_interp().has_compiled_targets(key);
                if !has_partial && has_targets {
                    let outcome = match bridge_origin {
                        // Guard-origin: existing bridge path.
                        Some(_) => {
                            driver
                                .meta_interp_mut()
                                .compile_trace(key, &live_args, bridge_origin)
                        }
                        // pyjitpl.py interp-origin: a
                        // function-entry trace (ResumeFromInterpDescr)
                        // closes as an entry bridge jumping into the
                        // already-compiled hot loop (compile.py);
                        // a trace rooted at a *loop header* falls back to
                        // the plain bridge shape.
                        None => match driver.compile_trace_entry_data() {
                            Some((original_green_key, mut entry_meta)) => {
                                // `compile_trace_entry_data` clones the active
                                // trace metadata, whose `namespace_dependent` is
                                // only finalized by `finish_trace_namespace_dependency`
                                // after the walk returns. An entry bridge is
                                // compiled mid-walk, before that finalize, so a
                                // trace that has already read a module global
                                // would otherwise install the bridge with a stale
                                // `namespace_dependent = false` and let it be
                                // re-entered after later namespace growth. Fold in
                                // the live per-trace flag so the bridge keeps the
                                // conservative namespace gate.
                                entry_meta.namespace_dependent |= ctx.trace_ctx.reads_module_global;
                                driver.meta_interp_mut().compile_trace_from_interp(
                                    key,
                                    &live_args,
                                    original_green_key,
                                    entry_meta,
                                )
                            }
                            None => driver
                                .meta_interp_mut()
                                .compile_trace(key, &live_args, None),
                        },
                    };
                    if matches!(outcome, majit_metainterp::CompileOutcome::Compiled { .. }) {
                        if majit_metainterp::majit_log_enabled() {
                            eprintln!(
                                "[jit][walker-reached-loop-header] compile_trace success: \
                                 key={} pc={} bridge={:?}",
                                key, next_instr, bridge_origin
                            );
                        }
                        // pyjitpl.py raise_if_successful() — the
                        // successful compile_trace ends tracing; surface
                        // the dedicated outcome so the driver maps it to
                        // `TraceAction::CompileTrace` (no further compile
                        // or abort on this session).
                        driver.note_compile_trace_success();
                        return Ok((
                            DispatchOutcome::CompileTracePending {
                                loop_header_pc: next_instr,
                            },
                            op.next_pc,
                        ));
                    }
                    // The jump did not take (`compile.compile_trace` returns
                    // None when none of the existing loop tokens match). Fall
                    // through to the merge-point scan below, exactly as
                    // `reached_loop_header` does after its own
                    // `self.compile_trace(...)` call returns (pyjitpl.py:3003-3018):
                    // the scan closes at the first same-greenkey merge point, and
                    // `compile_loop` gives that trace up at its own
                    // `has_compiled_targets` (pyjitpl.py:3185-3189); a first visit
                    // registers a merge point (pyjitpl.py:3057-3059) and keeps
                    // tracing.
                    majit_metainterp::mc_diag_bump(27);
                }
            }

            // pyjitpl.py: a matching merge point (same green key
            // + red-bank shape) closes the loop; first visit registers and
            // continues to unroll.
            if ctx
                .trace_ctx
                .has_merge_point_with_shape_assert(key, live_args.len())
            {
                // The matched merge point need not be the one tracing started
                // from: `reached_loop_header` scans every registered merge
                // point in reverse and closes at the first same_greenkey hit,
                // whichever loop that is, and `compile_loop` re-derives the
                // green key from the merge point it closed at.
                let loop_header_marker_jit_pc = {
                    let sym_ptr = ctx.fbw_mode.snapshot_sym;
                    if sym_ptr.is_null() {
                        None
                    } else {
                        let sym = unsafe { &*sym_ptr };
                        if sym.jitcode().is_null() {
                            None
                        } else {
                            unsafe {
                                (&*sym.jitcode())
                                    .payload
                                    .resume_marker_for_jitcode_pc(op.pc)
                            }
                        }
                    }
                };
                Ok((
                    DispatchOutcome::CloseLoop {
                        jump_args: live_args,
                        loop_header_pc: next_instr,
                        loop_header_marker_jit_pc,
                    },
                    op.next_pc,
                ))
            } else {
                // This merge point registers (does not close) — no already
                // registered merge point carries this green key and red-bank
                // shape, so this is the first arrival at the loop whose header
                // is `next_instr`.  The intermediate merge-point
                // vable→heap writeback (#62 / #67-remaining) already ran
                // above, before the live-args build, mirroring
                // `close_loop_args_at`'s ordering.
                let green_boxes: Vec<majit_metainterp::GreenBox> = live_args
                    .iter()
                    .map(|opref| {
                        let ty = ctx.trace_ctx.get_opref_type(*opref).unwrap_or_else(|| {
                            panic!(
                                "jit_merge_point live arg {opref:?} has no type in \
                                 OptContext; RPython Box always carries its type"
                            )
                        });
                        majit_metainterp::GreenBox::new(*opref, ty)
                    })
                    .collect();
                ctx.trace_ctx.add_merge_point(key, green_boxes, next_instr);
                Ok((DispatchOutcome::Continue, op.next_pc))
            }
        }
        other => Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: other,
        }),
    }
}

#[cfg(test)]
mod tests;
