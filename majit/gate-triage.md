# majit environment gate triage

This catalog must contain exactly the live `MAJIT_*` environment gates read by workspace Rust and Python sources. `pyre/pyrex/tests/gate_triage_complete.rs` checks both directions.

Each entry records its reader, purpose, and retirement condition. `UNRECORDED` marks information that must not be guessed.

## Live gates

### `MAJIT_BH_DEBUG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `bh_debug_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BH_NULL_ARG`

- Read sites: 1 — `majit/majit-metainterp/src/blackhole.rs`
- Accessor: `bh_null_arg_report()`
- What it does: `MAJIT_BH_NULL_ARG`: report a null ref argument about to be handed to a residual call, with the jitcode coordinate, before the callee can dereference it.  Some ABIs pass a legitimate null sentinel (e.g. the CallFn `null_or_self` slot), so this reports rather than aborts.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BRIDGE_DEBUG`

- Read sites: 5 — `majit/majit-macros/src/jit_interp/codegen_state.rs`, `majit/majit-metainterp/src/lib.rs`
- Accessor: `ref_identity_slots_end()`; also read inline in `setup_bridge_sym()` and `bridge_debug_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BRIDGE_DIAG`

- Read sites: 2 — `majit/majit-macros/src/jit_interp/codegen_state.rs`, `majit/majit-metainterp/src/resume_box_reader.rs`
- Accessor: `setup_bridge_sym()`; also read inline in `replay_pending_fields()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BRIDGE_FUEL_LOG`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: read inline in `bridge_fuel_take()`
- What it does: Prints the sequence number of each bridge compilation that consumes bridge fuel.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_CLOSEDBG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `closedbg_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_CL_GCSTORE_LOG`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: read inline in `do_compile()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_COVERAGE_AUDIT`

- Read sites: 1 — `majit/majit-translate/src/codewriter/assembler.rs`
- Accessor: `assemble_with_callcontrol()`
- What it does: Lists every SSA variable without a register-allocation color, grouped by graph. This complements the panic-on-first-gap mode.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_COVERAGE_PANIC`

- Read sites: 1 — `majit/majit-translate/src/codewriter/assembler.rs`
- Accessor: read inline in `assemble_with_callcontrol()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DEBUG_DECLARES`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: read inline in `do_compile()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DIAG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `diag_enabled()`
- What it does: Enables the metainterpreter's diagnostic counters. Bridge-close counters distinguish declined compilation attempts from closes for which compilation was not attempted.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP`

- Read sites: 1 — `majit/majit-backend-dynasm/src/lib.rs`
- Accessor: `majit_dump_enabled()`
- What it does: Whether `MAJIT_DUMP` is set, cached at first access.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_BYTECODE`

- Read sites: 1 — `pyre/pyre-jit/src/eval.rs`
- Accessor: `dump_bytecode_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_CLIF`

- Read sites: 2 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: read inline in `do_compile()`, at both sites
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_LIVENESS`

- Read sites: 1 — `majit/majit-macros/src/jit_interp/jitcode_lower/liveness.rs`
- Accessor: `maybe_dump_liveness()`
- What it does: Print per-marker live sets to stderr when `MAJIT_DUMP_LIVENESS` is set in the proc-macro build environment. `label` is the lowerer scope being dumped (e.g. helper name) so concurrent expansions are distinguishable.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_SSAREPR`

- Read sites: 1 — `pyre/pyre-jit/src/jit/assembler.rs`
- Accessor: `dump_assembled_ssarepr()`
- What it does: Print the assembled instruction stream, byte position first, for graphs whose name matches `MAJIT_DUMP_SSAREPR`. A blackhole failure reports a raw `(jitcode, position)` pair; without the stream there is no way back from that byte offset to the op that wrote it or to the register operands it reads.  The env lookup is cached because `try_assemble` runs per graph on the tracing path.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_FAILVALS`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `failvals_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_FIELD_MINT_TRACE`

- Read sites: 2 — `majit/majit-ir/src/descr.rs`, `pyre/pyre-jit-trace/build.rs`
- Accessor: `field_mint_trace_enabled()`; the build script also declares it as a rerun input and bypasses its code-generation cache while enabled
- What it does: Setting it to `1` prints descriptor-mint disagreements and keeps the analyzer live so those diagnostics cannot be hidden by a restored artifact cache. Unset is inert.
- Retirement condition: Remove when field and size descriptor identity no longer has fallback or disagreement paths to diagnose.

### `MAJIT_FIELD_POS_UNRESOLVED`

- Read sites: 1 — `majit/majit-ir/src/descr.rs`
- Accessor: `field_position_unresolved_limit()`
- What it does: Dumps unresolved field-position rows. A positive integer limits the number of rows; `1` or a non-numeric value dumps the whole table; unset disables the dump.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_BH_PROBE`

- Read sites: 1 — `majit/majit-gc/src/lib.rs`
- Accessor: `bh_probe_enabled()`
- What it does: Whether the blackhole-object probe is enabled.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_DRAIN_CENSUS`

- Read sites: 1 — `majit/majit-gc/src/lib.rs`
- Accessor: `drain_census_dump_interval()`
- What it does: Set `MAJIT_GC_DRAIN_CENSUS` to a positive integer to also dump the running summary every that many collections. The end-of-run summary is unreachable for the runs this census is most needed on — a collection storm that has to be killed rather than waited out — so those need the periodic line.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_LIFETIME_LOG`

- Read sites: 1 — `majit/majit-gc/src/lib.rs`
- Accessor: `gc_lifetime_log_enabled()`
- What it does: `MAJIT_GC_LIFETIME_LOG` — trace remembered-set adds and old-gen frees. Read once.  The gate sits in the write barrier and the old-gen sweep, and `std::env::var_os` takes the environment lock and scans it linearly on every call, so asking per event costs whether or not the variable is set.  Same shape as `majit_metainterp::majit_log_enabled`.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_NURSERY_POISON`

- Read sites: 2 — `majit/majit-gc/src/nursery.rs`, `majit/majit-gc/src/oldgen.rs`
- Accessor: `new()`
- What it does: **UNRECORDED** — no doc comment describes the gate. Read off the sites, not quoted: the two reads initialise `poison_on_reset` (`nursery.rs`) and `poison_on_alloc` (`oldgen.rs`).
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_STRESS`

- Read sites: 1 — `majit/majit-gc/src/collector.rs`
- Accessor: read inline in `with_config()`, behind `#[cfg(feature = "gc_stress")]`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GUARDLOG`

- Read sites: 2 — `majit/majit-metainterp/src/jitdriver.rs`, `majit/majit-metainterp/src/pyjitpl.rs`
- Accessor: `guardlog_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GUARD_CENSUS`

- Read sites: 2 — `majit/majit-metainterp/src/lib.rs`, `pyre/pyrex/src/lib.rs`
- Accessor: `guard_census_enabled()`; also read inline in `maybe_print_jit_stats()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_HEAPDBG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `heapdbg_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_J2PLAN_LOG`

- Read sites: 2 — `majit/majit-backend-dynasm/src/aarch64/assembler.rs`, `majit/majit-backend-dynasm/src/lib.rs`
- Accessor: `majit_j2plan_log_enabled()`; also read inline in `_assemble()`
- What it does: Whether `MAJIT_J2PLAN_LOG` is set, cached at first access.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LEAF3_PROV`

- Read sites: 1 — `majit/majit-metainterp/src/resume.rs`
- Accessor: `leaf3_prov_enabled()`
- What it does: Emits the resume tag, value, and null status of the virtualizable identity slot consumed by `consume_vable_info()`.
- Retirement condition: Remove after the unseeded-snapshot route into `_number_boxes()` is rejected or proven unreachable.

### `MAJIT_LLBC_EXTRACTION`

- Read sites: 2 — `pyre/pyre-jit-trace/build.rs`
- Accessor: `main()` — one `cargo::rerun-if-env-changed=` declaration and one `env::var_os` read, on adjacent lines
- What it does: **UNRECORDED** — no doc comment describes the gate; the one above `main()` describes the build script. Read off the site, not quoted: `=1` calls `emit_llbc_extraction_placeholders()` and returns early, so the extraction does not run.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LOG`

- Read sites: 19 — `majit/majit-backend-cranelift/src/compiler.rs`, `majit/majit-backend-dynasm/src/lib.rs`, `majit/majit-gc/src/lib.rs`, `majit/majit-gc/src/rewrite.rs`, `majit/majit-ir/src/debug.rs`, `majit/majit-metainterp/src/lib.rs`, `majit/majit-trace/src/logger.rs`
- Accessor: no single accessor — `majit_log_enabled()` is defined once per crate and several sites read the environment inline; relocate with `rg -w MAJIT_LOG <file>`.
- What it does: Whether `MAJIT_LOG` is set, cached at first access.  Mirrors PyPy's `PYPYLOG` env-var check (`rpython/rlib/debug.py:31-38`).
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LOG_JTET`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `log_jtet_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LOG_OPT`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `log_opt_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MACRO_DEBUG`

- Read sites: 8 — `majit/majit-macros/src/jit_interp/jitcode_lower/dispatch.rs`, `majit/majit-macros/src/jit_interp/jitcode_lower/lower_stmt.rs`
- Accessor: `try_inline_dispatch_arm()`; also read inline in `lower_dispatch_chain()`, `lower_return_stmt()` and `lower_stmt_fallback()`
- What it does: **UNRECORDED** — no doc comment describes the gate. Read off the sites, not quoted: every read guards an `eprintln!` and nothing else. The prose around them documents the lowering decisions being printed, not what the gate is for.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MAX_BRIDGES`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `bridge_fuel_take()`
- What it does: Allows the first N bridge compilations, then suppresses further bridges. Fuel is consumed only after the other `should_bridge` conditions pass.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MINT_INDEX_CENSUS`

- Read sites: 1 — `pyre/pyre-jit-trace/build.rs`
- Accessor: read inline in `real_main()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MPTRACE`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `mptrace_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_NO_BRIDGE`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `no_bridge_enabled()`
- What it does: `MAJIT_NO_BRIDGE`: suppress bridge recording so every guard failure resumes through the blackhole.  Public because a frontend that owns its own guard-failure entry point has to honour it there too — gating only the jitdriver-internal paths leaves the variable set but inert, which reads as "bridges are off" while they keep recording.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_OPREF_VARIANT_AUDIT`

- Read sites: 1 — `majit/majit-ir/src/opref_audit.rs`
- Accessor: read inline in `resolve_mode()`
- What it does: available only in a `jit-audits` build; `=1` reports and `=abort` panics on the first collision of two `OpRef` variants on one `raw()` key. The ordinary build contains neither the state nor its call sites. In an audit build with the environment gate off, each site performs one thread-local read and returns. State and mode are thread-local so two tests in parallel cannot read each other's collisions or silence one another.
- Retirement condition: Remove when the two `OpRef` namespaces are structurally unable to collide.

### `MAJIT_OPTRACE`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `optrace_enabled()`
- What it does: Per-op trace of `run_to_end`'s dispatch loop (frame depth, pc, raw opcode). Diagnostic for pinpointing the op that faults a hardware-signal crash (SIGBUS/SIGSEGV) which `catch_unwind` cannot capture.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_PCSEQ`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `pcseq_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_PORTAL_INLINE`

- Read sites: 1 — `majit/majit-metainterp/src/pyjitpl/dispatch.rs`
- Accessor: `portal_inline_experiment_enabled()`
- What it does: Enables the experimental recursive-portal inline re-entry path. Unset keeps the clean-abort fallback when `portal_jitcode` is absent.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_PROBE_LIVENESS`

- Read sites: 1 — `pyre/pyre-jit/src/call_jit.rs`
- Accessor: `majit_probe_liveness_enabled()`
- What it does: Whether `MAJIT_PROBE_LIVENESS` is set, cached at first access.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_REG_WRITE_AUDIT`

- Read sites: 1 — `majit/majit-ir/src/reg_write_audit.rs`
- Accessor: read inline in `resolve()`, cached per thread by `enabled()`
- What it does: available only in a `jit-audits` build; records the source location of the latest write to each integer register, so a later read can report its writer. The ordinary build contains neither the state nor its call sites. Diagnostic state is thread-local to keep parallel traces independent.
- Default polarity: **OFF**; unset, empty, and `0` disable it.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_SMALLIR`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `smallir_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_SPDIAG`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `spdiag_enabled()`
- What it does: Enables stack-position diagnostics on hot back-edge and guard-failure paths. The value is cached to avoid repeated environment reads.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STALL_WINDOW`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `stall_window()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STATS`

- Read sites: 3 — `majit/majit-trace/src/logger.rs`, `pyre/pyre-wasm-runner/src/main.rs`, `pyre/pyrex/src/lib.rs`
- Accessor: `stats_enabled()`; also read inline in `run()` and `maybe_print_jit_stats()`
- What it does: Whether JIT statistics collection is enabled. Checks MAJIT_STATS=1 or MAJIT_LOG=1.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STEP_LIMIT`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `step_limit()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STRICT`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `jit_strict_mode()`
- What it does: Strict JIT mode: a non-`InvalidLoop` panic during compilation is a bug and must fail loudly rather than silently degrade to the interpreter and mask the bug behind correct output. Enabled in debug builds (`cargo test`) and whenever `MAJIT_STRICT` is set (release benches / CI); off in plain release so production keeps graceful degradation. Cached like `majit_log_enabled`.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STRUCT_LAYOUT_CENSUS`

- Read sites: 2 — `majit/majit-translate/src/lib.rs`, `pyre/pyre-jit-trace/build.rs`
- Accessor: `struct_layout_census_enabled()`; the build script also declares it as a rerun input and bypasses its code-generation cache while enabled
- What it does: Setting it to `1` reports structure IDs that resolve to multiple spellings or conflicting concrete layouts during translation. Unset is inert.
- Retirement condition: Remove when one structure ID cannot collect conflicting layouts by construction.

### `MAJIT_TLDBG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `tldbg_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_VERIFY`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: `majit_verify_enabled()`
- What it does: Whether `MAJIT_VERIFY` is set, cached at first access.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_X2_PROBE`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: `drop()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.
