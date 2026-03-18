# majit vs rpython Line-by-Line Comparison Report

## 요약: 파일별 구현 완성도

| 파일 | 완성도 | 주요 누락 |
|------|--------|-----------|
| **resoperation.rs** | 99% | 완벽 일치 |
| **intutils.rs** | 95% | make_guards, getnullness |
| **trace.rs** | 95% | 완료 |
| **heapcache.py 기능 (heap.rs에 통합)** | 95% | 별도 Rust 파일 없음. `heap.rs`에 array/quasi-immut/arraylen/loopinvariant cache 통합, 남은: explicit version system |
| **pure.rs** | 90% | ~~postponed_op, constant_fold, ovf, GUARD_NO_EXCEPTION, short_preamble~~ 모두 추가됨. 남은: COND_CALL_VALUE |
| **optimizer.rs** | 90% | ~~call_pure_results, speculative_operation~~ 추가됨. 남은: resumedata_memo, guard sharing |
| **simplify.rs** | 90% | ~~GUARD_FUTURE_CONDITION~~ 스텁 추가됨 |
| **descr.rs** | 85% | bitstring optimization, CallInfoCollection |
| **intbounds.rs** | 85% | ~~backward propagation~~ 추가됨. 남은: STRGETITEM/GETFIELD bounds |
| **quasi_immut.rs** | 80% | CPU 통합, 디버그/통계 |
| **virtualstate.rs** | 80% | ~~lenbound, make_inputargs~~ 추가됨. 남은: GenerateGuardState |
| **shortpreamble.rs** | 75% | ~~ExtendedShortPreambleBuilder~~ 추가됨. 남은: CompoundOp, ShortInputArg |
| **virtualizable.rs** | 80% | optimizer bridge의 descriptor/effect parity |
| **info.rs** | 75% | ~~visitor_walk_recursive, force_at_end_of_preamble~~ 추가됨. 남은: force_box |
| **rewrite.rs** | 70% | 남은: arraycopy/move, INT_PY_DIV/MOD, GUARD_SUBCLASS/IS_OBJECT/GC_TYPE |
| **graph.rs / front/ast.rs** | 70% | 남은: match/switch lowering, effect info |
| **warmstate.rs** | 70% | ~~vectorize params~~ 추가됨. 남은: confirm_enter_jit, get_location |
| **virtualize.rs** | 65% | 남은: COND_CALL, JIT_FORCE_VIRTUAL, GETARRAYITEM_RAW |
| **opencoder.rs** | 65% | 남은: constant pooling 최적화 |
| **vstring.rs** | 60% | ~~_int_add/_int_sub, postprocess_strlen~~ 추가됨. 남은: copy_str_content, initialize_forced_string |
| **blackhole.rs** | 60% | goto_if_not, raise/reraise, 문자열 ops |
| **virtual_ref.rs** | 60% | ~~continue_tracing~~ 추가됨. 남은: graph transformation, is_virtual_ref |
| **flatten.rs / pipeline.rs** | 60% | 남은: liveness, regalloc |
| **majit-analyze lib.rs** | 65% | legacy `AnalysisResult` 병행, graph emission 미사용 |
| **counter.rs** | 55% | ~~compute_threshold~~ 추가됨. 남은: JitCell chain, fetch_next_hash |
| **bridgeopt.rs** | 55% | 완료 |
| **annotate.rs / rtype.rs** | 55% | 남은: repr/descriptor specialization |
| **patterns.rs** | 55% | call/field name heuristic 잔존 |
| **codewriter/codegen.rs** | 60% | `generate_from_graph()` 미통합 |
| **jit_interp mod.rs** | 65% | `state_fields` sugar와 canonical IR 공유 미완 |
| **jitdriver.rs / pyjitpl.rs** | 70% | internal vable length staging seam |
| **guard.rs** | 55% | Guard implication, transitive loop bounds |
| **heap.rs** | 55% | 남은: variable-index array, serialization, lazy_set_for_guard |
| **vector.rs** | 50% | ~~accumulation~~ 추가됨. 남은: loop unrolling, extend/combine packset |
| **resume.rs** | 45% | Reader/Decoder, 문자열 virtual info |
| **unroll.rs** | 40% | 남은: export/import state, inline_short_preamble |
| **majit-runtime jit hints** | 35% | framework-wide hint semantics |
| **pyjitpl.rs** | 다른 설계 | Python=해석 기반, Rust=컴파일 실행 기반 |
| **jitcode.rs** | 다른 설계 | Python=데이터 컨테이너, Rust=완전 해석기+빌더 |
| **codegen lib.rs** | 다른 설계 | bh_* 22개 추가됨 |

---

## 파일별 상세 비교

### 1. intbounds.rs ↔ intbounds.py

**누락된 최적화:**
- Pure argument synthesis (INT_OR/XOR → INT_ADD, INT_ADD → INT_SUB 역변환)
- INT_ADD/SUB constant inversion synthesis
- INT_LSHIFT reverse operation synthesis (→ INT_RSHIFT)
- CALL_PURE_I with oopspec dispatch (OS_INT_PY_DIV, OS_INT_PY_MOD)
- String/field access bounds (STRGETITEM [0,255], GETFIELD/GETARRAYITEM bounds)
- Guard postprocessing (propagate_bounds_backward)
- Backward propagation dispatcher

**로직 차이:**
- Overflow guard: Python은 last_emitted_operation 직접 확인, Rust는 PendingOverflowGuard 상태머신
- INT_SIGNEXT: Python은 make_equal_to, Rust는 ctx.replace_op

---

### 2. heap.rs ↔ heap.py

**누락된 최적화:**
- DICT_LOOKUP caching (연속 dict lookup 캐싱)
- Postponed operation queueing (비교/call_may_force/ovf 연산 지연 emit)
- GUARD_NO_EXCEPTION/GUARD_EXCEPTION deduplication
- Effect-info based selective cache invalidation
- Index bound-based lazy array item forcing
- Guard-only lazy set forcing (pendingfields)
- Variable index array caching
- Postprocessing hooks (GETFIELD/GETARRAYITEM 후 캐시 등록)
- Serialization/deserialization

**Rust 추가:**
- Explicit unescaped object tracking
- ARRAYCOPY/ARRAYMOVE-specific cache invalidation
- Loop-invariant call caching

---

### 3. pure.rs ↔ pure.py

**이미 올라온 부분:**
- OVF postponed op handling
- `lookup1` / `lookup2`
- `GUARD_NO_EXCEPTION` 제거 경로
- `RECORD_KNOWN_RESULT` / `extra_call_pure`
- `call_pure_positions` / short-preamble 연계
- loop-invariant call caching

**남은 격차:**
- `COND_CALL_VALUE`
- RPython의 descriptor/effectinfo identity를 그대로 쓰는 수준까지는 아직 아님

---

### 4. rewrite.rs ↔ rewrite.py

**누락된 최적화:**
- 가드 최적화 10종: GUARD_ISNULL, GUARD_IS_OBJECT, GUARD_GC_TYPE, GUARD_SUBCLASS, GUARD_NONNULL, GUARD_CLASS 등
- Loop invariant call (CALL_LOOPINVARIANT)
- COND_CALL, COND_CALL_VALUE
- 객체 포인터 비교 (_optimize_oois_ooisnot with virtual info)
- Array operation dispatch (CALL_ARRAYCOPY, CALL_ARRAYMOVE)
- INT_PY_DIV/MOD (magic number division)
- Postprocess 메커니즘 전체

**로직 차이:**
- FLOAT_MUL: Python은 -1.0 → FLOAT_NEG 변환, Rust는 1.0 identity만
- FLOAT_TRUEDIV: Python은 frexp 기반 역수 변환, Rust는 1.0만
- GUARD_VALUE: Python은 복잡한 가드 교체 로직, Rust는 단순 상수 비교

**Rust 추가:**
- 범용 constant folding helpers (try_fold_binary_int/float)
- INT_INVERT, INT_FORCE_GE_ZERO, INT_BETWEEN
- FLOAT_FLOORDIV, FLOAT_MOD

---

### 5. virtualize.rs ↔ virtualize.py

**누락된 최적화:**
- GUARD_NO_EXCEPTION, GUARD_NOT_FORCED_2 tracking
- FINISH postprocessing
- CALL_MAY_FORCE with JIT_FORCE_VIRTUAL
- COND_CALL with JIT_FORCE_VIRTUALIZABLE
- INT_ADD raw slice optimization
- GETARRAYITEM_RAW, SETARRAYITEM_RAW
- CALL_N dispatch (OS_RAW_MALLOC_VARSIZE_CHAR, OS_RAW_FREE)

**Rust 추가 (majit 고유):**
- Virtualizable frame field tracking 시스템
- JUMP augmentation with virtualizable fields
- Guard fail_args augmentation
- Record* hint opcodes
- extract_virtual_int_field helper

**보고서 누락 보충:**
- runtime 쪽 standard/nonstandard virtualizable 경로는 예전보다 훨씬 진전됨
- 그러나 translator 전체 기준으로 보면 `hint_access_directly` / `hint_fresh_virtualizable` / `hint_force_virtualizable`는 아직 public API 차원에서 no-op 성격이 남음
- 즉 runtime parity는 많이 올라왔지만, `rpython/rlib/jit.py` + `rpython/rtyper/rvirtualizable.py` + `jtransform.py` 전체 체인으로 보면 아직 incomplete
- `storage.virtualizable` user-facing seam은 제거되었고 proc-macro parser는 이제 이를 deprecated error로 거부한다
- 따라서 현재 virtualizable 격차의 중심은 더 이상 raw storage mode가 아니라, translator-wide hint semantics와 canonical metadata ownership이다

---

### 6. simplify.rs ↔ simplify.py

**현재 상태:**
- `GUARD_FUTURE_CONDITION` opcode 자체와 제거 경로는 이미 있음
- `CALL_PURE_*` / `CALL_LOOPINVARIANT_*` demotion, hint op 제거도 구현됨

**남은 격차:**
- RPython의 `notice_guard_future_condition()`과 optimizeopt ordering 전체를 그대로 재현하진 않음
- Python은 descr/effect-level 분기가 더 풍부하고, Rust는 여전히 type/opcode 중심이 강함

---

### 7. unroll.rs ↔ unroll.py

**누락 (대부분):**
- UnrollOptimizer wrapper class (optimize_preamble, optimize_peeled_loop, optimize_bridge)
- Virtual state picking, target token generation
- Short preamble inlining (fix-point loop)
- Export/import state
- Retrace limit management

**Rust:** Minimal streaming loop-peeling pass only

---

### 8. guard.rs ↔ guard.py

**다른 목적:** Python=vector optimizer (array bounds), Rust=general guard optimization
**누락:** Guard implication (numeric range analysis), transitive loop bounds
**Rust 추가:** Descriptor fusion, foldable guard filtering, truthy_values subsumption

---

### 9-10. bridgeopt.rs / shortpreamble.rs

**bridgeopt:** Python=직렬화/역직렬화 기반, Rust=정적 정수 범위 분석 기반 (다른 접근)
**shortpreamble:** `ExtendedShortPreambleBuilder`까지 추가됨. 남은: `CompoundOp`, `ShortInputArg`, Python short preamble의 전체 object model

---

### 11. vstring.rs ↔ vstring.py

**남은 격차:**
- Unicode support 전체
- Oopspec call handlers (STR_CONCAT, STR_SLICE, STR_EQUAL, STR_CMP, SHRINK_ARRAY)
- `copy_str_content` heuristic
- `initialize_forced_string`
- String forcing to constants

**이미 올라온 부분:**
- `_int_add` / `_int_sub`
- `postprocess_strlen`

---

### 12. info.rs ↔ info.py

**이미 올라온 부분:**
- `FloatConstInfo`
- `visitor_walk_recursive`
- `force_at_the_end_of_preamble`
- `make_guards`
- `copy_fields_to_const`에 해당하는 copy helper

**부분 구현 / skeleton:**
- `getstrlen` / `getstrhash` surface는 있으나 아직 실질 값 계산은 비어 있음

**남은 격차:**
- `force_box` parity
- guard tracking (`last_guard_pos`, `mark_last_guard`) 계층
- Python의 richer visitor/force protocol 전체

**Rust:** 순수 data carrier 성격이 더 강함 (info는 `VirtualInfo` / `PtrInfo` enum 중심)

---

### 13. optimizer.rs ↔ optimizer.py

**이미 올라온 부분:**
- `call_pure_results`
- speculative operation protection
- `notice_guard_future_condition` / `replace_guard`
- `pendingfields`
- `force_box`

**남은 격차:**
- resumedata memo
- full guard sharing/emission parity
- trace iteration / debug logging breadth

---

### 14. intutils.rs ↔ intutils.py

**95% 완성.** 모든 핵심 transfer function 구현됨.
**누락:** make_guards, getnullness, _are_knownbits_implied, RPython type wrappers
**Rust 추가:** 1080줄 테스트 스위트

---

### 15. virtualstate.rs ↔ virtualstate.py

**이미 올라온 부분:**
- `lenbound`
- `make_inputargs`
- `force_boxes`

**남은 격차:**
- Runtime value inspection (CPU backend calls)
- renum tracking (필드 앨리어싱 일관성)
- VirtualStateConstructor (visitor factory)
- `GenerateGuardState`

---

### 16. vector.rs ↔ vector.py

**누락:**
- Loop unrolling (unroll_loop_iterations)
- Index guard analysis (guard → header 이동)
- Accumulation pack support
- Pack merging (2-pack → 4-pack)
- Chain following (def-use/use-def)
- Sophisticated cost model

---

### 17-20. warmstate/counter/heapcache/opencoder

**warmstate (70%):** `set_param()` / `get_param()` / `set_param_to_default()` / vectorization params까지 구현. 남은: `confirm_enter_jit`, `get_location`, callback breadth
**counter (55%):** `compute_threshold()`까지 구현. 남은: JitCell chain, `fetch_next_hash`, GC 통합
**heapcache (95%, heap.rs에 통합):** field/array/quasi-immutable/arraylen/loop-invariant cache 구현. 남은: explicit version system
**opencoder (65%):** Snapshot/TopSnapshot/SnapshotStorage/TraceIterator/CutTrace 추가됨. 남은: constant pooling 최적화와 RPython의 전체 resumedata 연계

---

### 21. majit-analyze/lib.rs ↔ codewriter.py 전체 orchestrator

**새로 추가된 내용 (보고서 누락):**
- `analyze_full()` 추가
- `ResolvedCall.graph`, `MethodInfo.graph` 추가
- graph-based classification을 legacy path 위에 겹쳐 올리는 혼합 구조 도입
- `resolve_call_chain()`는 graph rewrite → raw graph → method-name → opcode pattern text 순으로 단계적 fallback을 둔다

**현재 격차:**
- RPython [codewriter.py]는 typed flow graph를 canonical IR로 쓰지만, `majit-analyze`는 아직 legacy string path와 graph path가 병행된다
- `analyze_full()`은 graph pipeline 결과를 따로 반환하지만, 기본 `analyze_multiple()` 결과 타입은 아직 legacy `AnalysisResult` 중심이다
- `pyre-jit/build.rs`는 이제 `analyze_multiple_with_config()`로 virtualizable-aware graph rewrite metadata를 넘기지만, consumer build 전체가 graph pipeline 결과를 canonical source-of-truth로 쓰는 단계까지는 아직 아니다
- `generate_from_graph()`가 존재해도 consumer build는 아직 `generate_trace_code()` legacy path를 사용한다
- 따라서 현재 상태는 “전환 중”이지 “완전한 codewriter parity”는 아니다

### 22. graph.rs / front/ast.rs ↔ flowspace

**새로 추가된 내용 (보고서 누락):**
- `MajitGraph`, `BasicBlock`, `Terminator`, `inputargs`(Phi) 도입
- AST lowering에서 `if`/`match`/`while`/`loop`/`for`의 block split 구현
- field/array/call/binop/unary op의 구조적 lowering 도입
- `parse.rs`도 opcode arm 추출 시 가능하면 method graph를 붙이고 `classify_from_graph()`를 먼저 시도한다

**현재 격차:**
- Rust compiler IR를 직접 쓰는 것이 아니라 `syn` AST 위의 좁은 graph이므로 semantic precision이 제한됨
- `match` lowering은 여전히 단순화되어 있고, `for`는 iterator semantics를 모델링하지 못함
- unsupported expression은 `Unknown { summary }`로 남아 여전히 문자열 fallback에 의존
- effect info, exception edges, borrow/alias model, exact switch lowering은 없다

### 23. annotate.rs / rtype.rs ↔ annrpython.py / rtyper.py

**새로 추가된 내용 (보고서 누락):**
- fixpoint annotation pass 추가
- annotation → concrete type resolution pass 추가

**현재 격차:**
- 현재 annotation은 `ValueType::{Int,Ref,Float,...}` 수준의 얕은 lattice
- `Call` 추론도 target name heuristic(`contains("add")`, `contains("len")`)에 크게 의존
- RPython의 repr specialization, low-level descriptor production, exception/data-layout integration과는 아직 거리가 멀다

### 24. passes/jtransform.rs ↔ jtransform.py

**새로 추가된 내용 (보고서 누락):**
- graph-based `rewrite_graph()` 추가
- virtualizable field/array rewrite
- call effect 분류(`CallElidable`, `CallResidual`, `CallMayForce`) scaffold

**현재 격차:**
- RPython은 `SpaceOperation` + descriptor/effectinfo 중심인데, 현재 Rust pass는 field/array name과 call target string에 크게 의존
- `vable_array_vars`, legality checks, raw/gc dispatch, descriptor attachment 수준은 아직 아니다
- 즉 “형태는 jtransform처럼 생겼지만 semantic source-of-truth는 아직 약하다”

### 25. flatten.rs / pipeline.rs ↔ flatten.py + codewriter orchestration

**새로 추가된 내용 (보고서 누락):**
- CFG → `FlatOp` linearization
- label/jump/phi move
- full analysis pipeline (`annotate -> rtype -> jtransform -> flatten`) 추가
- graph-only emission entry point `generate_from_graph()`가 존재한다

**현재 격차:**
- liveness, register allocation, parallel move ordering, descriptor-aware flattening은 아직 없다
- pipeline은 테스트 스캐폴딩으로는 유의미하지만, 실제 consumer build 경로와 완전히 통합되지는 않았다
- 특히 `generate_from_graph()`는 build.rs consumer에서 아직 canonical output으로 사용되지 않는다

### 26. patterns.rs ↔ 기존 string classifier

**새로 추가된 내용 (보고서 누락):**
- legacy `body_summary.contains(...)` 중심 분류에서 graph-based `classify_from_graph()`가 primary path로 들어오기 시작함

**현재 격차:**
- 예전의 `classify_method_body_with_vable()`류 string classifier는 제거됐지만, 완전히 typed descriptor matching으로 간 것은 아니다
- 그래도 `classify_from_graph()` 자체가 call target / field name 문자열 heuristic를 많이 쓴다
- 즉 “source text contains”에서 “graph op + symbolic name contains”로 한 단계 진전됐지만, 아직 RPython의 typed descriptor matching은 아니다

### 27. majit-runtime/lib.rs ↔ rlib/jit.py / rvirtualizable.py

**보고서 누락 보충:**
- `hint_access_directly`, `hint_fresh_virtualizable`, `hint_force_virtualizable` 표면은 존재한다
- 그러나 현재는 proc-macro/runtime 일부 경로가 소비할 뿐, RPython처럼 annotator/rtyper/codewriter/runtime 전체에서 canonical하게 소비되는 수준은 아니다
- 특히 public hint 함수는 여전히 identity/no-op surface이고, semantic consumption은 macro lowering 쪽에 더 치우쳐 있다
- 따라서 virtualizable parity의 남은 핵심 격차는 이제 runtime보다도 translator-wide hint semantics에 있다

### 28. codewriter/codegen.rs ↔ assembler.py / translation emission

**새로 확인된 내용 (보고서 누락):**
- generic codewriter는 state-only interpreter에 대해 `state_fields = { ... }`를 실제로 생성한다
- scalar `int/ref/float`와 `Vec<T>` 배열을 `#[jit_interp(...)]` attr로 추론한다
- `virtualizable_fields` attr emission도 별도로 지원한다

**현재 격차:**
- generic emission과 proc-macro front-end가 아직 완전히 같은 canonical lowering surface를 공유하지 않는다
- generic codewriter는 이제 `state_fields`에서 scalar `int/ref/float`와 `Vec<T>` 배열을 추론하지만, 이 emission은 여전히 legacy `AnalysisResult` / `TracePattern` 출력 위에 얹혀 있다
- `StorageConfig.virtualizable` user-facing seam은 제거됐지만, generic codewriter는 여전히 legacy `AnalysisResult` / `TracePattern` 출력과 graph pipeline 출력이 병행된다
- RPython codewriter는 `vinfo/descr/effectinfo`를 단일 canonical source로 쓰고, 별도 legacy emission path가 병행되지 않는다

### 29. jit_interp mod.rs / codegen_state.rs / codegen_trace.rs ↔ translator/codewriter bridge

**새로 확인된 내용 (보고서 누락):**
- `state_fields` mode가 실제로 존재하며 `int/ref/float`, `[int]/[ref]/[float]`, `[T; virt]`를 지원한다
- virtualizable field/array rewrite와 hint suppression/forcing rewrite가 macro lowerer에 실제로 들어와 있다

**현재 격차:**
- parser는 이제 `storage.virtualizable`를 deprecated error로 거부하고, generated trace wrapper도 `trace_jitcode_with_data_ptr(...)`를 더 이상 기본 경로로 사용하지 않는다
- 다만 `state_fields`는 여전히 `majit` 고유 front-end sugar이고, generic analyzer/codewriter와 proc-macro lowering이 완전히 같은 internal IR를 공유하는 단계까지는 아직 아니다
- `state_fields` 자체도 RPython에 직접 대응하는 기능은 아니므로, parity를 주장하려면 결국 기존 virtualizable/state op model로 lowering되어야 한다

### 30. consumer build ownership ↔ RPython translation ownership

**새로 확인된 내용 (보고서 누락):**
- `aheui-jit`는 더 이상 `StorageConfig { virtualizable: true, ... }`를 설정하지 않는다
- `pyre-jit`는 build 단계에서 PyFrame virtualizable metadata를 analyzer에 넘기기 시작했다
- `pyre-jit` runtime과 build-time analyzer는 이제 `virtualizable_spec.rs`를 공유해 같은 field/array index contract를 사용한다
- 그래도 storage scan 함수와 일부 policy choice는 여전히 consumer build script에 남아 있다

**현재 격차:**
- RPython은 이런 정책을 translator/codewriter가 소유하지, consumer build script가 선택하지 않는다
- `pyre-jit/build.rs`에는 아직 `generate_jitcode(...)` 경로가 TODO로 남아 있고, generated `jit_mainloop_gen.rs`를 canonical consumer path로 쓰지 않는다
- 따라서 `majit`는 runtime/macro core는 많이 올라왔지만, translation ownership은 아직 framework 내부로 완전히 수렴하지 않았다

### 31. jitdriver.rs / pyjitpl.rs ↔ warmstate.py / compile.py / pyjitpl.py

**새로 확인된 내용 (보고서 누락):**
- `JitDriver::sync_before()`는 이제 virtualizable array lengths를 trace-entry box layout용으로 캐시할 때, 가능한 경우 실제 virtualizable object에서 `VirtualizableInfo::load_list_of_boxes()`로 읽고, 없을 때만 `JitState::virtualizable_array_lengths()`로 fallback한다
- `pyjitpl.rs`는 trace start / force-start / loop preamble patching에서 같은 `trace_entry_vable_lengths()` helper를 공유한다

**현재 격차:**
- 이 경로는 `RPython compile.py` 쪽 semantics에 더 가까워졌지만, `majit` 내부에는 아직 `vable_ptr` / `vable_array_lengths` staging cache가 남아 있다
- 즉 user-facing seam은 대부분 줄었지만, framework 내부에는 trace-entry virtualizable metadata를 임시 캐시하는 `majit` 고유 단계가 아직 존재한다
