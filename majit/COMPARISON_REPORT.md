# majit vs rpython Line-by-Line Comparison Report

## 요약: 파일별 구현 완성도

| 파일 | 줄수 | 완성도 | 주요 누락 |
|------|------|--------|-----------|
| **resoperation.rs** | 3,360 | 99% | 완벽 일치 |
| **intutils.rs** | 2,680 | 95% | make_guards, getnullness |
| **intbounds.rs** | 2,204 | 95% | ~~STRGETITEM/GETFIELD bounds, backward propagation~~ 모두 추가됨 |
| **pure.rs** | 1,499 | 95% | ~~COND_CALL_VALUE~~ 추가됨. 완료 |
| **heapcache.rs** | 685 | 95% | 남은: explicit version system (현 아키텍처에서 불필요할 수 있음) |
| **simplify.rs** | 242 | 95% | 완료 |
| **trace.rs (history.rs)** | 764 | 95% | 완료 |
| **optimizer.rs** | 1,727 | 90% | ~~resumedata_memo~~ 추가됨. 남은: guard sharing |
| **virtualstate.rs** | 1,648 | 90% | ~~GenerateGuardState~~ 추가됨. 완료 |
| **info.rs** | 1,211 | 85% | ~~force_box~~ 추가됨. 남은: guard tracking (last_guard_pos) |
| **descr.rs** | 1,500 | 85% | bitstring optimization, CallInfoCollection |
| **shortpreamble.rs** | 1,888 | 85% | ~~CompoundOp, ShortInputArg~~ 추가됨. 남은: full ShortBoxes traversal |
| **majit-analyze lib.rs** | — | 82% | canonical pipeline 정착. 남은: descriptor/effectinfo object parity |
| **quasi_immut.rs** | 297 | 80% | CPU 통합, 디버그/통계 |
| **virtualizable.rs** | — | 80% | descriptor/effect parity |
| **blackhole.rs** | 3,550 | 80% | goto_if_not, raise/reraise (현 fallback 전략에서는 불필요) |
| **warmstate.rs** | 2,359 | 80% | ~~confirm_enter_jit, get_location~~ 추가됨. 남은: callback breadth |
| **rewrite.rs** | 3,548 | 75% | ~~GUARD_SUBCLASS/IS_OBJECT~~ 추가됨. 남은: INT_PY_DIV/MOD, postprocess |
| **virtualize.rs** | 3,881 | 75% | ~~COND_CALL, JIT_FORCE_VIRTUAL, RAW_MALLOC~~ 추가됨. 남은: GETARRAYITEM_RAW |
| **jit_interp mod.rs** | — | 72% | canonical IR 공유 미완 |
| **jitdriver.rs / pyjitpl.rs** | 11,253 | 70% | vable length staging seam |
| **graph.rs / front/ast.rs** | — | 70% | match/switch lowering, effect info |
| **opencoder.rs** | 1,139 | 65% | constant pooling 최적화 |
| **vstring.rs** | 1,298 | 65% | ~~initialize_forced_string~~ 추가됨. 남은: copy_str_content |
| **codewriter/codegen.rs** | — | 65% | graph-native final assembler |
| **patterns.rs / call_match.rs** | — | 65% | descriptor/effectinfo object parity |
| **virtual_ref.rs** | 259 | 65% | ~~is_virtual_ref~~ 추가됨. 남은: graph transformation |
| **counter.rs** | 397 | 60% | ~~fetch_next_hash~~ 추가됨. 남은: full JitCell chain |
| **flatten.rs / pipeline.rs** | — | 60% | liveness, regalloc |
| **heap.rs** | 3,569 | 60% | ~~pendingfields, variable_index~~ 추가됨. 남은: version, serialization |
| **bridgeopt.rs** | 829 | 60% | 완료 |
| **guard.rs** | 931 | 55% | Guard implication, transitive loop bounds |
| **annotate.rs / rtype.rs** | — | 55% | repr/descriptor specialization |
| **vector.rs** | 1,513 | 55% | ~~extend/combine_packset~~ 추가됨. 남은: loop unrolling |
| **unroll.rs** | 3,130 | 90% | Python 17 methods 중 15개 구현 + compile.py 2단계 peeling 통합. force_op_from_preamble/setinfo는 apply_exported_info로 재설계 |
| **resume.rs** | 3,735 | 50% | ~~VStrPlain~~ 추가됨. 남은: VirtualCache |
| **majit-runtime jit hints** | — | 35% | framework-wide hint semantics |
| **pyjitpl.rs** | 8,739 | 다른 설계 | Python=해석, Rust=컴파일 실행 |
| **jitcode.rs** | 3,541 | 다른 설계 | Python=컨테이너, Rust=해석기+빌더 |
| **codegen lib.rs** | 1,167 | 다른 설계 | bh_* 22개 추가됨 |

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
- `hint_access_directly` / `hint_fresh_virtualizable` / `hint_force_virtualizable`의 public 함수가 identity/no-op surface인 점 자체는 RPython `rlib/jit.py`와 같다
- 남은 격차는 public surface가 아니라, 그 hint vocabulary가 `rpython/rtyper/rvirtualizable.py` + `jtransform.py` + downstream lowering 전체에서 descriptor/effectinfo 수준으로 소비되는 깊이 쪽이다
- 즉 runtime parity는 많이 올라왔지만, translator-wide hint ownership은 아직 `rpython/rlib/jit.py` + `rpython/rtyper/rvirtualizable.py` + `jtransform.py` 전체 체인과 완전히 같지는 않다
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
- pipeline-only canonical API: `analyze_pipeline(_with_config)` / `analyze_multiple_pipeline(_with_config)` / `generate_trace_code_from_pipeline()`
- `ResolvedCall.graph`, `MethodInfo.graph` 추가
- canonical opcode dispatch는 parse/front dispatch arm + graph/pipeline classification에서 직접 생성된다
- canonical opcode dispatch arm은 이제 raw pattern string이 아니라 `OpcodeDispatchSelector::{Path,Wildcard,Or,Unsupported}` 구조를 canonical source-of-truth로 쓴다
- canonical non-test path는 더 이상 `body_summary` metadata를 들고 다니지 않고, function graph/name만 모은다
- 이전 `AnalysisResult` / `resolve_call_chain()` / `classify.rs` compatibility 경로는 제거되었고, analyzer public/test path도 pipeline-only로 수렴했다

**현재 격차:**
- RPython [codewriter.py]는 typed flow graph를 canonical IR로 쓰는데, `majit-analyze`도 이제 public consumer 경로는 pipeline-only canonical API를 쓴다
- `pyre-jit/build.rs`는 이제 `analyze_multiple_pipeline_with_config()` + `generate_trace_code_from_pipeline()`만 사용한다
- canonical opcode dispatch는 더 이상 `legacy.opcodes`에서 복사되지 않고, parse/front에서 뽑은 dispatch arm + graph/pipeline classification을 직접 결합해 만든다
- generated consumer의 `CANONICAL_TRACE_PATTERNS`도 이제 quoted match-arm text가 아니라 selector의 exact canonical key(`Instruction::LoadFast` 같은 path key)만 내보낸다
- canonical pipeline 경로는 legacy compatibility analysis를 내부적으로 계산하지 않는다. 즉 analyzer는 이제 graph/pipeline만 평가한다
- `AnalysisResult` 기반 compatibility analyzer는 제거되었다
- canonical opcode dispatch는 이제 pipeline function name lookup이 아니라, parse/front에서 resolve한 callee graph를 직접 rewrite/classify한다
- opcode arm의 `handler_calls`는 이제 단순 lossy symbol string이 아니라 `Method { name, receiver_root }` / `FunctionPath(CallPath)`로 구조화된다
- canonical resolver는 concrete receiver root와 trait-bound default body를 우선 따라가고, receiver identity를 알고 있을 때는 broad unrelated impl sweep을 하지 않도록 줄었다
- canonical call/effect 분류는 이제 `call_match.rs`의 `CallDescriptor { target, effect_info }`를 기본 source-of-truth로 쓰고, active consumer는 source-owned call spec override를 함께 넘긴다
- bare `describe_call()` fallback은 이제 arithmetic helper 중심으로 줄었고, `PyFrame` method residual/elidable policy는 source-owned override가 canonical path다
- `CallEffectClass` / `CallDescriptorId` 같은 중간 effect/id layer는 제거됐고, explicit override도 `EffectInfo`를 직접 들고 간다
- 그래도 method call/effect identity는 아직 real backend call descr/effectinfo object graph가 아니라 AST-level call identity + canonical descriptor table에 머문다. 즉 RPython의 full descriptor/effectinfo identity join까지는 아직 아니다
- `generate_from_graph()` 자체는 아직 summary emitter이고, 최종 tracing helper emission은 canonical pipeline metadata를 이용해 trace-helper output shape를 조립하는 과도기 구조다
- 따라서 현재 상태는 “전환 중”이지 “완전한 codewriter parity”는 아니다

### 22. graph.rs / front/ast.rs ↔ flowspace

**새로 추가된 내용 (보고서 누락):**
- `MajitGraph`, `BasicBlock`, `Terminator`, `inputargs`(Phi) 도입
- AST lowering에서 `if`/`match`/`while`/`loop`/`for`의 block split 구현
- field/array/call/binop/unary op의 구조적 lowering 도입
- `parse.rs`는 canonical path에서 opcode dispatch arm과 function graph를 직접 추출하며, legacy summary-bearing collectors는 test-only로 격하됐다
- dispatch arm도 이제 `quote!(#pat).to_string()` raw text 대신 structured selector로 추출된다. 남은 string은 emission/debug용 canonical key뿐이다
- graph builder의 unary/binary op도 이제 token text를 그대로 보존하지 않고 `syn::{UnOp,BinOp}` exact mapping(`add`, `sub`, `lt`, `neg` 등)으로 canonicalize된다

**현재 격차:**
- Rust compiler IR를 직접 쓰는 것이 아니라 `syn` AST 위의 좁은 graph이므로 semantic precision이 제한됨
- `match` lowering은 여전히 단순화되어 있고, `for`는 iterator semantics를 모델링하지 못함
- unsupported statement/expression은 이제 `UnknownKind::{MacroStmt,UnsupportedLiteral,UnsupportedExpr}`로 남는다. 큰 문자열 summary fallback은 제거됐지만, 여전히 typed residual bucket은 남아 있다
- effect info, exception edges, borrow/alias model, exact switch lowering은 없다

### 23. annotate.rs / rtype.rs ↔ annrpython.py / rtyper.py

**새로 추가된 내용 (보고서 누락):**
- fixpoint annotation pass 추가
- annotation → concrete type resolution pass 추가

**현재 격차:**
- 현재 annotation은 `ValueType::{Int,Ref,Float,...}` 수준의 얕은 lattice
- `Call` 추론은 이제 structured `CallTarget`의 primary symbol을 읽고, path string 재정규화에 의존하지 않는다
- 그래도 여전히 descriptor/repr specialization 수준은 아니다
- RPython의 repr specialization, low-level descriptor production, exception/data-layout integration과는 아직 거리가 멀다

### 24. passes/jtransform.rs / call_match.rs ↔ jtransform.py / effectinfo

**새로 추가된 내용 (보고서 누락):**
- graph-based `rewrite_graph()` 추가
- virtualizable field/array rewrite
- call effect 분류(`CallElidable`, `CallResidual`, `CallMayForce`) scaffold
- explicit `call_effects` override surface 추가
- canonical call descriptor table(`CallDescriptor { target, effect_info }`) 추가
- builtin helper 분류는 public enum이 아니라 exact `CallTargetPattern` matcher로 내부화됨
- `pyre-jit/build.rs`는 이제 `virtualizable_spec.rs`뿐 아니라 source-owned `call_spec.rs`도 analyzer에 넘긴다

**현재 격차:**
- RPython은 `SpaceOperation` + descriptor/effectinfo 중심인데, 현재 Rust pass는 structured `CallTarget` + canonical `CallDescriptor { target, effect_info }` table과 exact target matcher에 의존한다
- `call_effects` override도 이제 string key가 아니라 exact `CallTarget` identity를 쓴다
- `vable_array_vars`, legality checks, raw/gc dispatch, descriptor attachment 수준은 아직 아니다
- broad helper enum(`BuiltinCallKind`)은 active path에서 제거됐지만, helper semantics 자체는 아직 real descr/effectinfo object graph보다는 exact target table에 더 가깝다
- 즉 semantic source-of-truth는 한 단계 강해졌지만, 아직 true descr/effectinfo object까지는 아니다
- opcode arm identity도 이제 raw match-arm text가 아니라 selector object를 쓰지만, RPython처럼 실제 opcode number / bytecode descriptor object를 source-of-truth로 두는 단계까지는 아직 아니다

### 25. flatten.rs / pipeline.rs ↔ flatten.py + codewriter orchestration

**새로 추가된 내용 (보고서 누락):**
- CFG → `FlatOp` linearization
- label/jump/phi move
- full analysis pipeline (`annotate -> rtype -> jtransform -> flatten`) 추가
- graph-only emission entry point `generate_from_graph()`가 존재한다

**현재 격차:**
- liveness, register allocation, parallel move ordering, descriptor-aware flattening은 아직 없다
- pipeline은 이제 `pyre-jit` build의 canonical 경로에 실제로 연결돼 있지만, 최종 helper assembly는 아직 완전한 graph-native emitter라기보다 pipeline summary를 조합하는 emitter에 가깝다
- 즉 consumer 통합은 이뤄졌지만, RPython codewriter처럼 descriptor/effectinfo 기반 최종 assembler를 완전히 갖춘 단계는 아직 아니다

### 26. patterns.rs ↔ 기존 string classifier

**새로 추가된 내용 (보고서 누락):**
- legacy `body_summary.contains(...)` 중심 분류는 제거되었고, canonical path는 graph + 최소 symbol metadata만 사용한다
- call/helper classification은 이제 `call_match.rs`의 `CallDescriptor`를 canonical source로 사용한다

**현재 격차:**
- 예전의 `classify_method_body_with_vable()`류 string classifier와 body-summary path는 제거됐지만, 완전히 typed descriptor matching으로 간 것은 아니다
- `classify_from_graph()`는 더 이상 normalized call-path string table을 만들지 않고 `CallTarget` + `CallDescriptor`를 직접 본다
- 즉 “source text contains”에서 “graph op + structured call identity + canonical call descriptor”로 한 단계 진전됐지만, 아직 RPython의 typed descriptor matching은 아니다

### 27. majit-runtime/lib.rs ↔ rlib/jit.py / rvirtualizable.py

**보고서 누락 보충:**
- `hint_access_directly`, `hint_fresh_virtualizable`, `hint_force_virtualizable` 표면은 존재한다
- `VirtualizableHintKind`와 `classify_virtualizable_hint_segments()`가 runtime crate에 canonical vocabulary로 존재하고, analyzer/macro는 이제 path string이 아니라 segment identity로 이를 공유한다
- public hint 함수가 identity/no-op surface인 점은 RPython과 동일하다
- 그러나 현재는 proc-macro/runtime 일부 경로가 소비할 뿐, RPython처럼 annotator/rtyper/codewriter/runtime 전체에서 canonical하게 소비되는 수준은 아니다
- 따라서 virtualizable parity의 남은 핵심 격차는 이제 runtime보다도 translator-wide hint semantics에 있다

### 28. codewriter/codegen.rs ↔ assembler.py / translation emission

**새로 확인된 내용 (보고서 누락):**
- generic codewriter는 state-only interpreter에 대해 `state_fields = { ... }`를 실제로 생성한다
- old codewriter path의 `state_fields` config도 이제 raw string이 아니라 typed `TokenStream`/`Ident`를 source로 쓴다
- `virtualizable_fields` attr emission도 별도로 지원한다

**현재 격차:**
- generic emission과 proc-macro front-end가 아직 완전히 같은 canonical lowering surface를 공유하지 않는다
- generic codewriter는 이제 `state_fields`를 int-only surface로 제한하고, canonical pipeline result만 읽는다
- `StorageConfig.virtualizable` user-facing seam은 제거됐고, canonical generated output도 더 이상 legacy `TRACE_PATTERNS` alias나 body-summary metadata를 싣지 않는다
- old `generate_jitcode(...)` path도 string roundtrip은 더 줄었지만, 여전히 `aheui-jit` 전용 proc-macro/codewriter ownership으로 남아 있다
- 다만 generic codewriter는 아직 graph-native emitter라기보다 pipeline summary/dispatch를 조합하는 단계이며, descriptor/effectinfo 중심 final assembler와는 차이가 남는다
- generated dispatch table은 이제 selector key를 exact equality로 lookup하며 `.contains("LoadFast")` 같은 consumer fallback은 제거됐다
- active consumer는 `generate_trace_code_from_pipeline()`를 쓰고, `generate_from_graph()`는 아직 테스트/보조 emitter 성격이 강하다
- RPython codewriter는 `vinfo/descr/effectinfo`를 단일 canonical source로 쓰고, 별도 legacy emission path가 병행되지 않는다

### 29. jit_interp mod.rs / codegen_state.rs / codegen_trace.rs ↔ translator/codewriter bridge

**새로 확인된 내용 (보고서 누락):**
- `state_fields` mode는 실제로 존재하지만, active JitCode/runtime semantics에 맞춰 `int`, `[int]`, `[int; virt]`만 지원한다
- virtualizable field/array rewrite와 hint suppression/forcing rewrite가 macro lowerer에 실제로 들어와 있다
- abort-pattern 판정도 active path에서는 `Ident -> String` 변환이 아니라 AST ident/path 비교로 줄었다

**현재 격차:**
- parser는 이제 `storage.virtualizable`를 deprecated error로 거부하고, generated trace wrapper도 `trace_jitcode_with_data_ptr(...)`를 더 이상 기본 경로로 사용하지 않는다
- `jitcode_lower.rs`의 broad statement/expression `contains(...)` storage detection은 AST walk 기반으로 줄었다. 다만 helper/pool identity 전체가 typed descriptor resolution으로 바뀐 것은 아직 아니다
- 다만 `state_fields`는 여전히 `majit` 고유 front-end sugar이고, generic analyzer/codewriter와 proc-macro lowering이 완전히 같은 internal IR를 공유하는 단계까지는 아직 아니다
- `state_fields` 자체도 RPython에 직접 대응하는 기능은 아니므로, parity를 주장하려면 결국 기존 virtualizable/state op model로 lowering되어야 한다
- dummy storage bootstrap은 `state_fields` trace wrapper에서 제거되었고, state-only mode는 더 이상 fake storage pool을 요구하지 않는다
- old `storage.virtualizable` user-facing seam과 generic `ArmTransformer` extension point는 제거됐다. 남은 차이는 proc-macro lowering과 canonical analyzer pipeline이 아직 완전히 같은 internal IR를 공유하지 않는다는 점이다

### 30. consumer build ownership ↔ RPython translation ownership

**새로 확인된 내용 (보고서 누락):**
- `aheui-jit`는 더 이상 `StorageConfig { virtualizable: true, ... }`를 설정하지 않는다
- `pyre-jit`는 build 단계에서 PyFrame virtualizable metadata를 analyzer에 넘기기 시작했다
- `pyre-jit` runtime과 build-time analyzer는 이제 `virtualizable_spec.rs`를 공유해 같은 field/array index contract를 사용한다
- `pyre-jit` runtime/build는 이제 `call_spec.rs`를 통해 PyFrame helper effect policy도 source-owned canonical spec로 공유한다
- `aheui-jit`의 branch-arm detection / storage rewrite도 문자열 치환이 아니라 AST 구조 매칭으로 옮겨졌다
- `aheui-jit/build.rs`는 이제 source 읽기와 output write만 맡고, opcode-dispatch 해석/arm transform/JIT config 조립은 `src/jit_spec.rs`로 옮겨졌다
- `aheui-jit`는 더 이상 generic `ArmTransformer` extension point에 기대지 않고, source-owned local AST transform만 사용한다
- 그래도 storage scan 함수와 일부 policy choice는 여전히 consumer build script에 남아 있다

**현재 격차:**
- RPython은 이런 정책을 translator/codewriter가 소유하지, consumer build script가 선택하지 않는다
- `pyre-jit/build.rs`는 canonical pipeline consumer이고, `aheui-jit`은 여전히 `generate_jitcode(...)` proc-macro/codewriter path를 쓴다
- 다만 `aheui-jit`의 interpreter-specific JIT policy는 더 이상 build.rs가 직접 들고 있지 않고 `src/jit_spec.rs`로 source-owned화되었다
- `aheui-jit`의 남은 격차는 build.rs ownership보다는, 아직 canonical pipeline consumer가 아니라 old proc-macro/codewriter path를 유지한다는 점이다
- 실제 tracing code는 이제 pipeline-only canonical 결과를 직접 받아 `generate_trace_code_from_pipeline()`가 조립한다
- 따라서 `majit`는 runtime/macro core는 많이 올라왔지만, translation ownership은 아직 framework 내부로 완전히 수렴하지 않았다

### 31. jitdriver.rs / pyjitpl.rs ↔ warmstate.py / compile.py / pyjitpl.py

**새로 확인된 내용 (보고서 누락):**
- `JitDriver::sync_before()`는 이제 virtualizable array lengths를 trace-entry box layout용으로 준비할 때, 가능한 경우 실제 virtualizable object에서 `VirtualizableInfo::load_list_of_boxes()`로 읽고, 그게 불가능할 때만 `JitState::virtualizable_array_lengths()`로 fallback한다
- `VirtualizableInfo`는 embedded array container layout까지 지원하며, `PyFrame.locals_cells_stack_w`도 이제 object/vinfo에서 직접 길이를 읽을 수 있는 spec을 갖는다
- `pyjitpl.rs`는 trace start / force-start / loop preamble patching에서 같은 `trace_entry_vable_lengths()` helper를 공유하고, heap-readable layout에서는 cached lengths보다 object/vinfo를 우선한다

**현재 격차:**
- 이 경로는 `RPython compile.py` 쪽 semantics에 더 가까워졌지만, `majit` 내부에는 아직 fallback-only `vable_ptr` / `vable_array_lengths` staging cache가 남아 있다
- 즉 user-facing seam은 대부분 줄었지만, framework 내부에는 trace-entry virtualizable metadata를 임시 캐시하는 `majit` 고유 단계가 아직 존재한다
