# majit vs rpython Line-by-Line Comparison Report

## 요약: 파일별 구현 완성도

| 파일 | 완성도 | 주요 누락 |
|------|--------|-----------|
| **resoperation.rs** | 99% | 238개 opcode variant 완벽 일치. FloatFloorDiv만 Rust 전용 추가 |
| **intutils.rs** | 95% | make_guards, getnullness, RPython type wrappers |
| **trace.rs** | 95% | iter_ops, iter_guards, split_at_label, get_iter 등 추가 완료 |
| **descr.rs** | 85% | ~~can_invalidate/can_collect~~ 추가됨. 남은: bitstring optimization, CallInfoCollection, graph analyzers |
| **simplify.rs** | 85% | GUARD_FUTURE_CONDITION |
| **quasi_immut.rs** | 80% | ~~QuasiImmutDescr~~ 추가됨. 남은: CPU 통합, 디버그/통계 |
| **virtualizable.rs** | 75% | Python의 finish() 그래프 변환, 동적 JitCell subclass |
| **intbounds.rs** | 70% | pure_from_args 합성, CALL_PURE_I/oopspec, string/field bounds, backward propagation dispatcher |
| **opencoder.rs** | 65% | ~~Snapshot/TopSnapshot/SnapshotStorage/TraceIterator/CutTrace~~ 추가됨. 남은: constant pooling 최적화 |
| **virtualize.rs** | 60% | GUARD_NO_EXCEPTION, CALL_MAY_FORCE, COND_CALL, JIT_FORCE_VIRTUAL, INT_ADD raw slice |
| **blackhole.rs** | 60% | 제어 흐름(goto_if_not), 예외(raise/reraise), 문자열 ops, 포탈/재귀 호출 |
| **warmstate.rs** | 55% | JC_FORCE_FINISH, 12+ set_param_* methods, vectorization params, callbacks |
| **rewrite.rs** | 50% | 가드 최적화 10종, loop invariant, arraycopy/move, INT_PY_DIV/MOD, postprocess 메커니즘 |
| **guard.rs** | 50% | Guard implication (numeric range), transitive loop bounds |
| **counter.rs** | 50% | JitCell chain, compute_threshold, fetch_next_hash, parameterized decay |
| **heap.rs** | 45% | DICT_LOOKUP, postponed ops, exception guards, effect-info invalidation, variable-index array caching |
| **resume.rs** | 45% | Reader/Decoder 클래스, 문자열 virtual info 6종, VirtualCache |
| **pure.rs** | 40% | OVF 처리, constant folding, preamble ops, descriptor-based caching, COND_CALL_VALUE |
| **optimizer.rs** | 40% | Resume data, guard sharing/emission, constant folding, speculative ops, call_pure_results |
| **virtualstate.rs** | 40% | Runtime value inspection, force_boxes, renum tracking, lenbound |
| **bridgeopt.rs** | 40% | serialize/deserialize_optimizer_knowledge (다른 접근) |
| **vstring.rs** | 35% | Unicode, oopspec call handlers (STR_CONCAT/SLICE/EQUAL/CMP), postprocessing |
| **heapcache.rs** | 35% | Array caching, quasi-immutable tracking, loop invariant caching, version system |
| **info.rs** | 30% | Guard tracking, force mechanics, visitor pattern, string optimization, float constants |
| **unroll.rs** | 30% | UnrollOptimizer wrapper, preamble/bridge optimization, virtual state, short preamble inlining |
| **vector.rs** | 30% | Loop unrolling, guard analysis, accumulation packs, pack merging |
| **shortpreamble.rs** | 25% | PreambleOp, HeapOp, PureOp, LoopInvariantOp, ShortBoxes |
| **virtual_ref.rs** | 20% | force_virtual(), tracing callbacks, graph transformation |
| **pyjitpl.rs** | 다른 설계 | Python=해석 기반(84 methods), Rust=컴파일 실행 기반(93 methods) |
| **jitcode.rs** | 다른 설계 | Python=데이터 컨테이너(167줄), Rust=완전한 해석기+빌더(2810줄) |
| **codegen lib.rs** | 다른 설계 | ~~bh_* 헬퍼 누락~~ 22개 추가됨. Python=90개 메서드, Rust=42+개 메서드 |
| **majit-analyze lib.rs** | 60% | legacy 문자열 경로와 graph pipeline 병행. `analyze_full`, `ResolvedCall.graph` 추가됐지만 consumer는 아직 legacy 중심 |
| **graph.rs / front/ast.rs** | 55% | BasicBlock/Phi/CFG/dataflow lowering 추가. 남은: 정확한 match/switch lowering, effect info, 예외/borrow/model |
| **annotate.rs / rtype.rs** | 40% | fixpoint annotation + concrete type pass 스캐폴딩 추가. 남은: 실제 annotator/rtyper 수준의 repr/descriptor specialization |
| **flatten.rs / pipeline.rs** | 45% | CFG flatten/label/jump/phi move + end-to-end pipeline 추가. 남은: liveness, regalloc, parallel move ordering, consumer integration |
| **patterns.rs** | 55% | graph-based classify_from_graph 추가. 남은: 여전히 call/field name 기반 heuristic 다수 |
| **majit-runtime jit hints** | 35% | promote/elidable류 표면은 있으나 access_directly/fresh/force_virtualizable는 framework-wide semantics로 닫히지 않음 |
| **codewriter/codegen.rs** | 55% | `state_fields`/`virtualizable_fields` attr emission 추가. 남은: macro path와 완전 통합되지 않았고 `storage.virtualizable` emit도 잔존 |
| **jit_interp mod.rs / codegen_state.rs / codegen_trace.rs** | 55% | `state_fields`/virtualizable lowering 추가. 남은: `storage.virtualizable` + `trace_jitcode_with_data_ptr` 는 RPython에 없는 majit 고유 seam |

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

**누락된 최적화:**
- Overflow (OVF) operation deferred handling
- Constant folding (별도 패스에 위임)
- Preamble operation forcing
- Descriptor-based lookup (lookup1/lookup2)
- COND_CALL_VALUE
- GUARD_NO_EXCEPTION handling
- RECORD_KNOWN_RESULT
- Short preamble optimization
- Helper methods (pure(), pure_from_args*(), get_pure_result())

**Rust 추가:**
- Loop-invariant call caching (loopinvariant_cache)

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

---

### 6. simplify.rs ↔ simplify.py

**누락:** GUARD_FUTURE_CONDITION (notice_guard_future_condition 호출)
**로직 차이:** Python은 descr 기반 CALL_PURE 변환, Rust는 type 기반

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
**shortpreamble:** Python에 13개 class/function 중 Rust에는 ShortPreambleBuilder만 부분 구현

---

### 11. vstring.rs ↔ vstring.py

**누락 (대부분):**
- Unicode support 전체
- Oopspec call handlers (STR_CONCAT, STR_SLICE, STR_EQUAL, STR_CMP, SHRINK_ARRAY)
- Postprocessing (STRLEN(NEWSTR) pure fold)
- String forcing to constants
- copy_str_content heuristic (M-character inline)
- _int_add/_int_sub constant folding helpers

---

### 12. info.rs ↔ info.py

**누락 (대부분):**
- Guard tracking (last_guard_pos, mark_last_guard)
- Force mechanics (force_box, force_at_the_end_of_preamble)
- Visitor pattern (visitor_walk_recursive)
- String optimization (getstrlen, getstrhash)
- Float constants (FloatConstInfo)
- make_guards, copy_fields_to_const

**Rust:** 순수 data carrier로 설계 (info는 VirtualInfo/PtrInfo enum)

---

### 13. optimizer.rs ↔ optimizer.py

**누락:**
- Resume data memo
- Guard sharing/emission
- Constant folding + speculative operation protection
- Trace iteration
- Intbounds debug logging
- Call pure results tracking

---

### 14. intutils.rs ↔ intutils.py

**95% 완성.** 모든 핵심 transfer function 구현됨.
**누락:** make_guards, getnullness, _are_knownbits_implied, RPython type wrappers
**Rust 추가:** 1080줄 테스트 스위트

---

### 15. virtualstate.rs ↔ virtualstate.py

**누락:**
- Runtime value inspection (CPU backend calls)
- force_boxes mechanism
- renum tracking (필드 앨리어싱 일관성)
- Lenbound validation
- VirtualStateConstructor (visitor factory)
- make_inputargs_and_virtuals

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

**warmstate (55%):** 핵심 상태 관리 구현, 복잡한 param/callback 시스템 미구현
**counter (50%):** 핵심 카운팅 구현, JitCell chain/GC 통합 미구현
**heapcache (35%):** 기본 field cache 구현, array/quasi-immut/loop-invariant 미구현
**opencoder (65%):** Snapshot/TopSnapshot/SnapshotStorage/TraceIterator/CutTrace 추가됨. 남은: constant pooling 최적화와 RPython의 전체 resumedata 연계

---

### 21. majit-analyze/lib.rs ↔ codewriter.py 전체 orchestrator

**새로 추가된 내용 (보고서 누락):**
- `analyze_full()` 추가
- `ResolvedCall.graph`, `MethodInfo.graph` 추가
- graph-based classification을 legacy path 위에 겹쳐 올리는 혼합 구조 도입

**현재 격차:**
- RPython [codewriter.py]는 typed flow graph를 canonical IR로 쓰지만, `majit-analyze`는 아직 legacy string path와 graph path가 병행된다
- `analyze_full()`은 graph pipeline 결과를 따로 반환하지만, 기본 `analyze_multiple()` 결과 타입은 아직 legacy `AnalysisResult` 중심이다
- consumer build는 아직 graph pipeline을 canonical source-of-truth로 쓰지 않는다
- 따라서 현재 상태는 “전환 중”이지 “완전한 codewriter parity”는 아니다

### 22. graph.rs / front/ast.rs ↔ flowspace

**새로 추가된 내용 (보고서 누락):**
- `MajitGraph`, `BasicBlock`, `Terminator`, `inputargs`(Phi) 도입
- AST lowering에서 `if`/`match`/`while`/`loop`/`for`의 block split 구현
- field/array/call/binop/unary op의 구조적 lowering 도입

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

**현재 격차:**
- liveness, register allocation, parallel move ordering, descriptor-aware flattening은 아직 없다
- pipeline은 테스트 스캐폴딩으로는 유의미하지만, 실제 consumer build 경로와 완전히 통합되지는 않았다

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
- `StorageConfig.virtualizable`가 여전히 user-facing 설정으로 남아 있고 codewriter가 그대로 emit한다
- RPython codewriter는 `storage.virtualizable` 같은 별도 사용자 knob가 아니라 `vinfo/descr/effectinfo`를 canonical source로 쓴다

### 29. jit_interp mod.rs / codegen_state.rs / codegen_trace.rs ↔ translator/codewriter bridge

**새로 확인된 내용 (보고서 누락):**
- `state_fields` mode가 실제로 존재하며 `int/ref/float`, `[int]/[ref]/[float]`, `[T; virt]`를 지원한다
- virtualizable field/array rewrite와 hint suppression/forcing rewrite가 macro lowerer에 실제로 들어와 있다

**현재 격차:**
- parser와 runtime bridge는 여전히 `storage = { virtualizable: true }`를 허용한다
- `codegen_trace.rs`는 이 경우 `trace_jitcode_with_data_ptr(...)` 경로를 사용한다
- 이것은 RPython의 `virtualizable object + descr + virtualizable_boxes` 모델과 다른, `majit` 고유 raw-storage seam이다
- `state_fields` 자체도 RPython에 직접 대응하는 기능은 아니므로, parity를 주장하려면 결국 기존 virtualizable/state op model로 lowering되어야 한다

### 30. consumer build ownership ↔ RPython translation ownership

**새로 확인된 내용 (보고서 누락):**
- `aheui-jit` consumer는 여전히 `StorageConfig { virtualizable: true, ... }`를 직접 설정한다
- storage scan 함수와 일부 policy choice도 consumer build script에 남아 있다

**현재 격차:**
- RPython은 이런 정책을 translator/codewriter가 소유하지, consumer build script가 선택하지 않는다
- 따라서 `majit`는 runtime/macro core는 많이 올라왔지만, translation ownership은 아직 framework 내부로 완전히 수렴하지 않았다
