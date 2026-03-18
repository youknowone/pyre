# majit vs rpython Line-by-Line Comparison Report

## 요약: 파일별 구현 완성도

| 파일 | 완성도 | 주요 누락 |
|------|--------|-----------|
| **intutils.rs** | 95% | make_guards, getnullness, RPython type wrappers |
| **intbounds.rs** | 70% | pure_from_args 합성, CALL_PURE_I/oopspec, string/field bounds, backward propagation dispatcher |
| **rewrite.rs** | 50% | 가드 최적화 10종, loop invariant, arraycopy/move, INT_PY_DIV/MOD, postprocess 메커니즘 |
| **pure.rs** | 40% | OVF 처리, constant folding, preamble ops, descriptor-based caching, COND_CALL_VALUE, short preamble |
| **heap.rs** | 45% | DICT_LOOKUP, postponed ops, exception guards, effect-info invalidation, variable-index array caching, serialization |
| **virtualize.rs** | 60% | GUARD_NO_EXCEPTION, CALL_MAY_FORCE, COND_CALL, JIT_FORCE_VIRTUAL, INT_ADD raw slice, GETARRAYITEM_RAW |
| **simplify.rs** | 85% | GUARD_FUTURE_CONDITION |
| **guard.rs** | 50% | Guard implication (numeric range), transitive loop bounds (다른 목적: Python=vector, Rust=general) |
| **unroll.rs** | 30% | UnrollOptimizer wrapper, preamble/bridge optimization, virtual state, short preamble inlining, export/import state |
| **shortpreamble.rs** | 25% | PreambleOp, HeapOp, PureOp, LoopInvariantOp, ShortBoxes, ExtendedShortPreambleBuilder |
| **bridgeopt.rs** | 40% | serialize/deserialize_optimizer_knowledge (다른 접근: Python=직렬화, Rust=정적 범위 분석) |
| **vstring.rs** | 35% | Unicode, oopspec call handlers (STR_CONCAT/SLICE/EQUAL/CMP), postprocessing, string forcing to constants |
| **info.rs** | 30% | Guard tracking, force mechanics, visitor pattern, string optimization, float constants, preamble ops |
| **optimizer.rs** | 40% | Resume data, guard sharing/emission, constant folding, speculative ops, trace iteration, call_pure_results |
| **virtualstate.rs** | 40% | Runtime value inspection, force_boxes, renum tracking, CPU backend calls, lenbound, make_inputargs |
| **vector.rs** | 30% | Loop unrolling, guard analysis, accumulation packs, pack merging, chain following |
| **warmstate.rs** | 55% | JC_FORCE_FINISH, 12+ set_param_* methods, vectorization params, dynamic JitCell subclass, callbacks |
| **counter.rs** | 50% | JitCell chain, compute_threshold, fetch_next_hash, parameterized decay, GC integration |
| **heapcache.rs** | 35% | Array caching, quasi-immutable tracking, loop invariant caching, dependency tracking, version system |
| **opencoder.rs** | 20% | Snapshot management, trace iterator, constant pooling, live/dead range analysis |
| **pyjitpl.rs** | 다른 설계 | Python=해석 기반(84 methods), Rust=컴파일 실행 기반(93 methods). 해석 루프/연산 기록 없음, 대신 백엔드 통합/브릿지 관리 추가 |
| **blackhole.rs** | 60% | 제어 흐름(goto_if_not), 예외(raise/reraise), 문자열 ops, 포탈/재귀 호출. Vector ops는 Rust만 있음 |
| **resume.rs** | 45% | Reader/Decoder 클래스, 문자열 virtual info 6종, VectorInfo, NumberingState, VirtualCache. Rust는 encoding/decoding 전용 |
| **jitcode.rs** | 다른 설계 | Python=데이터 컨테이너(167줄), Rust=완전한 해석기+빌더(2810줄). Rust가 훨씬 포괄적 |
| **virtualizable.rs** | 75% | Python의 finish() 그래프 변환, 동적 JitCell subclass. Rust는 테스트 24개, 헬퍼 함수 14개 추가 |
| **virtual_ref.rs** | 20% | force_virtual(), virtual_ref_during_tracing(), tracing callbacks, graph transformation 전부 누락. 상수/구조체만 존재 |
| **quasi_immut.rs** | 50% | QuasiImmutDescr, CPU 통합, 디버그/통계, memory compression. Rust는 Arc/AtomicBool 기반 |
| **resoperation.rs** | 99% | 238개 opcode variant 완벽 일치. FloatFloorDiv만 Rust 전용 추가 |
| **descr.rs** | 30% | Descriptor tracking (read/write), bitstring optimization, can_invalidate/can_collect, CallInfoCollection, graph analyzers |
| **codegen lib.rs** | 다른 설계 | Python=90개 메서드 추상 클래스, Rust=20개 메서드 trait. bh_* 헬퍼 30개 누락. Rust에 exit layout/version 시스템 추가 |

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
**opencoder (20%):** 기본 LEB128 인코딩만, snapshot/trace iterator/constant pooling 미구현
