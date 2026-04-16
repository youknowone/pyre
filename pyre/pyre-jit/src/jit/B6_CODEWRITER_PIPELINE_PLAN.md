# B6: codewriter 파이프라인 재구성 계획

## 문제

pyre의 `CodeWriter::transform_graph_to_jitcode` (`codewriter.rs`) 는 CPython
바이트코드를 직접 순회하면서 `JitCodeBuilder` 메서드를 호출합니다. RPython은 다섯
단계로 분리된 파이프라인입니다:

```
transform_graph(graph)          ← jtransform instruction selection
  → regalloc(graph, kind)       ← register allocation per kind
  → flatten_graph(graph, regallocs) → SSARepr
  → compute_liveness(ssarepr)   ← backward dataflow on SSARepr
  → assembler.assemble(ssarepr) → JitCode
```

이 분리가 주는 이점:
- liveness는 JitCode IR 수준에서 계산 (Python bytecode 수준이 아님)
- `-live-`를 guard/residual-call/exception-edge에만 배치 (B3 해소)
- register allocation이 독립 패스 (coloring, spill 가능)
- assembler가 SSARepr에만 의존 (입력 불가지론)

## 전략

pyre의 입력은 CPython stack-machine 바이트코드이고 RPython의 입력은 SSA-like flow
graph입니다. 입력이 다르므로 RPython의 `transform_graph` + `flatten_graph`를
그대로 포팅하는 것은 불가능합니다. 대신 pyre의 바이트코드 순회를 **두 패스**로
분리합니다:

1. **flatten pass**: CPython bytecode → `SSARepr` (명령 tuple 리스트)
2. **liveness pass**: `compute_liveness(ssarepr)` (backward dataflow)
3. **assemble pass**: `assembler.assemble(ssarepr)` → `JitCode`

register allocation은 현재의 stack-depth 기반 할당을 유지합니다 — 이것은
PRE-EXISTING-ADAPTATION이며, RPython의 graph coloring과 다르지만 stack-machine
입력에서는 불가피합니다.

## 데이터 구조

### SSARepr (rpython/jit/codewriter/flatten.py:6-10)

```rust
// pyre/pyre-jit/src/jit/ssarepr.rs (신규)
pub struct SSARepr {
    pub name: String,
    pub insns: Vec<Insn>,
}
```

### Insn (flatten.py instruction tuple의 Rust 표현)

RPython의 instruction tuple은 `(opname, arg1, ..., argN)` 형태입니다.
Rust에서는 enum + Vec으로 표현합니다:

```rust
pub enum Insn {
    /// `-live-` placeholder — expanded by compute_liveness
    Live(Vec<Operand>),
    /// `('---')` — unreachable marker (clears alive set)
    Unreachable,
    /// Label marker (flatten.py Label)
    Label(LabelId),
    /// Regular instruction: (opname, args...)
    Op {
        name: &'static str,
        args: Vec<Operand>,
    },
}

pub enum Operand {
    Reg(RegKind, u16),        // Register(kind, index)
    ConstInt(i64),            // Constant(Signed)
    ConstRef(i64),            // Constant(GCREF)
    ConstFloat(i64),          // Constant(Float)
    TLabel(LabelId),          // TLabel(name) — forward ref
    ListOfKind(RegKind, Vec<Operand>),
}

pub type LabelId = usize;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegKind { Int, Ref, Float }
```

### 현재 JitCodeBuilder 메서드 → SSARepr 매핑

현재 codewriter가 직접 호출하는 JitCodeBuilder 메서드와 SSARepr Insn의 대응:

| JitCodeBuilder 메서드 | SSARepr Insn |
|---|---|
| `assembler.move_r(dst, src)` | `Op { name: "move_r", args: [Reg(Ref,dst), Reg(Ref,src)] }` |
| `assembler.load_const_i_value(dst, val)` | `Op { name: "load_const_i", args: [Reg(Int,dst), ConstInt(val)] }` |
| `assembler.call_ref_typed(fn_idx, args, dst)` | `Op { name: "call_ref", args: [ConstInt(fn_idx), ...args, Reg(Ref,dst)] }` |
| `assembler.call_may_force_ref_typed(...)` | `Op { name: "call_may_force_ref", args: [...] }` |
| `assembler.vable_getarrayitem_ref(...)` | `Op { name: "getarrayitem_vable_r", args: [...] }` |
| `assembler.live_placeholder()` | `Live(vec![])` (placeholder, no args yet) |
| `assembler.mark_label(id)` | `Label(id)` |
| `assembler.jump(target)` | `Op { name: "goto", args: [TLabel(target)] }` |
| `assembler.jit_merge_point(gi, gr, rr)` | `Op { name: "jit_merge_point", args: [...] }` |

## 단계별 실행 계획

### Phase 1: SSARepr 정의 + 기존 테스트 유지 (파일 신규)

**파일**: `pyre/pyre-jit/src/jit/ssarepr.rs`

SSARepr, Insn, Operand, RegKind 타입 정의. 빌드 확인만.

**검증**: `cargo build --workspace --features dynasm`

### Phase 2: flatten pass 추출

**변경**: `codewriter.rs`의 `transform_graph_to_jitcode` 내부 바이트코드 순회를
두 단계로 분리:

```
fn flatten_bytecode(code, ...) -> SSARepr     // Phase A: 기존 순회 로직
fn assemble_ssarepr(ssarepr, ...) -> JitCode  // Phase B: JitCodeBuilder 소비
```

Phase A는 기존 `assembler.move_r()` 등의 호출을 `ssarepr.insns.push(Insn::Op{...})`
로 교체합니다. Phase B는 SSARepr을 읽어 JitCodeBuilder를 호출합니다.

이 단계에서 JitCode 출력은 **비트 동일**해야 합니다.

**검증**: `./pyre/check.sh` 14/14 + 14/14

### Phase 3: compute_liveness 포팅

**파일**: `pyre/pyre-jit/src/jit/liveness.rs` (신규)

RPython `liveness.py:19-80`의 backward dataflow를 SSARepr 위에 구현:

```rust
pub fn compute_liveness(ssarepr: &mut SSARepr) {
    let mut label2alive: HashMap<LabelId, HashSet<(RegKind, u16)>> = HashMap::new();
    loop {
        if !compute_liveness_one_pass(ssarepr, &mut label2alive) {
            break;
        }
    }
    remove_repeated_live(ssarepr);
}
```

이 단계에서 `-live-` placeholder가 실제 live register set으로 확장됩니다.

**검증**: unit test + `./pyre/check.sh`

### Phase 4: -live- 배치 정책 변경 (B3 해소)

현재: 모든 Python PC에 `-live-` 배치
목표: guard / residual-call / exception-edge 지점에만

이것은 Phase 2의 flatten pass에서 `Live(vec![])` 를 emit하는 위치를 변경하는
것입니다. RPython의 `flatten.py`에서 `-live-`를 emit하는 위치:
- `insert_call_pop_alive(ssarepr)` — residual call 후
- guard opcode 직전
- exception handler entry
- `jit_merge_point` 직전

**검증**: `./pyre/check.sh` — resume encode/decode가 일치해야 함

### Phase 5: LiveVars fallback 제거

Phase 3-4가 안정화되면 `pyre_jit_trace::state.rs`와 `trace_opcode.rs`의
`LiveVars` fallback 경로를 제거할 수 있습니다. 모든 liveness가 SSARepr 기반
compute_liveness에서 나옵니다.

**검증**: `./pyre/check.sh` + fallback path가 hit하면 panic 삽입 후 확인

## 위험 요소

1. **resume encode/decode 정합성**: capture side (get_list_of_active_boxes)와
   decode side (consume_one_section)가 같은 liveness 데이터를 써야 합니다.
   Phase 4에서 `-live-` 위치가 바뀌면 양쪽을 동시에 바꿔야 합니다.

2. **exception landing pad**: pyre는 CPython exception table에서 landing block을
   합성합니다 (B4). Phase 2에서 이것도 SSARepr로 표현해야 합니다.

3. **성능 회귀**: Phase 2에서 중간 Vec 할당 추가. 허용 가능 (parity 우선).

4. **JitCodeBuilder 인터페이스 유지**: `assembler.rs`(majit-metainterp)의
   JitCodeBuilder는 Phase 2-B에서 SSARepr consumer로 래핑됩니다. 기존
   JitCodeBuilder 메서드는 그대로 유지.

## 파일 변경 요약

| 파일 | 변경 |
|---|---|
| `pyre/pyre-jit/src/jit/ssarepr.rs` | 신규 — SSARepr, Insn, Operand |
| `pyre/pyre-jit/src/jit/liveness.rs` | 신규 — compute_liveness |
| `pyre/pyre-jit/src/jit/codewriter.rs` | flatten / assemble 분리 |
| `pyre/pyre-jit/src/jit/mod.rs` | 모듈 등록 |

## 진행 상태 (2026-04-16)

- **Phase 1 ✓** — `flatten.rs` 완료. `SSARepr`, `Label`, `TLabel`, `Register`,
  `ListOfKind`, `Insn`, `Operand`, `Kind` 모두 `flatten.py:1-60` line-by-line
  대응. 4개 단위 테스트 통과.
- **Phase 2 ✓** — `liveness.rs` 완료. `compute_liveness`,
  `_compute_liveness_must_continue`, `remove_repeated_live` 모두
  `liveness.py:19-116` line-by-line 대응. 5개 단위 테스트 통과.
- **Phase 3a ✓** — `assembler.rs` skeleton 완료. `assemble(ssarepr, ...)` +
  `write_insn` + `dispatch_op` 프레임워크. 현재 dispatch 테이블은 비어있고,
  unknown opname에 panic — 회귀 방지용 가드.
- **Phase 3b (진행 중)** — pyre의 bytecode walker (`codewriter.rs` 약 700줄)를
  `assembler.xxx()` 직접 호출에서 `Insn::Op` 생성으로 전환. 각 핸들러마다:
  1. `Insn::Op { opname, args, result }` 를 `SSARepr.insns` 에 push
  2. `assembler.rs`의 `dispatch_op` 에 `opname` match arm 추가
  3. 출력 `JitCode` bit-identical 검증
- **Phase 3c (예정)** — Phase 3b 완료 후 `transform_graph_to_jitcode` 의 최종
  형태: `flatten(code) -> SSARepr` + `assemble(ssarepr, ...)`. 중간 단계에서는
  SSARepr 생성과 assembler 직접 호출을 병행할 수 있음.
- **Phase 4 (예정)** — `compute_liveness(ssarepr)` 을 pyre의 기존 `LiveVars`
  기반 liveness 경로와 교체. `-live-` emit 위치를 모든 Python PC에서
  guard/residual-call/exception-edge 로 한정. B3 해소.
- **Phase 5 (예정)** — `LiveVars` fallback 제거 (`pyre_jit_trace::state.rs`,
  `trace_opcode.rs`). SSARepr 기반 경로가 유일한 liveness 출처.

## Phase 3b 진행 방법 (다음 세션)

각 bytecode 핸들러를 하나씩 포팅합니다. 단순한 것부터 시작:

1. **RETURN_VALUE / RETURN_CONST** (가장 단순, `ref_return/r`):
   ```rust
   // 기존: assembler.ref_return(src_reg);
   // 신규: ssarepr.insns.push(Insn::op("ref_return", vec![Operand::reg(Kind::Ref, src_reg)]));
   // dispatch: "ref_return" => assembler.ref_return(expect_reg(&args[0], Kind::Ref))
   ```

2. **LOAD_FAST / STORE_FAST** (register move + vable 쓰기):
   - 비-portal: `move_r(dst_stack, src_local)` + depth 증감
   - portal: vable getarrayitem/setarrayitem + move_r

3. **LOAD_SMALL_INT**: `load_const_i` + `call_ref` (box_int)

4. **BINARY_OP**: `move_r` x 2 + `load_const_i` + `call_may_force_ref`

5. **POP_JUMP_IF_FALSE**: `move_r` + `call_int` (truth) + branch + jump

6. ... 나머지 핸들러들

각 핸들러당 commit 하나. PR은 누적 완료 후.

## 위험 요소

1. **resume encode/decode 정합성**: capture side (get_list_of_active_boxes)와
   decode side (consume_one_section)가 같은 liveness 데이터를 써야 합니다.
   Phase 4에서 `-live-` 위치가 바뀌면 양쪽을 동시에 바꿔야 합니다.

2. **exception landing pad**: pyre는 CPython exception table에서 landing block을
   합성합니다 (B4). Phase 2에서 이것도 SSARepr로 표현해야 합니다.

3. **성능 회귀**: Phase 2에서 중간 Vec 할당 추가. 허용 가능 (parity 우선).

4. **JitCodeBuilder 인터페이스 유지**: `assembler.rs`(majit-metainterp)의
   JitCodeBuilder는 Phase 2-B에서 SSARepr consumer로 래핑됩니다. 기존
   JitCodeBuilder 메서드는 그대로 유지.
