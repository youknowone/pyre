# majit

**M**eta-tr**A**cing **JIT** compiler framework — or, if you prefer, **Ma**gical **JIT**.

majit is a Rust port of [RPython's JIT infrastructure](https://rpython.readthedocs.io/en/latest/jit/index.html). Given an interpreter written in Rust, majit automatically generates a tracing JIT compiler for it.

## What it does

Write a bytecode interpreter. Annotate it with `#[jit_interp]`. majit does the rest:

```rust
#[jit_interp(state = State, env = Program, ...)]
fn mainloop(program: &Program, state: &mut State, driver: &mut JitDriver<State>) {
    while pc < program.len() {
        jit_merge_point!(driver, program, pc);
        match program[pc] {
            Op::Add => { /* ... */ }
            Op::Jump(target) => {
                pc = target;
                can_enter_jit!(driver, target, state, ...);
                continue;
            }
            // ...
        }
        pc += 1;
    }
}
```

Hot loops are detected, traced, optimized, and compiled to native code via Cranelift. Guard failures fall back to the interpreter transparently.

## RPython과의 공통점

majit과 RPython JIT는 동일한 핵심 아이디어를 공유합니다:

- **Meta-tracing**: 인터프리터 자체를 트레이싱하여 바이트코드 수준이 아닌 인터프리터 실행 수준에서 최적화
- **Guard-based speculation**: 타입/값 가정을 guard로 기록하고, 실패 시 interpreter로 deopt
- **Trace → Optimize → Compile → Execute 파이프라인**
- **8-pass optimizer**: IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap
- **Escape analysis**: 가상 객체 할당 제거 (NEW → field tracking → force on escape)
- **Resume/blackhole deoptimization**: guard 실패 시 interpreter 상태 복원

## RPython과의 차이점

### 언어 모델

RPython은 **제한된 Python 서브셋**을 컴파일 타임에 C로 번역합니다. annotator가 타입을 추론하고, rtyper가 low-level 타입으로 변환합니다. JIT 힌트(`jit_merge_point`, `promote`, `elidable` 등)는 Python 코드에 직접 삽입합니다.

majit은 **일반 Rust**에서 동작합니다. 타입 추론이 필요 없고 (Rust 컴파일러가 이미 처리), JIT 힌트는 proc-macro attribute로 제공합니다:

| RPython | majit |
|---------|-------|
| `@jit.elidable` | `#[elidable]` |
| `@jit.dont_look_inside` | `#[dont_look_inside]` |
| `jit.JitDriver(greens=[...], reds=[...])` | `#[jit_driver(greens = [...], reds = [...])]` |
| `driver.jit_merge_point(...)` | `jit_merge_point!(driver, ...)` |
| `driver.can_enter_jit(...)` | `can_enter_jit!(driver, ...)` |

### 번역 vs 분석

RPython의 codewriter는 RPython 소스를 **완전히 번역**하여 JitCode(바이트코드)를 생성합니다. 번역 시 모든 타입과 제어 흐름이 확정됩니다.

majit도 **build-time에 trace 코드를 자동 생성하여 삽입**합니다. 두 가지 경로가 있습니다:

1. **`majit-analyze` 기반 (pyre-mjit)**: `build.rs`에서 `majit_analyze::analyze_multiple()`로 인터프리터 소스를 분석하고, `generate_trace_code()`로 trace helper 함수를 생성하여 `OUT_DIR/jit_trace_gen.rs`에 씁니다. 메인 크레이트가 `include!`로 가져옵니다. opcode dispatch arm 추출, cross-file trait impl 해석, helper 분류, 타입 레이아웃 추출을 수행합니다.

2. **`#[jit_interp]` proc-macro 기반 (aheui-mjit)**: `build.rs`가 인터프리터 소스의 opcode match arm을 추출하고, `#[jit_interp]`가 붙은 JIT mainloop을 자동 생성합니다. `while`/`loop`는 branch bytecode로, `match`는 guard chain으로, `for` 루프는 abort fallback (RPython의 `@dont_look_inside` 상당)으로 변환합니다.

두 경로 모두 **자동 생성·삽입은 완성**되어 있습니다. RPython과의 남은 차이는 생성 기능의 **일반성** — 더 많은 interpreter shape와 complex CFG를 직접 lowering하는 breadth입니다.

### 백엔드

RPython은 x86, ARM, AArch64, s390x, PPC용 **수작업 어셈블러 백엔드** 6개를 유지합니다 (~300K LOC).

majit은 [Cranelift](https://cranelift.dev/) 하나로 모든 플랫폼을 지원합니다. Cranelift이 ISA별 코드 생성, 레지스터 할당, 명령어 선택을 처리하므로 백엔드 코드가 크게 줄어듭니다.

### GC

RPython의 incminimark GC는 JIT와 깊이 통합되어 있고, lltype 기반의 저수준 메모리 모델을 사용합니다.

majit의 GC(`majit-gc`)는 동일한 알고리즘(nursery + oldgen + incremental marking + card marking)을 구현하지만, Rust의 소유권 모델 위에서 동작합니다. JIT-GC 통합 훅(`jit_remember_young_pointer`, `gc_step`, `pin`/`unpin`)도 제공합니다.

### SIMD

RPython의 벡터화는 SSE/AVX를 직접 타겟팅합니다.

majit은 Cranelift의 `I64X2`/`F64X2` SIMD 타입을 사용하여 플랫폼 독립적인 벡터화를 제공합니다. 의존성 그래프 분석, pack group 감지, 비용 모델, instruction scheduling을 포함합니다.

## Crate 구조

```
majit/                          # facade crate (re-exports all below)
├── majit-ir/                   # IR: OpCode, Type, Value, Descr traits
├── majit-opt/                  # Optimizer: 8-pass pipeline + auxiliaries
├── majit-trace/                # Tracing: hot counter, recorder, warm state
├── majit-codegen/              # Backend abstraction: Backend trait
├── majit-codegen-cranelift/    # Cranelift backend: native code generation
├── majit-meta/                 # Meta-interpreter: JitDriver, MetaInterp, resume
├── majit-gc/                   # GC: nursery, oldgen, incremental, card marking
├── majit-macros/               # Proc macros: #[jit_driver], #[jit_interp], etc.
├── majit-runtime/              # Runtime: jit_merge_point!, can_enter_jit!
├── majit-analyze/              # Static analyzer: source → trace code generation
└── examples/                   # 8 toy interpreters (tlr, tl, tla, tiny2, ...)
```

## 실제 성능

| 인터프리터 | 프로그램 | Interpreter | JIT | 속도 향상 |
|-----------|---------|------------|-----|----------|
| aheuijit | logo.aheui | 6.1초 | 0.05초 | **110x** |
| pyre | fib(20) | — | 정확한 결과 | JIT 동작 확인 |

## 라이선스

MIT
