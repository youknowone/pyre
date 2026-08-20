//! An `array_fields` base spelled as a `ref(T)` state scalar.
//!
//! `array_fields = { S::f => E }` says the field `f` holds a buffer BASE
//! POINTER, so `<obj>.f[i]` is a `getfield_gc_r` of the base followed by a
//! `get/setarrayitem_gc` on the element. The vocabulary is accepted by both
//! `#[jit_inline]` and `#[jit_interp]`, but `<obj>` can be spelled two ways and
//! only one of them was ever resolved:
//!
//! * a local `ref_params` binding — already materialized in a register, and
//! * `state.<ref_scalar>` — a `ref(T)` state field, which has to be READ out of
//!   the state frame first, and that read emits.
//!
//! Both halves of the lowering handled only the first. On the JIT path
//! `match_array_field_base` required an `Expr::Path`, so the shape matched
//! nothing and the whole arm degraded to an abort stub; a dispatch whose arms
//! all degrade leaves its switch pointing at blocks that were never emitted.
//! On the concrete path `RefFieldRewriter` had no `Expr::Index` arm at all, so
//! the plain-field arms rewrote `<base>.<field>` on their own and left a raw
//! pointer being indexed with `[]` — which does not compile.
//!
//! That second failure mode is why this fixture is worth its length: the
//! concrete half regresses as a BUILD failure, so a test that merely ran the
//! machine would never get the chance to report it. What needs asserting is the
//! JIT half, and the census below is what separates "the trace lowered the
//! element access" from "the arm degraded and the concrete fallback ran and
//! happened to agree". A degraded arm still returns the right number.
//!
//! The read and the write reach the base through two different call sites, so
//! both are exercised. Getting a read to survive at all takes care: at a fixed
//! index it is loop-invariant and gets hoisted, and immediately after a write
//! to the same index it is answered from the store. The machine below walks its
//! index instead, which is also the only arrangement under which the base read
//! has to be redone rather than folded.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;
use majit_metainterp::JitDriver;

/// The optimized opcode list of the last loop compiled.
static COMPILED: Mutex<Vec<OpCode>> = Mutex::new(Vec::new());
static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// `data` is deliberately NOT the first member: a base read that ignores the
/// field's own offset lands on `guard` and indexes a sentinel, which the
/// element assertions below would catch. With `data` at offset 0 the wrong
/// answer and the right one coincide.
#[repr(C)]
struct ElemStack {
    guard: i64,
    data: *mut i64,
}

struct WalkState {
    /// `ref(ElemStack)` — the address of the holder, not of the buffer.
    stack: usize,
    /// The element index. It advances every iteration, so the read below is
    /// neither loop-invariant nor answerable from the write that follows it.
    sp: i64,
    acc: i64,
    ticks: i64,
}

pub type Bytecode = [u8];

/// `acc += stack.data[sp]` — reads what the buffer was seeded with
const OP_LOAD: u8 = 1;
/// `stack.data[sp] = ticks` — overwrites the slot just read
const OP_STORE: u8 = 2;
/// `sp += 1`
const OP_STEP: u8 = 3;
/// `ticks -= 1`; back edge to 0 while non-zero
const OP_TICK: u8 = 4;
const OP_HALT: u8 = 5;

#[majit_macros::jit_interp(
    state = WalkState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        stack: ref(ElemStack),
        sp: int,
        acc: int,
        ticks: int,
    },
    array_fields = { ElemStack::data => i64 },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_walk(program: &Bytecode, threshold: u32, stack: usize, ticks: i64) -> i64 {
    let mut driver: JitDriver<WalkState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
        *COMPILED.lock().unwrap() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = WalkState {
        stack,
        sp: 0i64,
        acc: 0i64,
        ticks,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_LOAD => {
                state.acc = state.acc + state.stack.data[state.sp];
            }
            OP_STORE => {
                state.stack.data[state.sp] = state.ticks;
            }
            OP_STEP => {
                state.sp = state.sp + 1i64;
            }
            OP_TICK => {
                state.ticks = state.ticks - 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            OP_HALT => break,
            _ => break,
        }
    }
    state.acc
}

const PROGRAM: [u8; 5] = [OP_LOAD, OP_STORE, OP_STEP, OP_TICK, OP_HALT];

/// How many iterations the machine runs, and how far `sp` therefore walks.
const TICKS: i64 = 200;
/// Enough slots that `sp` never leaves the buffer, so the machine needs no wrap
/// branch and the trace stays a single un-bridged loop.
const SLOTS: usize = 256;

struct Fixture {
    buf: Vec<i64>,
    holder: Box<ElemStack>,
}

impl Fixture {
    /// Slot `i` is seeded with `i + 1`, so a read from the wrong slot returns a
    /// wrong sum rather than the same one.
    fn new() -> Self {
        let mut buf: Vec<i64> = (0..SLOTS as i64).map(|i| i + 1).collect();
        let base = buf.as_mut_ptr();
        Self {
            buf,
            holder: Box::new(ElemStack {
                guard: i64::MIN,
                data: base,
            }),
        }
    }

    fn holder_addr(&mut self) -> usize {
        &mut *self.holder as *mut ElemStack as usize
    }
}

/// Iteration `i` reads the seed at slot `i` and writes `TICKS - i` over it.
fn expected_acc() -> i64 {
    (1..=TICKS).sum()
}

/// The machine answers the same warm and cold, and its writes land in the
/// buffer the field points at.
///
/// The cold arm is not decoration: it is the only statement of what the right
/// answer IS. Comparing the warm run against a constant would grade the JIT
/// against this file's arithmetic instead of against the interpreter.
#[test]
fn an_array_field_under_a_state_ref_base_agrees_warm_and_cold() {
    let mut cold = Fixture::new();
    let addr = cold.holder_addr();
    let cold_acc = dispatch_walk(&PROGRAM, u32::MAX, addr, TICKS);

    let mut warm = Fixture::new();
    let addr = warm.holder_addr();
    let warm_acc = dispatch_walk(&PROGRAM, 8, addr, TICKS);

    assert_eq!(cold_acc, expected_acc(), "the cold arm must interpret it");
    assert_eq!(
        warm_acc, cold_acc,
        "a compiled element access answered differently from the interpreter",
    );

    let touched: Vec<i64> = (0..TICKS).map(|i| TICKS - i).collect();
    assert_eq!(
        &warm.buf[..TICKS as usize],
        touched.as_slice(),
        "each visited slot must hold the tick written to it",
    );
    let untouched: Vec<i64> = (TICKS..SLOTS as i64).map(|i| i + 1).collect();
    assert_eq!(
        &warm.buf[TICKS as usize..],
        untouched.as_slice(),
        "the machine never walks past slot {TICKS}; a stride off by a word \
         would have disturbed a slot beyond it",
    );
    assert_eq!(
        warm.holder.guard,
        i64::MIN,
        "the element write must go through the buffer, not into the holder",
    );
}

/// The compiled trace must carry BOTH element accesses as array ops.
///
/// This is the assertion that fails on the regressed JIT half. When
/// `match_array_field_base` cannot resolve a `state.<ref_scalar>` base the arm
/// degrades to an abort stub, and the machine still returns the right number
/// through the concrete path — so the check above passes and grades nothing.
///
/// The read and the write are asserted separately because they reach the base
/// through separate call sites: `lower_ref_binding_array_read` and
/// `lower_ref_binding_array_write`. Either one alone would pass this test on a
/// build where the other still could not resolve the base.
#[test]
fn the_compiled_trace_lowers_both_element_accesses_to_array_ops() {
    let mut fixture = Fixture::new();
    let addr = fixture.holder_addr();
    let before = COMPILES.load(Ordering::Relaxed);
    let acc = dispatch_walk(&PROGRAM, 8, addr, TICKS);
    assert_eq!(acc, expected_acc());
    assert!(
        COMPILES.load(Ordering::Relaxed) > before,
        "no trace compiled, so the census below would read zero vacuously",
    );

    let loop_ops = COMPILED.lock().unwrap().clone();
    let count = |wanted: OpCode| loop_ops.iter().filter(|op| **op == wanted).count();
    let sets = count(OpCode::SetarrayitemGc);
    let gets = count(OpCode::GetarrayitemGcI);
    let bases = count(OpCode::GetfieldGcR);
    assert!(
        bases > 0,
        "the buffer base must be read out of the holder as a pointer field; \
         loop was {loop_ops:?}",
    );
    assert!(
        sets > 0,
        "the element write must survive as a setarrayitem; an arm that degraded \
         to an abort stub leaves none. Loop was {loop_ops:?}",
    );
    assert!(
        gets > 0,
        "the element read must survive as a getarrayitem; an arm that degraded \
         to an abort stub leaves none. Loop was {loop_ops:?}",
    );
}
