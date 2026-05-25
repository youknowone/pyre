//! Operation execution for the blackhole interpreter.
//!
//! Mirrors RPython's `executor.py`: executes individual JIT IR operations
//! by dispatching on the opcode and computing the result.

use majit_ir::{GcRef, Op, OpCode, OpRef};

use crate::blackhole::ExceptionState;

/// llgraph/runner.py:1245-1251 `execute_guard_class` parity helper:
/// mirrors `gc_ll_descr.get_actual_typeid(gcptr)` (gc.py:624-629)
/// through the thread-local GC seam installed by the active backend.
///
/// For managed objects the active callback reads the GC header word
/// and returns its typeid half-word; for pyre's foreign PyObject
/// layout it consults the `vtable_to_type_id` table populated via
/// `register_vtable_for_type`. Returns `None` when no backend is
/// installed (unit tests dispatching with placeholder integers), which
/// the caller translates into `executor.py:351-358` skip semantics.
fn read_typeid(obj_ptr: i64) -> Option<u32> {
    if obj_ptr <= 0 {
        return None;
    }
    let addr = obj_ptr as usize;
    if addr < 4096 {
        return None;
    }
    majit_gc::get_actual_typeid(GcRef(addr))
}

/// Fast value store for trace execution.
///
/// Op results (non-constant OpRef) → `results` Vec, direct indexed.
/// Constants (constant-namespace OpRef) → `constants` Vec, indexed by const_index.
///
/// Replaces `VecAssoc<u32, i64>` on the hot path with O(1) Vec indexing.
pub(crate) struct TraceValues {
    /// Op results, indexed by OpRef.0 (operation namespace).
    pub results: Vec<i64>,
    /// Constants, indexed by OpRef.const_index() (constant namespace).
    pub constants: Vec<i64>,
}

impl TraceValues {
    pub fn new(num_ops: usize, num_constants: usize) -> Self {
        Self {
            results: vec![0; num_ops],
            constants: vec![0; num_constants],
        }
    }

    pub fn from_vec_assoc(map: &crate::optimizeopt::vec_assoc::VecAssoc<u32, i64>) -> Self {
        // Index-keyed pool namespace probe (Slice P3 category E):
        // raw u32 keys carry the constant-namespace bit directly, so use
        // the bit-helpers rather than minting a typed `OpRef` solely
        // for the namespace test.
        let max_op = map
            .keys()
            .filter(|&&k| !OpRef::raw_is_constant(k))
            .max()
            .copied()
            .unwrap_or(0) as usize;
        let max_const = map
            .keys()
            .filter(|&&k| OpRef::raw_is_constant(k))
            .max()
            .map(|&k| OpRef::raw_const_index(k) as usize)
            .unwrap_or(0);
        let mut tv = Self::new(max_op + 1, max_const + 1);
        for (&k, &v) in map {
            tv.set(k, v);
        }
        tv
    }

    #[inline(always)]
    pub fn get(&self, idx: u32) -> i64 {
        if OpRef::raw_is_constant(idx) {
            let ci = OpRef::raw_const_index(idx) as usize;
            if ci < self.constants.len() {
                self.constants[ci]
            } else {
                0
            }
        } else {
            let i = idx as usize;
            if i < self.results.len() {
                self.results[i]
            } else {
                0
            }
        }
    }

    #[inline(always)]
    pub fn set(&mut self, idx: u32, value: i64) {
        if OpRef::raw_is_constant(idx) {
            let ci = OpRef::raw_const_index(idx) as usize;
            if ci >= self.constants.len() {
                self.constants.resize(ci + 1, 0);
            }
            self.constants[ci] = value;
        } else {
            let i = idx as usize;
            if i >= self.results.len() {
                self.results.resize(i + 1, 0);
            }
            self.results[i] = value;
        }
    }

    #[inline(always)]
    pub fn resolve(&self, opref: OpRef) -> i64 {
        self.get(opref.raw())
    }
}

/// Trait for resolving OpRef → i64 values in trace execution.
/// Allows both VecAssoc (legacy) and TraceValues (fast) backends.
pub(crate) trait ValueStore {
    fn resolve(&self, opref: OpRef) -> i64;
}

impl ValueStore for crate::optimizeopt::vec_assoc::VecAssoc<u32, i64> {
    #[inline(always)]
    fn resolve(&self, opref: OpRef) -> i64 {
        self.get(&opref.raw()).copied().unwrap_or(0)
    }
}

impl ValueStore for TraceValues {
    #[inline(always)]
    fn resolve(&self, opref: OpRef) -> i64 {
        self.get(opref.raw())
    }
}

pub(crate) enum OpResult {
    Value(i64),
    Void,
    Finish(Vec<OpRef>),
    Jump(Vec<OpRef>),
    GuardFailed,
    Unsupported(String),
}

pub(crate) fn execute_one(
    op: &Op,
    values: &(impl ValueStore + ?Sized),
    exc: &mut ExceptionState,
) -> OpResult {
    match op.opcode {
        // ── Control flow ──
        OpCode::Label => OpResult::Void,
        OpCode::Finish => OpResult::Finish(op.getarglist().to_vec()),
        OpCode::Jump => OpResult::Jump(op.getarglist().to_vec()),

        // ── Integer arithmetic ──
        OpCode::IntAdd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_add(b))
        }
        OpCode::IntSub => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_sub(b))
        }
        OpCode::IntMul => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_mul(b))
        }
        OpCode::IntFloorDiv => {
            let (a, b) = binop(values, op);
            if b == 0 {
                OpResult::Value(0)
            } else {
                OpResult::Value(a.wrapping_div(b))
            }
        }
        OpCode::IntMod => {
            let (a, b) = binop(values, op);
            if b == 0 {
                OpResult::Value(0)
            } else {
                OpResult::Value(a.wrapping_rem(b))
            }
        }
        OpCode::IntAnd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a & b)
        }
        OpCode::IntOr => {
            let (a, b) = binop(values, op);
            OpResult::Value(a | b)
        }
        OpCode::IntXor => {
            let (a, b) = binop(values, op);
            OpResult::Value(a ^ b)
        }
        OpCode::IntLshift => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_shl(b as u32))
        }
        OpCode::IntRshift => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_shr(b as u32))
        }
        OpCode::UintRshift => {
            let (a, b) = binop(values, op);
            OpResult::Value((a as u64).wrapping_shr(b as u32) as i64)
        }
        OpCode::IntNeg => {
            let a = unop(values, op);
            OpResult::Value(a.wrapping_neg())
        }
        OpCode::IntInvert => {
            let a = unop(values, op);
            OpResult::Value(!a)
        }
        OpCode::IntSignext => {
            let (a, b) = binop(values, op);
            // Sign extend from b bytes to i64
            let bits = b * 8;
            let shift = 64 - bits;
            OpResult::Value((a << shift) >> shift)
        }

        // ── Integer comparisons ──
        OpCode::IntLt => {
            let (a, b) = binop(values, op);
            OpResult::Value((a < b) as i64)
        }
        OpCode::IntLe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a <= b) as i64)
        }
        OpCode::IntGe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a >= b) as i64)
        }
        OpCode::IntGt => {
            let (a, b) = binop(values, op);
            OpResult::Value((a > b) as i64)
        }
        OpCode::IntEq => {
            let (a, b) = binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::IntNe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::UintLt => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) < (b as u64)) as i64)
        }
        OpCode::UintLe => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) <= (b as u64)) as i64)
        }
        OpCode::UintGe => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) >= (b as u64)) as i64)
        }
        OpCode::UintGt => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) > (b as u64)) as i64)
        }
        OpCode::IntIsZero => {
            let a = unop(values, op);
            OpResult::Value((a == 0) as i64)
        }
        OpCode::IntIsTrue => {
            let a = unop(values, op);
            OpResult::Value((a != 0) as i64)
        }
        OpCode::IntForceGeZero => {
            let a = unop(values, op);
            OpResult::Value(a.max(0))
        }
        OpCode::IntBetween => {
            // int_between(a, b, c) => a <= b < c
            let a = values.resolve(op.arg(0));
            let b = values.resolve(op.arg(1));
            let c = values.resolve(op.arg(2));
            OpResult::Value((a <= b && b < c) as i64)
        }

        // ── Float operations ──
        OpCode::FloatAdd => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a + b) as i64)
        }
        OpCode::FloatSub => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a - b) as i64)
        }
        OpCode::FloatMul => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a * b) as i64)
        }
        OpCode::FloatTrueDiv => {
            // `blackhole.py:714-718 bhimpl_float_truediv = a / b`: after
            // RPython translation `a / b` is plain C IEEE-754 division,
            // returning `inf` / `nan` on `b == 0.0` with no exception
            // raised at the JIT layer.  Python-level `ZeroDivisionError`
            // is raised by `floatobject.py descr_truediv` *above* the
            // JIT.  Rust's `/` on `f64` matches the same IEEE semantics,
            // so this is structural parity, not an adaptation.
            //
            // `execute_binary_float_const` still declines to fold on
            // `b == 0.0` so the trace emits the op for runtime handling
            // (preserving the IEEE result path through the recorded
            // operation rather than freezing a constant).
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a / b) as i64)
        }
        OpCode::FloatNeg => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(-a) as i64)
        }
        OpCode::FloatAbs => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(a.abs()) as i64)
        }
        OpCode::CastFloatToInt => {
            // `blackhole.py:800-808 bhimpl_cast_float_to_int = int(int(a))`
            // post-translation lowers to `lltype.cast_float_to_int` which
            // is a plain C `(long)f` — implementation-defined / UB on
            // non-finite, no Python exception raised at the JIT layer.
            // Python-level `OverflowError` / `ValueError` come from
            // `floatobject.py descr_int` *above* the JIT.  Rust's
            // `as i64` saturates on non-finite, matching the lenient
            // IEEE-cast policy already in `FloatTrueDiv` above.
            //
            // `execute_cast_const` declines to fold on non-finite so the
            // trace emits the op for runtime handling.
            let a = float_unop(values, op);
            OpResult::Value(a as i64)
        }
        OpCode::CastIntToFloat => {
            let a = unop(values, op);
            OpResult::Value(f64::to_bits(a as f64) as i64)
        }

        // ── Guards ──
        OpCode::GuardTrue | OpCode::VecGuardTrue => {
            let a = unop(values, op);
            if a != 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardFalse | OpCode::VecGuardFalse => {
            let a = unop(values, op);
            if a == 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardValue => {
            let (a, b) = binop(values, op);
            if a == b {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardNonnull => {
            let a = unop(values, op);
            if a != 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardIsnull => {
            let a = unop(values, op);
            if a == 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardClass | OpCode::GuardNonnullClass => {
            // llgraph/runner.py:1247-1255 execute_guard_class /
            //   execute_guard_nonnull_class: value.typeptr == klass
            let (a, b) = binop(values, op);
            if a == b {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardSubclass => {
            // llgraph/runner.py:1271-1281 execute_guard_subclass parity:
            //
            //     value = lltype.cast_opaque_ptr(rclass.OBJECTPTR, arg)
            //     expected_class = ...
            //     if (expected_class.subclassrange_min
            //             <= value.typeptr.subclassrange_min
            //             <= expected_class.subclassrange_max):
            //         pass
            //     else:
            //         self.fail_guard(descr)
            //
            // RPython lowers `a <= b <= c` via `op_int_between` /
            // x86/assembler.py:1975-1979 as the unsigned half-open
            // `(b - a) < (c - a)`; the majit backends emit the same
            // half-open form (`majit-backend-cranelift/src/compiler.rs`
            // :6439-6448, `majit-backend-wasm/src/codegen.rs:978`).
            // The executor matches the assembler / rtyper contract
            // rather than the inclusive bounds in the llgraph fallback,
            // because the executor is the trace re-execution path the
            // bridge / blackhole loop relies on — it must agree with
            // the JIT-compiled side bit-for-bit.
            //
            // Object side: `read_typeid` reads the GC header (managed)
            // or the registered vtable (pyre PyObject seam), then
            // `typeid_subclass_range` resolves the preorder lower
            // bound. Expected side: `subclass_range` translates the
            // constant vtable pointer embedded in the guard arg.
            //
            // RPython has no fallback for "typeid lookup failed" —
            // `cast_opaque_ptr` either succeeds or crashes. majit
            // can't crash on synthetic test inputs, so an unresolved
            // typeid translates to `GuardFailed`: if the executor
            // can't prove the guard succeeds, the safe answer is
            // "deopt to interpretation", never "silently allow".
            let (obj_ptr, expected_classptr) = binop(values, op);
            let value_min = read_typeid(obj_ptr)
                .and_then(|tid| majit_gc::typeid_subclass_range(tid).map(|(min, _)| min));
            let expected = majit_gc::subclass_range(expected_classptr as usize);
            match (value_min, expected) {
                (Some(vm), Some((emin, emax))) => {
                    if emin <= vm && vm < emax {
                        OpResult::Void
                    } else {
                        OpResult::GuardFailed
                    }
                }
                _ => OpResult::GuardFailed,
            }
        }
        OpCode::GuardNoOverflow => {
            if exc.ovf_flag {
                OpResult::GuardFailed
            } else {
                OpResult::Void
            }
        }
        OpCode::GuardOverflow => {
            if !exc.ovf_flag {
                OpResult::GuardFailed
            } else {
                OpResult::Void
            }
        }
        OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
            // In blackhole, check if a call set an exception (simulated force).
            if exc.is_pending() {
                OpResult::GuardFailed
            } else {
                OpResult::Void
            }
        }
        OpCode::GuardNotInvalidated | OpCode::GuardFutureCondition => OpResult::Void,
        OpCode::GuardAlwaysFails => OpResult::GuardFailed,
        OpCode::GuardNoException => {
            if exc.is_pending() {
                OpResult::GuardFailed
            } else {
                OpResult::Void
            }
        }
        OpCode::GuardException => {
            // Guard expects an exception of a specific class.
            // arg(0) is the expected exception class.
            if exc.is_pending() {
                let expected_class = values.resolve(op.arg(0));
                if exc.exc_class == expected_class {
                    // Match — return the exception value and clear exception state.
                    let (_, val) = exc.clear();
                    return OpResult::Value(val);
                }
            }
            OpResult::GuardFailed
        }
        OpCode::GuardGcType => {
            // llgraph/runner.py:1257-1261 execute_guard_gc_type literal:
            //
            //     assert isinstance(typeid, TypeIDSymbolic)
            //     TYPE = arg._obj.container._TYPE
            //     if TYPE != typeid.STRUCT_OR_ARRAY:
            //         self.fail_guard(descr)
            //
            // majit reads `arg`'s typeid from the GC header and
            // compares it against the immediate `typeid` constant
            // carried in arg[1] (rewrite.py emits a `ConstInt(type_id)`
            // there). RPython has no fallback for "typeid unreadable":
            // `arg._obj.container._TYPE` either succeeds or crashes.
            // For majit, an unreadable typeid translates to
            // `GuardFailed` so the bridge / blackhole loop deopts to
            // interpretation rather than silently allowing the guard
            // to pass.
            let (obj_ptr, expected_tid) = binop(values, op);
            match read_typeid(obj_ptr) {
                Some(actual_tid) => {
                    if actual_tid as i64 == expected_tid {
                        OpResult::Void
                    } else {
                        OpResult::GuardFailed
                    }
                }
                None => OpResult::GuardFailed,
            }
        }
        OpCode::GuardIsObject => {
            // llgraph/runner.py:1263-1269 execute_guard_is_object literal:
            //
            //     TYPE = arg._obj.container._TYPE
            //     while TYPE is not rclass.OBJECT:
            //         if not isinstance(TYPE, lltype.GcStruct):
            //             self.fail_guard(descr)
            //             return
            //         _, TYPE = TYPE._first_struct()
            //
            // The walk inspects each `_first_struct` ancestor for
            // GcStruct-ness; the loop returns silently when it
            // reaches `rclass.OBJECT`, otherwise fails the guard.
            // majit's TYPE_INFO table pre-computes the same answer
            // (`T_IS_RPYTHON_INSTANCE`, gctypelayout.py:642), so the
            // equivalent is a single seam lookup. Same fallback rule
            // as the other typeid guards: an unreadable typeid means
            // `GuardFailed`, never `Void`. Earlier majit revisions
            // skipped to `Void` here, which silently degraded the
            // guard into a nonnull check — that's the regression the
            // user flagged.
            let obj_ptr = unop(values, op);
            match read_typeid(obj_ptr).and_then(majit_gc::typeid_is_object) {
                Some(true) => OpResult::Void,
                Some(false) => OpResult::GuardFailed,
                None => OpResult::GuardFailed,
            }
        }

        // ── SameAs / Copy ──
        OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => {
            let a = same_as_value(values, op);
            OpResult::Value(a)
        }

        // ── No-op markers ──
        OpCode::Keepalive
        | OpCode::ForceSpill
        | OpCode::VirtualRefFinish
        | OpCode::RecordExactClass
        | OpCode::RecordExactValueR
        | OpCode::RecordExactValueI
        | OpCode::RecordKnownResult
        | OpCode::QuasiimmutField
        | OpCode::AssertNotNone
        | OpCode::IncrementDebugCounter => OpResult::Void,

        // ── ForceToken ──
        OpCode::ForceToken => {
            // Return a dummy token in blackhole mode
            OpResult::Value(0)
        }

        // ── Exception operations ──
        OpCode::SaveException => {
            // Return the pending exception value.
            OpResult::Value(exc.exc_value)
        }
        OpCode::SaveExcClass => {
            // Return the pending exception class.
            OpResult::Value(exc.exc_class)
        }
        OpCode::RestoreException => {
            // Restore exception state from (class, value) args.
            let cls = values.resolve(op.arg(0));
            let val = values.resolve(op.arg(1));
            exc.set(cls, val);
            OpResult::Void
        }
        OpCode::CheckMemoryError => {
            // If the allocation returned null, set a MemoryError exception.
            let ptr = values.resolve(op.arg(0));
            if ptr == 0 {
                // Set a generic memory error (class=1 by convention).
                exc.set(1, 0);
            }
            OpResult::Void
        }

        // ── Overflow arithmetic ──
        // executor.py: do_int_add_ovf/sub_ovf/mul_ovf set ovf_flag on overflow, return 0
        OpCode::IntAddOvf => {
            let (a, b) = binop(values, op);
            exc.ovf_flag = false;
            match a.checked_add(b) {
                Some(z) => OpResult::Value(z),
                None => {
                    exc.ovf_flag = true;
                    OpResult::Value(a.wrapping_add(b))
                }
            }
        }
        OpCode::IntSubOvf => {
            let (a, b) = binop(values, op);
            exc.ovf_flag = false;
            match a.checked_sub(b) {
                Some(z) => OpResult::Value(z),
                None => {
                    exc.ovf_flag = true;
                    OpResult::Value(a.wrapping_sub(b))
                }
            }
        }
        OpCode::IntMulOvf => {
            let (a, b) = binop(values, op);
            exc.ovf_flag = false;
            match a.checked_mul(b) {
                Some(z) => OpResult::Value(z),
                None => {
                    exc.ovf_flag = true;
                    OpResult::Value(a.wrapping_mul(b))
                }
            }
        }

        // ── Float comparisons ──
        OpCode::FloatLt => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a < b) as i64)
        }
        OpCode::FloatLe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a <= b) as i64)
        }
        OpCode::FloatGt => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a > b) as i64)
        }
        OpCode::FloatGe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a >= b) as i64)
        }
        OpCode::FloatEq => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::FloatNe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a != b) as i64)
        }

        // ── Additional float operations ──
        OpCode::FloatFloorDiv => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits((a / b).floor()) as i64)
        }
        OpCode::FloatMod => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a % b) as i64)
        }

        // ── VirtualRef (pass through in blackhole) ──
        OpCode::VirtualRefI | OpCode::VirtualRefR => {
            let a = unop(values, op);
            OpResult::Value(a)
        }

        // ── Call operations (pass through with concrete values) ──
        // In blackhole mode, calls should re-execute with concrete args.
        // For now, we handle CALL_PURE variants (can evaluate if all args known).
        // Call operations — return placeholder 0 in no-memory path.
        // The execute_one_with_memory path handles actual dispatch.
        OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF => OpResult::Value(0),
        OpCode::CallPureN => OpResult::Void,
        OpCode::CallI
        | OpCode::CallR
        | OpCode::CallF
        | OpCode::CallMayForceI
        | OpCode::CallMayForceR
        | OpCode::CallMayForceF
        | OpCode::CallReleaseGilI
        | OpCode::CallReleaseGilF => OpResult::Value(0),
        OpCode::CallN | OpCode::CallMayForceN | OpCode::CallReleaseGilN => OpResult::Void,

        // ── Memory access (raw) ──
        // In a full blackhole, these would dereference actual pointers.
        // For now, return 0 as placeholder.
        OpCode::GetfieldGcI
        | OpCode::GetfieldGcR
        | OpCode::GetfieldGcF
        | OpCode::GetfieldRawI
        | OpCode::GetfieldRawR
        | OpCode::GetfieldRawF
        | OpCode::GetfieldGcPureI
        | OpCode::GetfieldGcPureR
        | OpCode::GetfieldGcPureF => OpResult::Value(0),
        OpCode::SetfieldGc | OpCode::SetfieldRaw => {
            let resolved_args: Vec<i64> =
                op.getarglist().iter().map(|&r| values.resolve(r)).collect();
            if let (Some(&obj_ptr), Some(&value)) = (resolved_args.first(), resolved_args.get(1)) {
                let __descr_arc = op.getdescr();
                if let Some(fd) = __descr_arc.as_ref().and_then(|d| d.as_field_descr()) {
                    let offset = fd.offset();
                    if obj_ptr != 0 {
                        unsafe {
                            let dest = (obj_ptr as *mut u8).add(offset) as *mut i64;
                            *dest = value;
                        }
                    }
                }
            }
            OpResult::Void
        }

        // ── Array access ──
        OpCode::GetarrayitemGcI
        | OpCode::GetarrayitemGcR
        | OpCode::GetarrayitemGcF
        | OpCode::GetarrayitemRawI
        | OpCode::GetarrayitemRawR
        | OpCode::GetarrayitemRawF
        | OpCode::GetarrayitemGcPureI
        | OpCode::GetarrayitemGcPureR
        | OpCode::GetarrayitemGcPureF => OpResult::Value(0),
        OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => OpResult::Void,

        // ── Array/string length ──
        OpCode::ArraylenGc => OpResult::Value(0),
        OpCode::Strlen | OpCode::Unicodelen => OpResult::Value(0),

        // ── Allocation ──
        // Allocate a real object so SetfieldGc can write fields.
        // IR blackhole may encounter New+SetfieldGc when the trace
        // contains unoptimized allocation (e.g. result_type=Ref finish).
        OpCode::New | OpCode::NewWithVtable => {
            let size = op.with_size_descr(|sd| sd.size()).unwrap_or(16);
            let layout = std::alloc::Layout::from_size_align(size, 8)
                .unwrap_or(std::alloc::Layout::new::<[u8; 16]>());
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            OpResult::Value(ptr as i64)
        }
        OpCode::NewArray | OpCode::NewArrayClear => OpResult::Value(0),
        OpCode::Newstr | OpCode::Newunicode => OpResult::Value(0),

        // ── String/char access ──
        OpCode::Strgetitem | OpCode::Unicodegetitem => OpResult::Value(0),
        OpCode::Strsetitem | OpCode::Unicodesetitem => OpResult::Void,
        OpCode::Strhash | OpCode::Unicodehash => OpResult::Value(0),

        // ── Interior field access ──
        OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcR | OpCode::GetinteriorfieldGcF => {
            OpResult::Value(0)
        }
        OpCode::SetinteriorfieldGc | OpCode::SetinteriorfieldRaw => OpResult::Void,

        // ── Raw memory ──
        OpCode::RawStore => OpResult::Void,
        OpCode::RawLoadI | OpCode::RawLoadF => OpResult::Value(0),

        // ── GC write barriers (no-op in blackhole) ──
        OpCode::CondCallGcWb | OpCode::CondCallGcWbArray | OpCode::ZeroArray => OpResult::Void,

        // ── Nursery allocation (no-op in blackhole) ──
        OpCode::CallMallocNursery
        | OpCode::CallMallocNurseryVarsize
        | OpCode::CallMallocNurseryVarsizeFrame
        | OpCode::NurseryPtrIncrement => OpResult::Value(0),

        // ── Pointer comparisons/casts ──
        OpCode::PtrEq | OpCode::InstancePtrEq => {
            let (a, b) = binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::PtrNe | OpCode::InstancePtrNe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        // `assembler.py:1528-1529 genop_cast_ptr_to_int =
        // _genop_same_as` / `genop_cast_int_to_ptr = _genop_same_as`.
        // PyPy backends, executor, and test_lltype.py:693-701 /
        // runner_test.py:1957-1966 all treat the casts as raw identity.
        // The blackhole.py:603-610 `ll_assert` is a diagnostic check,
        // not a transformation — the value passes through unchanged.
        OpCode::CastPtrToInt | OpCode::CastIntToPtr => {
            let a = unop(values, op);
            OpResult::Value(a)
        }
        OpCode::CastOpaquePtr => {
            // `bhimpl_cast_opaque_ptr` is an lltype-level ptr→ptr
            // identity (no AddressAsInt encoding involved); pass
            // through unchanged.
            let a = unop(values, op);
            OpResult::Value(a)
        }

        // ── CALL_ASSEMBLER: cannot be executed in the blackhole ──
        // Must fall back to force_fn which creates a proper callee frame.
        OpCode::CallAssemblerI | OpCode::CallAssemblerR | OpCode::CallAssemblerF => {
            OpResult::Unsupported("CallAssembler requires force_fn fallback".to_string())
        }
        OpCode::CallAssemblerN => {
            OpResult::Unsupported("CallAssemblerN requires force_fn fallback".to_string())
        }

        // ── Cond call (conditional function call) ──
        OpCode::CondCallValueI | OpCode::CondCallValueR => OpResult::Value(0),
        OpCode::CondCallN => OpResult::Void,

        // ── Thread-local ref ──
        OpCode::ThreadlocalrefGet => OpResult::Value(0),

        // ── Loopinvariant calls ──
        OpCode::CallLoopinvariantI | OpCode::CallLoopinvariantR | OpCode::CallLoopinvariantF => {
            OpResult::Value(0)
        }
        OpCode::CallLoopinvariantN => OpResult::Void,

        // ── GC loads ──
        OpCode::GcLoadI | OpCode::GcLoadR | OpCode::GcLoadF => OpResult::Value(0),
        OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
            OpResult::Value(0)
        }
        OpCode::GcStore | OpCode::GcStoreIndexed => OpResult::Void,

        // ── Vec loads/stores ──
        OpCode::VecLoadI | OpCode::VecLoadF => OpResult::Value(0),
        OpCode::VecStore => OpResult::Void,

        // ── Vector arithmetic (scalar emulation in blackhole) ──
        OpCode::VecIntAdd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_add(b))
        }
        OpCode::VecIntSub => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_sub(b))
        }
        OpCode::VecIntMul => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_mul(b))
        }
        OpCode::VecIntAnd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a & b)
        }
        OpCode::VecIntOr => {
            let (a, b) = binop(values, op);
            OpResult::Value(a | b)
        }
        OpCode::VecIntXor => {
            let (a, b) = binop(values, op);
            OpResult::Value(a ^ b)
        }
        OpCode::VecFloatAdd => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a + b) as i64)
        }
        OpCode::VecFloatSub => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a - b) as i64)
        }
        OpCode::VecFloatMul => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a * b) as i64)
        }
        OpCode::VecFloatTrueDiv => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a / b) as i64)
        }
        OpCode::VecFloatNeg => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(-a) as i64)
        }
        OpCode::VecFloatAbs => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(a.abs()) as i64)
        }

        // ── Vector comparisons (scalar emulation) ──
        OpCode::VecFloatEq => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::VecFloatNe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::VecFloatXor => {
            let (a, b) = binop(values, op);
            OpResult::Value(a ^ b)
        }
        OpCode::VecIntIsTrue => {
            let a = unop(values, op);
            OpResult::Value((a != 0) as i64)
        }
        OpCode::VecIntNe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::VecIntEq => {
            let (a, b) = binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::VecIntSignext => {
            let (a, b) = binop(values, op);
            let bits = b * 8;
            let shift = 64 - bits;
            OpResult::Value((a << shift) >> shift)
        }

        // ── Vector casts (scalar emulation) ──
        OpCode::VecCastFloatToInt => {
            let a = float_unop(values, op);
            OpResult::Value(a as i64)
        }
        OpCode::VecCastIntToFloat => {
            let a = unop(values, op);
            OpResult::Value(f64::to_bits(a as f64) as i64)
        }
        OpCode::VecCastFloatToSinglefloat => {
            let a = float_unop(values, op);
            let f32_val = a as f32;
            OpResult::Value(f32_val.to_bits() as i64)
        }
        OpCode::VecCastSinglefloatToFloat => {
            let a = unop(values, op);
            let f32_val = f32::from_bits(a as u32);
            OpResult::Value(f64::to_bits(f32_val as f64) as i64)
        }

        // ── Vector pack/unpack/expand (scalar emulation) ──
        OpCode::VecI => OpResult::Value(0),
        OpCode::VecF => OpResult::Value(f64::to_bits(0.0) as i64),
        OpCode::VecUnpackI | OpCode::VecUnpackF => {
            // unpack(vec, lane, count) -> return vec (first scalar)
            let a = unop(values, op);
            OpResult::Value(a)
        }
        OpCode::VecPackI | OpCode::VecPackF => {
            // pack(vec, scalar, lane, count) -> return scalar
            let scalar = values.resolve(op.arg(1));
            OpResult::Value(scalar)
        }
        OpCode::VecExpandI | OpCode::VecExpandF => {
            // expand(scalar) -> return scalar
            let a = unop(values, op);
            OpResult::Value(a)
        }

        // ── String/unicode copy ──
        OpCode::Copystrcontent | OpCode::Copyunicodecontent => OpResult::Void,

        // ── Misc conversions ──
        OpCode::UintMulHigh => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64 as u128 * b as u64 as u128) >> 64) as i64)
        }
        OpCode::CastFloatToSinglefloat => {
            let a = float_unop(values, op);
            let f32_val = a as f32;
            OpResult::Value(f32_val.to_bits() as i64)
        }
        OpCode::CastSinglefloatToFloat => {
            let a = unop(values, op);
            let f32_val = f32::from_bits(a as u32);
            OpResult::Value(f64::to_bits(f32_val as f64) as i64)
        }
        OpCode::ConvertFloatBytesToLonglong => {
            let a = unop(values, op);
            OpResult::Value(a) // f64 bits already stored as i64
        }
        OpCode::ConvertLonglongBytesToFloat => {
            let a = unop(values, op);
            OpResult::Value(a) // i64 bits reinterpreted as f64
        }

        // ── Debug / portal frame markers ──
        OpCode::DebugMergePoint
        | OpCode::EnterPortalFrame
        | OpCode::LeavePortalFrame
        | OpCode::JitDebug => OpResult::Void,

        // ── LoadFromGcTable / LoadEffectiveAddress ──
        OpCode::LoadFromGcTable | OpCode::LoadEffectiveAddress => OpResult::Value(0),

        // All OpCode variants are explicitly handled above.
        // This arm is unreachable but kept for forward-compatibility
        // when new opcodes are added to the IR.
        #[allow(unreachable_patterns)]
        other => OpResult::Unsupported(format!(
            "blackhole: opcode {:?} has no interpreter handler",
            other
        )),
    }
}

pub(crate) fn binop(values: &(impl ValueStore + ?Sized), op: &Op) -> (i64, i64) {
    let a = values.resolve(op.arg(0));
    let b = values.resolve(op.arg(1));
    (a, b)
}

pub(crate) fn unop(values: &(impl ValueStore + ?Sized), op: &Op) -> i64 {
    values.resolve(op.arg(0))
}

pub(crate) fn same_as_value(values: &(impl ValueStore + ?Sized), op: &Op) -> i64 {
    if op.num_args() > 0 {
        unop(values, op)
    } else if !op.pos.get().is_none() {
        values.resolve(op.pos.get())
    } else {
        0
    }
}

pub(crate) fn float_binop(values: &(impl ValueStore + ?Sized), op: &Op) -> (f64, f64) {
    let a = f64::from_bits(values.resolve(op.arg(0)) as u64);
    let b = f64::from_bits(values.resolve(op.arg(1)) as u64);
    (a, b)
}

pub(crate) fn float_unop(values: &(impl ValueStore + ?Sized), op: &Op) -> f64 {
    f64::from_bits(values.resolve(op.arg(0)) as u64)
}

/// rpython/jit/metainterp/executor.py:524-528 `execute_varargs(cpu, metainterp, opnum, argboxes, descr)`.
///
/// ```python
/// def execute_varargs(cpu, metainterp, opnum, argboxes, descr):
///     # only for opnums with a variable arity (calls, typically)
///     check_descr(descr)
///     func = get_execute_function(opnum, -1, True)
///     return func(cpu, metainterp, argboxes, descr)
/// ```
///
/// For CALL_* opcodes, `func` ultimately calls `cpu.bh_call_*(funcaddr,
/// args)`.  Pyre routes every arm through `dispatch::call_int_function`
/// / `dispatch::call_void_function` using the concrete values carried
/// alongside each typed argbox.  The Float arm shares the i64-return
/// ABI: helper concrete pointers built by `#[jit_module]` pre-pack the
/// f64 result via `f64::to_bits` (majit-macros/src/lib.rs:194), and
/// callers recover the f64 with `f64::from_bits(resvalue as u64)`
/// (pyjitpl/mod.rs:8901-8902) — bit-identical to the BC_CALL_FLOAT
/// family in blackhole.rs:2349-2371 which uses the same convention.
/// `argboxes[0]` is the funcbox (carrying the function pointer in its
/// `i64` slot) and the remaining slots are the typed call arguments.
///
/// `metainterp` mirrors RPython's `self` parameter: helper-side
/// exceptions published on the `BH_LAST_EXC_VALUE` thread-local seam
/// (the convention `bh_call_fn_impl` and friends use) are transcribed
/// onto `metainterp.last_exc_value` before returning, the same way
/// RPython's cpu auto-publishes onto `cpu.last_exc_value`.  Pyre has
/// no separate `cpu` object; the metainterp owns the field directly.
///
/// Returns the concrete result value as an `i64` — for void-returning
/// calls returns `0` (caller ignores it); for float-returning calls
/// the i64 carries the f64 bits via `f64::to_bits` and the caller
/// must unpack with `f64::from_bits(resvalue as u64)`.  When the
/// helper raises (BH_LAST_EXC_VALUE seam fires), the post-call hook
/// transcribes onto `metainterp.last_exc_value` and overrides the
/// result with `0` — matching `executor.py:52-78`'s neutral-zero
/// behavior across INT, REF (NULL), FLOAT (longlong.ZEROF), and VOID.
// executor.py:188-190
//
//     def do_getfield_gc_i(cpu, _, structbox, fielddescr):
//         struct = structbox.getref_base()
//         return cpu.bh_getfield_gc_i(struct, fielddescr)
//
// `structbox.getref_base()` returns the GCREF the box carries; pyre's
// flat box analog is the concrete `i64` shadow paired with the symbolic
// OpRef (the (OpRef, i64) tuple returned by `read_ref_reg`). The
// caller has already projected `structbox.getref_base()` and passes it
// as `structbox` here so the function shape matches RPython 1:1.
pub fn do_getfield_gc_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> i64 {
    let struct_ = structbox;
    cpu.bh_getfield_gc_i(struct_, fielddescr)
}

// executor.py:192-194
pub fn do_getfield_gc_r(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> majit_ir::GcRef {
    let struct_ = structbox;
    cpu.bh_getfield_gc_r(struct_, fielddescr)
}

// executor.py:196-198
pub fn do_getfield_gc_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> f64 {
    let struct_ = structbox;
    cpu.bh_getfield_gc_f(struct_, fielddescr)
}

// executor.py:206-212 do_getarrayitem_gc_{i,r,f}: project box → gcref +
// concrete index, dispatch to `cpu.bh_getarrayitem_gc_*`.  Pyre's flat
// box analog passes `(array, index)` as plain `i64` values projected
// from the symbolic OpRefs by the caller.
pub fn do_getarrayitem_gc_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> i64 {
    cpu.bh_getarrayitem_gc_i(arraybox, indexbox, arraydescr)
}

pub fn do_getarrayitem_gc_r(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> majit_ir::GcRef {
    cpu.bh_getarrayitem_gc_r(arraybox, indexbox, arraydescr)
}

pub fn do_getarrayitem_gc_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> f64 {
    cpu.bh_getarrayitem_gc_f(arraybox, indexbox, arraydescr)
}

// blackhole.py:1370 bhimpl_arraylen_gc(cpu, array, arraydescr): direct
// `cpu.bh_arraylen_gc(array, arraydescr)`.  RPython has no explicit
// `do_arraylen_gc` in executor.py; the dispatch goes through the
// blackhole fallback wrapper.  Pyre exposes it here in `executor.rs`
// for `TraceCtx::arraylen_sanity_load` to consume directly without
// importing the blackhole module.  Array is projected to `i64` from
// the BoxRef (`arraybox.getref_base()` analog).
pub fn do_arraylen_gc(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> i64 {
    cpu.bh_arraylen_gc(arraybox, arraydescr)
}

// executor.py:132 do_getarrayitem_raw_{i,f}: project arraybox → int
// (raw pointer), dispatch to `cpu.bh_getarrayitem_raw_*`.  Distinct
// from `do_getarrayitem_gc_*` (executor.py:117) which projects via
// `arraybox.getref_base()` — raw arrays carry their pointer as a
// plain integer.  Pyre passes the raw pointer as `i64`.
pub fn do_getarrayitem_raw_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> i64 {
    cpu.bh_getarrayitem_raw_i(arraybox, indexbox, arraydescr)
}

pub fn do_getarrayitem_raw_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> f64 {
    cpu.bh_getarrayitem_raw_f(arraybox, indexbox, arraydescr)
}

// executor.py:200 do_getfield_raw_{i,r,f}: project structbox → int
// (raw pointer via `structbox.getint()`), dispatch to
// `cpu.bh_getfield_raw_*`.  Distinct from `do_getfield_gc_*`
// (executor.py:188) which projects via `structbox.getref_base()` —
// raw structs carry their pointer as a plain integer.  Pyre's caller
// projects the symbolic OpRef carrier to `i64` directly.
pub fn do_getfield_raw_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> i64 {
    cpu.bh_getfield_raw_i(structbox, fielddescr)
}

pub fn do_getfield_raw_r(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> majit_ir::GcRef {
    cpu.bh_getfield_raw_r(structbox, fielddescr)
}

pub fn do_getfield_raw_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> f64 {
    cpu.bh_getfield_raw_f(structbox, fielddescr)
}

pub fn execute_varargs<M: Clone>(
    metainterp: &mut crate::pyjitpl::MetaInterp<M>,
    opnum: OpCode,
    argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
    descr: &dyn majit_ir::descr::CallDescr,
) -> i64 {
    debug_assert!(opnum.is_call(), "execute_varargs requires a call opcode");
    // RPython's `cpu` parameter is the seam through which helper-side
    // exceptions reach `metainterp.last_exc_value` (`cpu.bh_call_*`
    // writes onto `cpu.last_exception` and the metainterp's
    // post-execute hook copies it).  Pyre has no separate cpu; we use
    // the `BH_LAST_EXC_VALUE` thread-local that production helpers
    // (`bh_call_fn_impl` etc.) publish on, mirroring the same convention
    // every blackhole.rs CALL_* arm uses (blackhole.rs:2270-2392).
    // Clear before dispatch so a stale value from a prior call cannot
    // bleed into this one.
    crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    // Inner closure carries every existing return path so the outer
    // wrapper can run BH_LAST_EXC_VALUE transcription regardless of
    // which arm fires.
    let result = (|| -> i64 {
        // COND_CALL / COND_CALL_VALUE_* layout (pyjitpl.py:2128-2151):
        //   argboxes[0] = condbox / valuebox
        //   argboxes[1] = funcbox
        //   argboxes[2..] = call args
        // RPython's executor handles cond-call dispatch via a per-opcode
        // execute function (`do_cond_call_*`); pyre inlines the same
        // semantics here.
        //
        // Two distinct shapes (blackhole.py:1257-1276):
        //   bhimpl_conditional_call_ir_v(condition, func, ...):
        //       if condition: cpu.bh_call_v(func, ...)        # void
        //   bhimpl_conditional_call_value_ir_{i,r}(value, func, ...):
        //       if value == 0: value = cpu.bh_call_*(func, ...)
        //       return value
        if matches!(opnum, OpCode::CondCallN) {
            debug_assert!(
                argboxes.len() >= 2,
                "COND_CALL_N requires [condbox, funcbox, *args]",
            );
            let cond = argboxes[0].2;
            if cond == 0 {
                // condition false → skip the call.
                return 0;
            }
            let func_ptr = argboxes[1].2 as *const ();
            let concrete_args: Vec<i64> = argboxes[2..].iter().map(|(_, _, c)| *c).collect();
            crate::pyjitpl::call_void_function(func_ptr, &concrete_args);
            return 0;
        }
        if matches!(opnum, OpCode::CondCallValueI | OpCode::CondCallValueR) {
            debug_assert!(
                argboxes.len() >= 2,
                "COND_CALL_VALUE_* requires [valuebox, funcbox, *args]",
            );
            let value = argboxes[0].2;
            if value != 0 {
                // blackhole.py:1267 / 1274: nonzero `value` short-circuits
                // and returns the existing value without calling.
                return value;
            }
            // value == 0 → call and return the call's result.
            let func_ptr = argboxes[1].2 as *const ();
            let concrete_args: Vec<i64> = argboxes[2..].iter().map(|(_, _, c)| *c).collect();
            return crate::pyjitpl::call_int_function(func_ptr, &concrete_args);
        }
        debug_assert!(
            !argboxes.is_empty(),
            "execute_varargs: argboxes must include funcbox at slot 0",
        );
        let func_ptr = argboxes[0].2 as *const ();
        let concrete_args: Vec<i64> = argboxes[1..].iter().map(|(_, _, c)| *c).collect();
        match descr.result_type() {
            // RPython dispatches Int and Ref through the same backend
            // primitive cpu.bh_call_i (returns i64); pyre's
            // call_int_function does the same — Ref is bit-identical to Int
            // at the ABI level.
            majit_ir::Type::Int | majit_ir::Type::Ref => {
                crate::pyjitpl::call_int_function(func_ptr, &concrete_args)
            }
            majit_ir::Type::Void => {
                crate::pyjitpl::call_void_function(func_ptr, &concrete_args);
                0
            }
            majit_ir::Type::Float => {
                // pyjitpl.py:2119 — CALL_F dispatches through cpu.bh_call_f.
                // Caller-contract: every path that reaches this arm carries
                // `funcbox.2` as a hand-written or `#[jit_module]`-generated
                // function pointer with i64-return ABI:
                //   * `do_recursive_call` (mod.rs:11148-11152) sets funcbox.2
                //     to `targetjitdriver_sd.portal_runner_adr`.  Pyre's
                //     portal entry is `bh_portal_runner(all_i, all_r, all_f)
                //     -> i64` (pyre-jit/src/call_jit.rs:467); it never
                //     declares an f64 return.
                //   * `#[jit_module]` (majit-macros/src/lib.rs:267) emits a
                //     Float helper's `concrete_ptr` as `extern "C" fn(...)
                //     -> i64` with the f64 result pre-packed via
                //     `f64::to_bits`; the f64-ABI `trace_ptr` is consumed
                //     only by pyre-jit-trace's `TraceCtx::call_may_force_*`
                //     family, which has its own seam and never reaches
                //     this arm.
                // Therefore route through `call_int_function` and let the
                // caller recover the f64 via `f64::from_bits` when needed
                // — bit-identical to blackhole.rs:2349-2371 BC_CALL_FLOAT
                // family which makes the same ABI choice for the same
                // reason (`registers_f` is an i64-carrier mirroring
                // RPython's `longlong.ZEROF` packing).
                crate::pyjitpl::call_int_function(func_ptr, &concrete_args)
            }
        }
    })();
    // Mirror RPython's executor.py:52-78 post-call exception flow:
    // each `cpu.bh_call_*` arm wraps the call in `try: ... except
    // Exception as e: metainterp.execute_raised(e); result = ZERO`.
    // Pyre observes the same condition through the BH_LAST_EXC_VALUE
    // thread-local seam.  Two pieces have to fire together:
    //   1. `metainterp.execute_raised(bh_exc, constant=False)` — sets
    //      `last_exc_value` AND clears `class_of_last_exc_is_const`,
    //      so a stale `True` from a prior GUARD_EXCEPTION cannot make
    //      `handle_possible_exception` treat the new exception's
    //      class as constant (pyjitpl.py:2745-2755).
    //   2. Override the returned value with the type's neutral zero —
    //      `make_result_of_lastop` (pyjitpl/mod.rs:8893) snapshots the
    //      concrete result *before* `handle_possible_exception` runs,
    //      so leaving the helper's return value in place can pin a
    //      garbage value into the resume snapshot.  `i64 == 0` covers
    //      INT=0, REF=NULL, and FLOAT=longlong.ZEROF (0.0_f64.to_bits()
    //      as i64 == 0); VOID callers ignore the slot.
    let bh_exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| {
        let v = c.get();
        c.set(0);
        v
    });
    if bh_exc != 0 {
        metainterp.execute_raised(bh_exc, false);
        return 0;
    }
    result
}

/// executor.py:555 `execute_nonspec_const` for binary integer opcodes.
///
/// Returns the folded `i64` result when the operation is recognized and
/// the result is well-defined; returns `None` to abort folding when:
///   * the opcode is not a recognized binary int op
///   * an OVF arithmetic op (IntAddOvf/SubOvf/MulOvf) overflows —
///     RPython's `do_int_add_ovf` then hits
///     `assert metainterp is not None` (executor.py:287) which
///     AssertionErrors in the `constant_fold` path (metainterp=None);
///     pyre prefers the softer `None` skip so the op stays in the
///     trace and the runtime guard fires
///   * a shift count is outside `0..64` (mirrors
///     `blackhole.py:258 check_shift_count`)
///   * IntFloorDiv / IntMod with a zero divisor
///
/// Non-OVF IntAdd/IntSub/IntMul match `bhimpl_int_add/_sub/_mul`
/// (`blackhole.py:459-468`) which compute `intmask(a + b)` — i.e.
/// wrapping i64 arithmetic. Earlier `checked_*` use here would have
/// aborted the fold on a representable wrapping result.
///
/// Mirrors the `do_int_*` entries at executor.py:279-309 (OVF) +
/// `EXECUTE_BY_NUM_ARGS` binary-int rows (the unrolled dispatch table
/// generated at executor.py:495-498).
pub fn execute_binary_int_const(opcode: OpCode, a: i64, b: i64) -> Option<i64> {
    let result = match opcode {
        OpCode::IntAdd => a.wrapping_add(b),
        OpCode::IntSub => a.wrapping_sub(b),
        OpCode::IntMul => a.wrapping_mul(b),
        OpCode::IntAddOvf => a.checked_add(b)?,
        OpCode::IntSubOvf => a.checked_sub(b)?,
        OpCode::IntMulOvf => a.checked_mul(b)?,
        OpCode::IntAnd => a & b,
        OpCode::IntOr => a | b,
        OpCode::IntXor => a ^ b,
        OpCode::IntLshift if b >= 0 && b < 64 => a << b,
        OpCode::IntRshift if b >= 0 && b < 64 => a >> b,
        OpCode::UintRshift if b >= 0 && b < 64 => (a as u64 >> b as u64) as i64,
        OpCode::IntLt => (a < b) as i64,
        OpCode::IntLe => (a <= b) as i64,
        OpCode::IntGt => (a > b) as i64,
        OpCode::IntGe => (a >= b) as i64,
        OpCode::IntEq => (a == b) as i64,
        OpCode::IntNe => (a != b) as i64,
        OpCode::UintLt => ((a as u64) < (b as u64)) as i64,
        OpCode::UintLe => ((a as u64) <= (b as u64)) as i64,
        OpCode::UintGe => ((a as u64) >= (b as u64)) as i64,
        OpCode::UintGt => ((a as u64) > (b as u64)) as i64,
        OpCode::IntFloorDiv if b != 0 => {
            let (q, r) = (a / b, a % b);
            if (r != 0) && ((r ^ b) < 0) { q - 1 } else { q }
        }
        OpCode::IntMod if b != 0 => {
            let r = a % b;
            if (r != 0) && ((r ^ b) < 0) { r + b } else { r }
        }
        OpCode::IntSignext if b >= 1 && b <= 8 => {
            // blackhole.py:568 bhimpl_int_signext → support.py:30 int_signext.
            // Sign-extend `a` from `b` bytes. b=8 yields shift=0, i.e.
            // identity — matches the upstream r_uint round-trip when the
            // mask `(1 << 64) - 1` covers the whole word.
            let shift = 64 - b * 8;
            (a << shift) >> shift
        }
        OpCode::UintMulHigh => {
            // blackhole.py bhimpl_uint_mul_high — high 64 of (a as u64) * (b as u64).
            (((a as u64) as u128 * (b as u64) as u128) >> 64) as i64
        }
        _ => return None,
    };
    Some(result)
}

/// executor.py:495-498 ptr-compare row of EXECUTE_BY_NUM_ARGS.
/// Mirrors blackhole.py bhimpl_ptr_eq/_ne and instance_ptr_eq/_ne —
/// straight pointer identity once both args are constant references.
pub fn execute_ptr_compare_const(opcode: OpCode, a: usize, b: usize) -> Option<i64> {
    let result = match opcode {
        OpCode::PtrEq | OpCode::InstancePtrEq => a == b,
        OpCode::PtrNe | OpCode::InstancePtrNe => a != b,
        _ => return None,
    };
    Some(result as i64)
}

/// executor.py:495-498 unary-int row of EXECUTE_BY_NUM_ARGS, the
/// 1-arg variant. Mirrors blackhole.py:528-566 bhimpl_int_neg /
/// _invert / _is_zero / _is_true / _force_ge_zero.
///
/// Returns `None` for unrecognized opcodes so the caller can fall
/// through to other dispatch paths (executor.py:559's `assert False`
/// shape is reserved for non-matching opnums via the unrolled match).
pub fn execute_unary_int_const(opcode: OpCode, a: i64) -> Option<i64> {
    let result = match opcode {
        OpCode::IntNeg => a.wrapping_neg(),
        OpCode::IntInvert => !a,
        OpCode::IntIsZero => (a == 0) as i64,
        OpCode::IntIsTrue => (a != 0) as i64,
        OpCode::IntForceGeZero => a.max(0),
        _ => return None,
    };
    Some(result)
}

/// executor.py:495-498 unary-float row mirrors blackhole.py float
/// unops: bhimpl_float_neg / _abs.
pub fn execute_unary_float_const(opcode: OpCode, a: f64) -> Option<f64> {
    let result = match opcode {
        OpCode::FloatNeg => -a,
        OpCode::FloatAbs => a.abs(),
        _ => return None,
    };
    Some(result)
}

/// executor.py:495-498 binary-float row. Float arithmetic + comparisons
/// (comparisons return bool wrapped as 0/1 in the caller). Mirrors
/// blackhole.py bhimpl_float_add/_sub/_mul/_truediv (`:697-718`).
/// FLOAT_TRUEDIV with `b == 0.0` is NOT folded — see upstream
/// `test_optimizebasic.test_float_division_by_multiplication` which
/// preserves `float_truediv(f, 0.0)` in the optimized loop rather than
/// freezing the IEEE inf/nan constant. The runtime executor still
/// performs `a / b` per `blackhole.py:717` (translated C semantics);
/// only trace-time folding is suppressed.
pub fn execute_binary_float_const(opcode: OpCode, a: f64, b: f64) -> Option<f64> {
    let result = match opcode {
        OpCode::FloatAdd => a + b,
        OpCode::FloatSub => a - b,
        OpCode::FloatMul => a * b,
        OpCode::FloatTrueDiv if b != 0.0 => a / b,
        _ => return None,
    };
    Some(result)
}

/// executor.py:495-498 float→bool row. Mirrors blackhole.py
/// bhimpl_float_lt/_le/_eq/_ne/_gt/_ge.
pub fn execute_float_compare_const(opcode: OpCode, a: f64, b: f64) -> Option<i64> {
    let result = match opcode {
        OpCode::FloatLt => a < b,
        OpCode::FloatLe => a <= b,
        OpCode::FloatEq => a == b,
        OpCode::FloatNe => a != b,
        OpCode::FloatGt => a > b,
        OpCode::FloatGe => a >= b,
        _ => return None,
    };
    Some(result as i64)
}

/// `executor.py:555 execute_nonspec_const` delegates to `_execute_arglist`,
/// which raises `NotImplementedError` at `:610` when no helper is registered
/// for the opnum. RPython's `optimizer.py:810 constant_fold` does not catch
/// that exception; it propagates to the caller. Pyre encodes the same
/// dispatch distinction at the type level:
///   * `Err(NotImplemented)` — no helper claimed the opnum (terminal
///     fall-through below), mirroring the upstream raise.
///   * `Ok(None)` — a helper claimed the opnum but declined to fold
///     (null gcref, unsupported field size, etc.); pyre keeps these
///     as `Ok(None)` so the caller can still see "helper ran, fold
///     skipped" distinctly from "no helper".
///   * `Ok(Some(value))` — successful fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotImplemented;

/// executor.py:555 `execute_nonspec_const` free function — the
/// generic opnum dispatch invoked by `optimizer.py:810 constant_fold`
/// once every arg has been resolved to a `Const*` via
/// `get_constant_box`. Mirrors the RPython structure:
///
/// ```python
/// def execute_nonspec_const(cpu, metainterp, opnum, argboxes,
///                           descr=None, type='i'):
///     for num in unrolled_range:
///         if num == opnum:
///             return wrap_constant(_execute_arglist(cpu, metainterp, num,
///                                                    argboxes, descr))
///     assert False
/// ```
///
/// `_execute_arglist` (executor.py:563-610) selects
/// `EXECUTE_BY_NUM_ARGS[arity, withdescr][opnum]` and raises
/// `NotImplementedError` (`:610`) only when no function is registered
/// for the opnum. Pyre returns [`Err(NotImplemented)`](NotImplemented)
/// for that case and `Ok(None)` for helper-internal "decline to fold"
/// outcomes (e.g. null gcref, unsupported field size).
///
/// `_type` is accepted for signature parity with RPython's `type`
/// parameter; it is not consulted because the Value variant in
/// `argboxes` already determines the result type via the helper that
/// fires.
pub fn execute_nonspec_const(
    cpu: &dyn crate::cpu::Cpu,
    opnum: OpCode,
    argboxes: &[majit_ir::Value],
    descr: Option<&majit_ir::descr::DescrRef>,
    _type: majit_ir::Type,
) -> Result<Option<majit_ir::Value>, NotImplemented> {
    use majit_ir::Value;
    let arity = argboxes.len();

    // ── arity == 1 row of EXECUTE_BY_NUM_ARGS ──
    if arity == 1 {
        let a = argboxes[0];
        // executor.py:314-321 `do_same_as_i/r/f` — identity fold for
        // SAME_AS_I/R/F (`bhimpl_int_same_as` etc., `blackhole.py:455`).
        match opnum {
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => return Ok(Some(a)),
            _ => {}
        }
        if let Value::Int(i) = a {
            if let Some(folded) = execute_unary_int_const(opnum, i) {
                return Ok(Some(Value::Int(folded)));
            }
        }
        if let Value::Float(f) = a {
            if let Some(folded) = execute_unary_float_const(opnum, f) {
                return Ok(Some(Value::Float(folded)));
            }
        }
        if let Some(folded) = execute_cast_const(opnum, a) {
            return Ok(Some(folded));
        }
        // GETFIELD_GC_PURE_I/R/F — withdescr arity-1.
        // `optimizer.py:829-832 protect_speculative_operation` has
        // validated the gcref is non-null and of a valid type for
        // `fielddescr.parent_descr` (`llmodel.py:555-567`); the
        // dereference below cannot fault.  Unsupported field_size is
        // `llmodel.py:478 read_int_at_mem`'s NotImplementedError and
        // would propagate to crash upstream — pyre matches via
        // fail-loud `unreachable!()`.
        if let (Value::Ref(struct_ref), Some(d)) = (a, descr) {
            if let Some(fd) = d.as_field_descr() {
                return Ok(Some(match opnum {
                    OpCode::GetfieldGcPureI => match fd.field_size() {
                        1 | 2 | 4 | 8 => Value::Int(cpu.bh_getfield_gc_i(struct_ref.0, fd)),
                        sz => unreachable!(
                            "GETFIELD_GC_PURE_I: unsupported field_size {} \
                             (llmodel.py:478 raises NotImplementedError)",
                            sz
                        ),
                    },
                    OpCode::GetfieldGcPureR => Value::Ref(cpu.bh_getfield_gc_r(struct_ref.0, fd)),
                    OpCode::GetfieldGcPureF => Value::Float(cpu.bh_getfield_gc_f(struct_ref.0, fd)),
                    _ => return Err(NotImplemented),
                }));
            }
        }
        // ARRAYLEN_GC — withdescr arity-1.
        // `executor.py:do_arraylen_gc` → `cpu.bh_arraylen_gc(array, ad)`.
        // `protect_speculative_array` validated the gcref + tid; the
        // arraydescr is expected to carry a `len_descr` (registered
        // by the backend's array metadata), so a missing one is a
        // bug — fail-loud per `llmodel.py:585 assert isinstance(...)`.
        if let (Value::Ref(array), Some(d)) = (a, descr) {
            if let Some(ad) = d.as_array_descr() {
                if opnum == OpCode::ArraylenGc {
                    let len = cpu.bh_arraylen_gc(array, ad).expect(
                        "ARRAYLEN_GC: arraydescr missing len_descr (llmodel.py:585 asserts)",
                    );
                    return Ok(Some(Value::Int(len)));
                }
            }
        }
        // STRLEN / UNICODELEN — by the time the fold is entered,
        // `protect_speculative_string / _unicode` has already validated
        // `str_descr / unicode_descr` is registered (under
        // `supports_guard_gc_type == true`; the false case is gated
        // out at `OptContext::protect_speculative_operation`).
        if let Value::Ref(s) = a {
            match opnum {
                OpCode::Strlen => {
                    let len = cpu
                        .bh_strlen(s)
                        .expect("STRLEN: str_descr unregistered after protect_speculative_string");
                    return Ok(Some(Value::Int(len)));
                }
                OpCode::Unicodelen => {
                    let len = cpu.bh_unicodelen(s).expect(
                        "UNICODELEN: unicode_descr unregistered after protect_speculative_unicode",
                    );
                    return Ok(Some(Value::Int(len)));
                }
                _ => {}
            }
        }
    }

    // ── arity == 2 row of EXECUTE_BY_NUM_ARGS ──
    if arity == 2 {
        if let (Value::Int(a), Value::Int(b)) = (argboxes[0], argboxes[1]) {
            if let Some(folded) = execute_binary_int_const(opnum, a, b) {
                return Ok(Some(Value::Int(folded)));
            }
        }
        if let (Value::Float(a), Value::Float(b)) = (argboxes[0], argboxes[1]) {
            if let Some(folded) = execute_binary_float_const(opnum, a, b) {
                return Ok(Some(Value::Float(folded)));
            }
            if let Some(folded) = execute_float_compare_const(opnum, a, b) {
                return Ok(Some(Value::Int(folded)));
            }
        }
        if let (Value::Ref(a), Value::Ref(b)) = (argboxes[0], argboxes[1]) {
            if let Some(folded) = execute_ptr_compare_const(opnum, a.0, b.0) {
                return Ok(Some(Value::Int(folded)));
            }
        }
        // GETARRAYITEM_GC_PURE_I/R/F — withdescr arity-2 (array, index).
        // `executor.py:do_getarrayitem_gc_pure_*` →
        //   `cpu.bh_getarrayitem_gc_*(array, index, ad)`.
        // `protect_speculative_array` + the array-bounds check at
        // `optimizer.py:865-867` validated the gcref/index pre-fold;
        // unsupported `item_size` matches `llmodel.py:478`'s
        // NotImplementedError, fail-loud via `unreachable!()`.
        if let (Value::Ref(array), Value::Int(index), Some(d)) = (argboxes[0], argboxes[1], descr) {
            if let Some(ad) = d.as_array_descr() {
                return Ok(Some(match opnum {
                    OpCode::GetarrayitemGcPureI => {
                        let v = cpu.bh_getarrayitem_gc_i(array, index, ad).expect(
                            "GETARRAYITEM_GC_PURE_I: unsupported item_size \
                             (llmodel.py:478 raises NotImplementedError)",
                        );
                        Value::Int(v)
                    }
                    OpCode::GetarrayitemGcPureR => {
                        Value::Ref(cpu.bh_getarrayitem_gc_r(array, index, ad))
                    }
                    OpCode::GetarrayitemGcPureF => {
                        Value::Float(cpu.bh_getarrayitem_gc_f(array, index, ad))
                    }
                    _ => return Err(NotImplemented),
                }));
            }
        }
        // STRGETITEM / UNICODEGETITEM — protect_speculative_string /
        // _unicode has validated str_descr / unicode_descr is
        // registered (under `supports_guard_gc_type == true`).
        if let (Value::Ref(s), Value::Int(index)) = (argboxes[0], argboxes[1]) {
            match opnum {
                OpCode::Strgetitem => {
                    let v = cpu.bh_strgetitem(s, index).expect(
                        "STRGETITEM: str_descr unregistered after protect_speculative_string",
                    );
                    return Ok(Some(Value::Int(v)));
                }
                OpCode::Unicodegetitem => {
                    let v = cpu.bh_unicodegetitem(s, index).expect(
                        "UNICODEGETITEM: unicode_descr unregistered after protect_speculative_unicode",
                    );
                    return Ok(Some(Value::Int(v)));
                }
                _ => {}
            }
        }
    }

    // ── arity == 3 row of EXECUTE_BY_NUM_ARGS ──
    if arity == 3 {
        // executor.py `do_int_between` -> blackhole.py:560
        // `bhimpl_int_between(a, b, c): return a <= b < c`.
        if let (Value::Int(a), Value::Int(b), Value::Int(c)) =
            (argboxes[0], argboxes[1], argboxes[2])
        {
            if opnum == OpCode::IntBetween {
                return Ok(Some(Value::Int((a <= b && b < c) as i64)));
            }
        }
    }

    // `executor.py:610 _execute_arglist` raises `NotImplementedError`
    // when `EXECUTE_BY_NUM_ARGS[arity, withdescr][opnum]` is None.
    // RPython's `optimizer.py:810 constant_fold` lets that propagate; Pyre
    // encodes the same missing-helper signal via [`NotImplemented`].
    Err(NotImplemented)
}

/// executor.py cross-type cast fold:
///   CAST_FLOAT_TO_INT / CAST_INT_TO_FLOAT — true numeric conversion
///   CAST_FLOAT_TO_SINGLEFLOAT / CAST_SINGLEFLOAT_TO_FLOAT — f64↔f32 bits
///   CONVERT_FLOAT_BYTES_TO_LONGLONG / CONVERT_LONGLONG_BYTES_TO_FLOAT —
///       reinterpret-bits pass-through (Value carries f64 bits already
///       as i64 in pyre's TraceValues; the cast just relabels)
///   CAST_PTR_TO_INT / CAST_INT_TO_PTR — pointer reinterpret
/// Mirrors blackhole.py bhimpl_cast_*.
pub fn execute_cast_const(opcode: OpCode, arg: majit_ir::Value) -> Option<majit_ir::Value> {
    use majit_ir::{GcRef, Value};
    match (opcode, arg) {
        (OpCode::CastFloatToInt, Value::Float(f)) => {
            // blackhole.py:800-808 `bhimpl_cast_float_to_int = int(int(a))`
            // lowers to `lltype.cast_float_to_int` (C `(long)f`) post-
            // translation. Skip fold on non-finite or out-of-range
            // floats: the runtime path's `as i64` saturates (lenient
            // IEEE policy, see runtime arm at executor.rs:329) but
            // trace-time fold would freeze that saturation as a
            // constant — emit the cast op instead so the runtime
            // computes consistently. The safe i64 window is
            // `i64::MIN ..= i64::MAX`; `i64::MAX + 1` rounds to the
            // same f64 as `i64::MAX` (precision loss), so use the
            // strictly-less-than upper bound `9.223372036854776e18`.
            if !f.is_finite() {
                return None;
            }
            if f < (i64::MIN as f64) || f >= 9.223372036854776e18_f64 {
                return None;
            }
            Some(Value::Int(f as i64))
        }
        (OpCode::CastIntToFloat, Value::Int(i)) => Some(Value::Float(i as f64)),
        (OpCode::CastFloatToSinglefloat, Value::Float(f)) => {
            Some(Value::Int((f as f32).to_bits() as i64))
        }
        (OpCode::CastSinglefloatToFloat, Value::Int(i)) => {
            Some(Value::Float(f32::from_bits(i as u32) as f64))
        }
        (OpCode::ConvertFloatBytesToLonglong, Value::Float(f)) => {
            Some(Value::Int(f.to_bits() as i64))
        }
        (OpCode::ConvertLonglongBytesToFloat, Value::Int(i)) => {
            Some(Value::Float(f64::from_bits(i as u64)))
        }
        // `assembler.py:1528-1529 genop_cast_ptr_to_int =
        // _genop_same_as` / `genop_cast_int_to_ptr = _genop_same_as`.
        // PyPy treats both casts as raw identity at every level:
        // backend, executor, and test_lltype.py:693-701 /
        // runner_test.py:1957-1966 expect
        // `cast_int_to_ptr(21) → cast_ptr_to_int == 21`.
        (OpCode::CastPtrToInt, Value::Ref(r)) => Some(Value::Int(r.0 as i64)),
        (OpCode::CastIntToPtr, Value::Int(i)) => Some(Value::Ref(GcRef(i as usize))),
        _ => None,
    }
}

/// Narrow [`execute_varargs`] carve-out usable without `&mut MetaInterp`.
///
/// RPython `executor.execute_varargs(cpu, ..., exc=False)`
/// (`executor.py:75-78`) skips the `metainterp.execute_raised` coordination
/// when the helper provably cannot raise — `pyjitpl.py:_record_helper_pure`
/// (`pyjitpl.py:1346-1400`) reaches this path for every `EF_ELIDABLE_CANNOT_RAISE`
/// callee. Pyre's walker (`pyre-jit-trace::jitcode_dispatch::
/// dispatch_residual_call_*`) cannot thread `&mut MetaInterp` through the
/// trace recorder seam, so this helper exposes the `exc=False` shape via
/// direct `call_int_function` / `call_void_function` dispatch.
///
/// **Caller contract** (debug-asserted): `descr.get_extra_info()` must report
/// both `check_is_elidable()` true AND `check_can_raise(false)` false.  Any
/// other EI risks landing in `BH_LAST_EXC_VALUE` with no metainterp around
/// to transcribe it, which would silently swallow the exception.
///
/// `args` follows `_build_allboxes` (`pyjitpl.py:1960-1993`) layout
/// **excluding** the funcbox: the funcbox concrete int is `func_ptr` and the
/// remaining concrete operand values pass straight through to the host ABI
/// dispatcher (`pyjitpl::call_int_function` / `call_void_function`).  Up to
/// `MAX_HOST_CALL_ARITY` (16) operand slots.
pub fn execute_pure_call(
    descr: &dyn majit_ir::descr::CallDescr,
    func_ptr: i64,
    args: &[i64],
) -> i64 {
    debug_assert!(
        descr.get_extra_info().check_is_elidable()
            && !descr.get_extra_info().check_can_raise(false),
        "execute_pure_call requires EF_ELIDABLE_CANNOT_RAISE EI"
    );
    let func_ptr = func_ptr as *const ();
    match descr.result_type() {
        // RPython dispatches Int and Ref through the same backend primitive
        // `cpu.bh_call_i` (returns i64); pyre's `call_int_function` does
        // the same — Ref is bit-identical to Int at the ABI level.
        majit_ir::Type::Int | majit_ir::Type::Ref => {
            crate::pyjitpl::call_int_function(func_ptr, args)
        }
        majit_ir::Type::Void => {
            crate::pyjitpl::call_void_function(func_ptr, args);
            0
        }
        // See `execute_varargs`'s Float arm for the i64-bits ABI rationale:
        // `#[jit_module]` Float helpers expose `concrete_ptr` as
        // `extern "C" fn(...) -> i64` with the f64 pre-packed via
        // `f64::to_bits`; routing through `call_int_function` is bit-identical.
        majit_ir::Type::Float => crate::pyjitpl::call_int_function(func_ptr, args),
    }
}

#[cfg(test)]
mod execute_pure_call_tests {
    use super::*;
    use majit_ir::descr::SimpleCallDescr;
    use majit_ir::{EffectInfo, ExtraEffect, Type};

    extern "C" fn double_i64(x: i64) -> i64 {
        x.wrapping_mul(2)
    }

    extern "C" fn add3_i64(a: i64, b: i64, c: i64) -> i64 {
        a.wrapping_add(b).wrapping_add(c)
    }

    extern "C" fn pack_float_to_bits(x: i64) -> i64 {
        let f = f64::from_bits(x as u64) * 2.0;
        f.to_bits() as i64
    }

    extern "C" fn void_no_op(_x: i64) {}

    fn make_descr(arg_types: Vec<Type>, result_type: Type) -> SimpleCallDescr {
        let mut effect = EffectInfo::default();
        effect.extraeffect = ExtraEffect::ElidableCannotRaise;
        SimpleCallDescr::new(0, arg_types, result_type, false, 8, effect)
    }

    #[test]
    fn executes_single_int_arg_and_returns_doubled_result() {
        let descr = make_descr(vec![Type::Int], Type::Int);
        let result = execute_pure_call(&descr, double_i64 as *const () as i64, &[21]);
        assert_eq!(result, 42, "double_i64(21) must return 42");
    }

    #[test]
    fn executes_three_int_args_routing_through_call_int_function() {
        let descr = make_descr(vec![Type::Int, Type::Int, Type::Int], Type::Int);
        let result = execute_pure_call(&descr, add3_i64 as *const () as i64, &[100, 20, 3]);
        assert_eq!(result, 123, "add3_i64(100, 20, 3) must return 123");
    }

    #[test]
    fn float_result_routes_through_call_int_function_with_bits_packing() {
        let descr = make_descr(vec![Type::Float], Type::Float);
        let input_bits = 3.5_f64.to_bits() as i64;
        let result = execute_pure_call(
            &descr,
            pack_float_to_bits as *const () as i64,
            &[input_bits],
        );
        let result_f = f64::from_bits(result as u64);
        assert_eq!(result_f, 7.0, "3.5 * 2.0 must equal 7.0");
    }

    #[test]
    fn void_return_routes_through_call_void_function_and_returns_zero_sentinel() {
        let descr = make_descr(vec![Type::Int], Type::Void);
        let result = execute_pure_call(&descr, void_no_op as *const () as i64, &[99]);
        assert_eq!(result, 0, "void execute_pure_call returns the 0 sentinel");
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "execute_pure_call requires EF_ELIDABLE_CANNOT_RAISE EI")]
    fn non_elidable_ei_panics_debug_assertion() {
        let mut effect = EffectInfo::default();
        effect.extraeffect = ExtraEffect::CannotRaise;
        let descr = SimpleCallDescr::new(0, vec![Type::Int], Type::Int, false, 8, effect);
        let _ = execute_pure_call(&descr, double_i64 as *const () as i64, &[1]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "execute_pure_call requires EF_ELIDABLE_CANNOT_RAISE EI")]
    fn elidable_can_raise_panics_debug_assertion() {
        let mut effect = EffectInfo::default();
        effect.extraeffect = ExtraEffect::ElidableCanRaise;
        let descr = SimpleCallDescr::new(0, vec![Type::Int], Type::Int, false, 8, effect);
        let _ = execute_pure_call(&descr, double_i64 as *const () as i64, &[1]);
    }
}
