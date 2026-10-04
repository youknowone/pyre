//! JIT-internal exception/control-flow types.
//!
//! Mirrors RPython's `jitexc.py`: exceptions raised and caught within
//! the JIT infrastructure, never exposed to user code.

use majit_ir::{GcRef, GreenKey, GreenType, Value};

/// jitexc.py JitException — base class for all JIT control flow.
///
/// In RPython these are Python exceptions. In Rust we model them as an enum
/// returned via `Result::Err(JitException)` from blackhole execution.
#[derive(Debug, Clone, PartialEq)]
pub enum JitException {
    /// jitexc.py DoneWithThisFrameVoid
    DoneWithThisFrameVoid,
    /// jitexc.py DoneWithThisFrameInt
    DoneWithThisFrameInt(i64),
    /// jitexc.py DoneWithThisFrameRef
    DoneWithThisFrameRef(GcRef),
    /// jitexc.py DoneWithThisFrameFloat
    DoneWithThisFrameFloat(f64),
    /// jitexc.py ExitFrameWithExceptionRef
    ExitFrameWithExceptionRef(GcRef),
    /// jitexc.py ContinueRunningNormally
    ContinueRunningNormally(Box<ContinueRunningNormallyArgs>),
    /// pyre-only: the blackhole stopped without finishing the frame.
    ///
    /// Reached from a pyre `abort_permanent` marker and from the two
    /// unresolved-callee refusals (`reject_symbolic_residual_call`,
    /// `reject_unresolved_inline_call`), none of which upstream has: RPython's
    /// jitcodes cover every operation the codewriter accepted, so a blackhole
    /// frame always ends in one of the variants above.
    ///
    /// It is deliberately NOT `DoneWithThisFrameVoid`.  That variant says the
    /// frame ran to its `return` and produced no value, which is a result the
    /// caller installs and the program observes; a bail says the opposite —
    /// the frame is mid-execution, its resume coordinate is already stamped
    /// into the interpreter frame, and the interpreter has to take it back.
    /// Spelling one as the other turned an abort into a Python-visible `None`.
    BailToInterpreter,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContinueRunningNormallyArgs {
    pub green_int: Vec<i64>,
    pub green_ref: Vec<i64>,
    pub green_float: Vec<i64>,
    pub red_int: Vec<i64>,
    pub red_ref: Vec<i64>,
    pub red_float: Vec<i64>,
}

/// Result of a completed trace execution.
///
/// Mirrors DoneWithThisFrame{Void,Int,Ref,Float} in jitexc.py.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DoneWithThisFrame {
    /// jitexc.py DoneWithThisFrameVoid
    Void,
    /// jitexc.py DoneWithThisFrameInt
    Int(i64),
    /// jitexc.py DoneWithThisFrameRef
    Ref(GcRef),
    /// jitexc.py DoneWithThisFrameFloat
    Float(f64),
}

impl DoneWithThisFrame {
    /// Create from a typed Value.
    pub fn from_value(v: Value) -> Self {
        match v {
            Value::Int(i) => DoneWithThisFrame::Int(i),
            Value::Ref(r) => DoneWithThisFrame::Ref(r),
            Value::Float(f) => DoneWithThisFrame::Float(f),
            Value::Void => DoneWithThisFrame::Void,
        }
    }
}

impl std::fmt::Display for DoneWithThisFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Void => f.write_str("DoneWithThisFrameVoid"),
            Self::Int(v) => write!(f, "DoneWithThisFrameInt({v})"),
            Self::Ref(r) => write!(f, "DoneWithThisFrameRef({:#x})", r.0),
            Self::Float(v) => write!(f, "DoneWithThisFrameFloat({:#x})", v.to_bits()),
        }
    }
}

impl std::error::Error for DoneWithThisFrame {}

/// The trace exited with an exception.
///
/// Mirrors ExitFrameWithExceptionRef in jitexc.py.
#[derive(Debug, Clone, PartialEq)]
pub struct ExitFrameWithException {
    pub exc_value: GcRef,
}

/// Request to continue running in the interpreter (leave JIT).
///
/// Mirrors ContinueRunningNormally in jitexc.py.
#[derive(Debug, Clone)]
pub struct ContinueRunningNormally {
    pub green_values: Vec<i64>,
    pub red_values: Vec<i64>,
}

impl ContinueRunningNormallyArgs {
    /// Green banks a merge point or header snapshot already grouped by kind.
    ///
    /// `warmspot.py handle_jitexception` reads `green_int` / `green_ref` /
    /// `green_float` in declaration order. The first int green is the
    /// portal pc (dispatch i0).
    pub fn from_green_banks(
        green_int: Vec<i64>,
        green_ref: Vec<i64>,
        green_float: Vec<i64>,
    ) -> Self {
        Self {
            green_int,
            green_ref,
            green_float,
            red_int: Vec::new(),
            red_ref: Vec::new(),
            red_float: Vec::new(),
        }
    }

    /// Inverse of a declared-green `GreenKey` (`can_enter_jit` cell key).
    ///
    /// `warmspot.py portalfunc_ARGS` asserts `kind != 'void'`. A short
    /// bank or a void green is a decode fault.
    pub fn from_declared_green_key(key: &GreenKey) -> Self {
        assert_eq!(
            key.types.len(),
            key.values.len(),
            "declared green key type/value lengths must match"
        );
        let mut green_int = Vec::new();
        let mut green_ref = Vec::new();
        let mut green_float = Vec::new();
        for (&tp, &value) in key.types.iter().zip(key.values.iter()) {
            match tp {
                GreenType::Int => green_int.push(value),
                GreenType::Void => {
                    panic!(
                        "void green cannot land in green_int \
                         (warmspot.py portalfunc_ARGS asserts kind != 'void')"
                    )
                }
                GreenType::Ref | GreenType::Str | GreenType::Unicode => green_ref.push(value),
                GreenType::Float => green_float.push(value),
            }
        }
        Self::from_green_banks(green_int, green_ref, green_float)
    }

    /// Portal pc: `green_int[0]`, the dispatch-entry i0 convention.
    pub fn portal_pc(&self) -> usize {
        let pc = *self
            .green_int
            .first()
            .expect("ContinueRunningNormally green_int[0] is the portal pc");
        usize::try_from(pc).unwrap_or(usize::MAX)
    }

    /// Copy the six banks into `self`, keeping each Vec's capacity.
    ///
    /// `blackhole.py` `recycle_merge_point_args` recovers the lists after
    /// `handle_jitexception` has read them. `Vec::clone` would drop that
    /// capacity; `Vec::clone_from` reuses it.
    pub fn copy_from_args(&mut self, other: &Self) {
        self.green_int.clone_from(&other.green_int);
        self.green_ref.clone_from(&other.green_ref);
        self.green_float.clone_from(&other.green_float);
        self.red_int.clone_from(&other.red_int);
        self.red_ref.clone_from(&other.red_ref);
        self.red_float.clone_from(&other.red_float);
    }

    /// Fill the green banks from a header snapshot, keeping capacity.
    pub fn copy_green_banks_from(&mut self, ints: &[i64], refs: &[i64], floats: &[i64]) {
        self.green_int.clear();
        self.green_int.extend_from_slice(ints);
        self.green_ref.clear();
        self.green_ref.extend_from_slice(refs);
        self.green_float.clear();
        self.green_float.extend_from_slice(floats);
        self.red_int.clear();
        self.red_ref.clear();
        self.red_float.clear();
    }
}

/// What `warmspot.py handle_jitexception` does after a compiled or
/// blackhole exit: re-enter the portal (`ContinueRunningNormally`),
/// return (`DoneWithThisFrame*`), or propagate
/// (`ExitFrameWithExceptionRef`). Distinct variants, not empty banks.
///
/// Each arm that still runs the original function — the native loop or
/// its epilogue — carries the six lists so loop-carried greens are
/// assigned before that code reads them.
#[derive(Clone, Debug, PartialEq)]
pub enum PortalResume {
    /// jitexc.py ContinueRunningNormally
    ContinueRunningNormally(ContinueRunningNormallyArgs),
    /// jitexc.py DoneWithThisFrame*
    DoneWithThisFrame(ContinueRunningNormallyArgs),
    /// pyre-only BailToInterpreter
    BailToInterpreter(ContinueRunningNormallyArgs),
    /// jitexc.py ExitFrameWithExceptionRef
    ExitFrameWithException {
        pc: usize,
        args: ContinueRunningNormallyArgs,
    },
    /// Successful compiled-loop JUMP, a tracing start, or a counter tick.
    ///
    /// The JUMP path has already passed the loop header's `guard_value` on
    /// every loop-carried green (`jtransform.py promote_greens`), so the
    /// native locals match those greens. A tick does not run compiled
    /// code. Every other exit carries the six `ContinueRunningNormally`
    /// lists (`warmspot.py handle_jitexception`).
    ResumeAt(usize),
}

impl PortalResume {
    pub fn args(&self) -> Option<&ContinueRunningNormallyArgs> {
        match self {
            Self::ContinueRunningNormally(args)
            | Self::DoneWithThisFrame(args)
            | Self::BailToInterpreter(args)
            | Self::ExitFrameWithException { args, .. } => Some(args),
            Self::ResumeAt(_) => None,
        }
    }

    /// Take the banks so the driver can return them to
    /// `recycle_merge_point_args` capacity (`portal_resume_scratch`).
    pub fn into_args(self) -> Option<ContinueRunningNormallyArgs> {
        match self {
            Self::ContinueRunningNormally(args)
            | Self::DoneWithThisFrame(args)
            | Self::BailToInterpreter(args)
            | Self::ExitFrameWithException { args, .. } => Some(args),
            Self::ResumeAt(_) => None,
        }
    }

    /// `Some(pc)` keeps the native dispatch loop running. `None` ends it
    /// so the function epilogue runs (`DoneWithThisFrame*`).
    pub fn resume_pc(&self) -> Option<usize> {
        match self {
            Self::ContinueRunningNormally(args) => {
                let pc = args.portal_pc();
                (pc != usize::MAX).then_some(pc)
            }
            Self::DoneWithThisFrame(_) | Self::BailToInterpreter(_) => None,
            Self::ExitFrameWithException { pc, .. } => (*pc != usize::MAX).then_some(*pc),
            Self::ResumeAt(pc) => (*pc != usize::MAX).then_some(*pc),
        }
    }
}

/// Inverse of `majit_ir::GreenAsI64` for a plain owned value.
///
/// The `#[jit_interp]` merge-point expansion assigns a loop-local green
/// from the `ContinueRunningNormally` banks. `as _` does not compile for
/// `bool`. A reference cannot be rebuilt from bits, so `&T` / `&mut T`
/// are not implemented.
pub trait GreenFromI64: Sized {
    fn from_green_i64(bits: i64) -> Self;
}

macro_rules! impl_green_from_i64_int {
    ($($ty:ty),*) => {
        $(
            impl GreenFromI64 for $ty {
                #[inline(always)]
                fn from_green_i64(bits: i64) -> Self {
                    bits as $ty
                }
            }
        )*
    };
}

impl_green_from_i64_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl GreenFromI64 for bool {
    #[inline(always)]
    fn from_green_i64(bits: i64) -> Self {
        bits != 0
    }
}

impl GreenFromI64 for f64 {
    #[inline(always)]
    fn from_green_i64(bits: i64) -> Self {
        f64::from_bits(bits as u64)
    }
}

impl GreenFromI64 for f32 {
    #[inline(always)]
    fn from_green_i64(bits: i64) -> Self {
        // `GreenAsI64 for f32` stores `(self as f64).to_bits()`.
        f64::from_bits(bits as u64) as f32
    }
}

/// The loop is not vectorizable.
///
/// Mirrors NotAVectorizeableLoop in jitexc.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotAVectorizeableLoop;

/// The loop is not profitable to vectorize.
///
/// Mirrors NotAProfitableLoop in jitexc.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotAProfitableLoop;
