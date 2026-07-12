/// JIT IR operations, faithfully translated from rpython/jit/metainterp/resoperation.py.
///
/// Operations with multiple result types (e.g., SAME_AS/1/ifr) are expanded
/// into type-suffixed variants (SameAsI, SameAsR, SameAsF).
///
/// Naming convention: CamelCase variant name, with type suffix I/R/F/N where applicable.
use smallvec::SmallVec;

use crate::descr::DescrRef;
use crate::forwarding::Forwarded;
use crate::operand::Operand;
use crate::value::{GcRef, Type, Value};

/// Index into an operation list, used as a reference to an operation's
/// result. Variant-tagged enum mirroring RPython's `AbstractValue` class
/// hierarchy (resoperation.py:29 + history.py:182).
///
/// Each typed variant carries the same raw u32 encoding shape (`CONST_BIT`
/// set for `Const*`, plain `pos` for `InputArg*` / `*Op`). The variant
/// tag IS the `box.type` (history.py:220 / resoperation.py:1693
/// `opclasses[opnum].type`); flat-OpRef encoding picks up Box-class
/// identity from the enum discriminant.
///
/// `PartialEq` / `Eq` / `Hash` include the enum variant, not just `.raw()`.
/// This keeps the disjoint RPython Box classes disjoint even when Pyre's
/// flat encoding reuses the same raw position across InputArg / ResOp /
/// Const namespaces. `ConstFloat` uses `f64::to_bits()` for
/// equality, hashing, and ordering to mirror RPython `ConstFloat._get_hash_`
/// / `same_constant` (history.py:283/292), where `0.0 != -0.0` and
/// `NaN == NaN` when their bit patterns agree. The `Ord`/`PartialOrd` impls
/// (kept for `vecset::VecMap<OpRef, _>` sorted-Vec storage) lift the same
/// bit-level total order over inline-f64 payloads.
#[derive(Clone, Copy, Debug)]
pub enum OpRef {
    /// Sentinel for missing/absent reference; `OpRef::NONE` aliases this.
    /// RPython has no equivalent — missing values are Python `None`.
    None,
    /// history.py:227 `ConstInt.value` carried inline as `i64`. Strict
    /// RPython parity: `Const{Int,Float,Ptr}.value` are inline value
    /// attributes on the Box class itself (history.py:227/268/314), no
    /// side-table lookup.
    ConstInt(i64),
    /// history.py:268 `ConstFloat.value` carried inline as `f64`. See
    /// `ConstInt`. Equality / Hash bitwise via `f64::to_bits()`
    /// per RPython `_get_hash_` / `same_constant` (history.py:283/292).
    ConstFloat(f64),
    /// history.py:314 `ConstPtr.value` carried inline as `GcRef`. See
    /// `ConstInt`. The inline `GcRef` lives directly in the `OpRef` and
    /// must be visited by the GC walker that traces the op-graph.
    ConstPtr(GcRef),
    /// resoperation.py:719 `InputArgInt` — `type = 'i'`. Payload: input
    /// arg slot position.
    InputArgInt(u32),
    /// resoperation.py:727 `InputArgFloat` — `type = 'f'`.
    InputArgFloat(u32),
    /// resoperation.py:739 `InputArgRef` — `type = 'r'`.
    InputArgRef(u32),
    /// `AbstractResOp` + `IntOp` mixin — `type = 'i'`. Payload: op
    /// result OpRef position.
    IntOp(u32),
    /// `AbstractResOp` + `FloatOp` mixin — `type = 'f'`.
    FloatOp(u32),
    /// `AbstractResOp` + `RefOp` mixin — `type = 'r'`.
    RefOp(u32),
    /// `AbstractResOp` default — `type = 'v'` (resoperation.py:260).
    /// Void-result ops (SETFIELD_GC, GUARD_*, JUMP, …) carry no result
    /// type but still occupy an op position.
    VoidOp(u32),
    /// Backend regalloc scratch box — RPython `TempVar()` /
    /// `TempInt()` parity (`rpython/jit/backend/llsupport/regalloc.py`,
    /// `x86/regalloc.py:470,514,521,605`,
    /// `aarch64/regalloc.py:990`). Each call to
    /// `RegAlloc::fresh_temp_var()` allocates a fresh `TempVar`
    /// carrying a unique counter; the raw payload lives in the
    /// reserved range `[SENTINEL_BASE, u32::MAX - 1]` so it does not
    /// collide with constant-namespace or op-position OpRefs. Lifetime
    /// is single-instruction: `force_allocate_reg` then
    /// `possibly_free_var` within one `consider_*` body.
    TempVar(u32),
}

impl PartialEq for OpRef {
    fn eq(&self, other: &Self) -> bool {
        use OpRef::*;
        match (self, other) {
            (None, None) => true,
            (InputArgInt(a), InputArgInt(b))
            | (InputArgFloat(a), InputArgFloat(b))
            | (InputArgRef(a), InputArgRef(b))
            | (IntOp(a), IntOp(b))
            | (FloatOp(a), FloatOp(b))
            | (RefOp(a), RefOp(b))
            | (VoidOp(a), VoidOp(b))
            | (TempVar(a), TempVar(b)) => a == b,
            (ConstInt(a), ConstInt(b)) => a == b,
            // history.py:292 ConstFloat.same_constant: bitwise compare
            (ConstFloat(a), ConstFloat(b)) => a.to_bits() == b.to_bits(),
            (ConstPtr(a), ConstPtr(b)) => a.0 == b.0,
            _ => false,
        }
    }
}

impl Eq for OpRef {}

impl std::hash::Hash for OpRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            OpRef::None => {}
            OpRef::InputArgInt(x)
            | OpRef::InputArgFloat(x)
            | OpRef::InputArgRef(x)
            | OpRef::IntOp(x)
            | OpRef::FloatOp(x)
            | OpRef::RefOp(x)
            | OpRef::VoidOp(x)
            | OpRef::TempVar(x) => x.hash(state),
            OpRef::ConstInt(v) => v.hash(state),
            // history.py:283 ConstFloat._get_hash_: bitwise
            OpRef::ConstFloat(v) => v.to_bits().hash(state),
            OpRef::ConstPtr(v) => v.0.hash(state),
        }
    }
}

impl OpRef {
    /// Lift to a total-order key `(discriminant_index, payload_bits)`.
    /// Inline-f64 uses `to_bits()` (same bitwise total order as `Eq`/`Hash`)
    /// so the impl can compose into `vecset::VecMap<OpRef, _>` sorted-Vec
    /// storage without a partial-order escape hatch.
    fn ord_key(&self) -> (u8, u64) {
        match *self {
            OpRef::None => (0, 0),
            OpRef::ConstInt(v) => (4, v as u64),
            OpRef::ConstFloat(v) => (5, v.to_bits()),
            OpRef::ConstPtr(v) => (6, v.0 as u64),
            OpRef::InputArgInt(x) => (7, x as u64),
            OpRef::InputArgFloat(x) => (8, x as u64),
            OpRef::InputArgRef(x) => (9, x as u64),
            OpRef::IntOp(x) => (10, x as u64),
            OpRef::FloatOp(x) => (11, x as u64),
            OpRef::RefOp(x) => (12, x as u64),
            OpRef::VoidOp(x) => (13, x as u64),
            OpRef::TempVar(x) => (14, x as u64),
        }
    }
}

impl PartialOrd for OpRef {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OpRef {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ord_key().cmp(&other.ord_key())
    }
}

impl OpRef {
    pub const NONE: OpRef = OpRef::None;
    /// High bit distinguishes constant-namespace OpRefs from operation OpRefs.
    /// opencoder.py: TAGINT/TAGCONSTPTR/TAGCONSTOTHER/TAGBOX use 2-bit tags;
    /// here a single high bit suffices (op vs const).
    const CONST_BIT: u32 = 1 << 31;
    /// Top of the u32 range reserved for `TempVar` (regalloc scratch)
    /// OpRefs. RPython `TempVar()` (`backend/llsupport/regalloc.py:18-23`,
    /// `__init__` body is `pass`, `__repr__` keys off `id(self)`) only
    /// carries Python object identity, so collision is structurally
    /// impossible upstream. pyre's flat-OpRef encoding cannot mint fresh
    /// objects, so it reserves the high u32 strip `[SENTINEL_BASE,
    /// u32::MAX - 1]` and assigns a unique counter per `fresh_temp_var()`
    /// call (raw = `SENTINEL_BASE | counter`, counter in `[0, 0xFFFE]`).
    /// The top sentinel `u32::MAX` is reserved for `OpRef::None`.
    ///
    /// Note: `SENTINEL_BASE & CONST_BIT != 0` — the raw payload of
    /// every `TempVar` carries `CONST_BIT`. Disambiguation is done two
    /// ways: variant-match `is_constant()` returns `false` on
    /// `TempVar(_)`, and the raw-bit-helper `raw_is_constant()` further
    /// rejects the sentinel strip via `raw < SENTINEL_BASE`. So the
    /// two namespaces are NOT raw-bit disjoint, they are
    /// variant-disjoint and range-disjoint.
    const SENTINEL_BASE: u32 = 0xFFFF_0000;

    pub fn is_none(self) -> bool {
        matches!(self, Self::None)
    }

    /// Mirrors RPython `isinstance(value, AbstractInputArg)` — the operand is
    /// one of the trace's typed input args (a loop/function entry value), not a
    /// recorded op result, constant, or temp.
    pub fn is_input_arg(self) -> bool {
        matches!(
            self,
            Self::InputArgInt(_) | Self::InputArgFloat(_) | Self::InputArgRef(_)
        )
    }

    /// Extract the raw u32 payload. For `None` returns `u32::MAX` to
    /// preserve pre-Phase-3 round-trip semantics.
    ///
    /// `Const{Int,Float,Ptr}` variants carry an inline value
    /// (`i64`/`f64`/`GcRef`) and have no u32 raw encoding. Calling
    /// `raw()` on them is a bug; it panics loud rather than silently
    /// truncate. Use `as_const_int` / `as_const_float` / `as_const_ptr`
    /// to read inline payloads.
    #[inline]
    #[track_caller]
    pub fn raw(self) -> u32 {
        match self {
            Self::None => u32::MAX,
            Self::InputArgInt(x)
            | Self::InputArgFloat(x)
            | Self::InputArgRef(x)
            | Self::IntOp(x)
            | Self::FloatOp(x)
            | Self::RefOp(x)
            | Self::VoidOp(x)
            | Self::TempVar(x) => x,
            Self::ConstInt(_) | Self::ConstFloat(_) | Self::ConstPtr(_) => {
                panic!(
                    "OpRef::raw() called on inline-Const variant {:?}; \
                     use as_const_*_inline accessor instead",
                    self
                )
            }
        }
    }

    /// Mirrors RPython `AbstractValue.type` — the type embedded in the
    /// variant tag for `Const{Int,Float,Ptr}`, `InputArg{Int,Float,Ref}`,
    /// and the `{Int,Float,Ref,Void}Op` mixins (history.py:220 / 261 /
    /// 307, resoperation.py:567 / 589 / 615 / 260). `None` returns
    /// `None`.
    ///
    /// `TempVar` also returns `None`: RPython's `TempVar`
    /// (`backend/llsupport/regalloc.py:18`) extends `AbstractResOpOrInputArg`
    /// without a `.type` attribute, and `_check_type` at
    /// `regalloc.py:405-407` exempts it via `isinstance(v, TempVar)`. A
    /// `TempVar` reaching `.ty()` should fall through to the regalloc-side
    /// `is_temp_var()` exemption rather than masquerade as an integer box.
    pub fn ty(self) -> Option<Type> {
        match self {
            Self::None | Self::TempVar(_) => None,
            Self::ConstInt(_) | Self::InputArgInt(_) | Self::IntOp(_) => Some(Type::Int),
            Self::ConstFloat(_) | Self::InputArgFloat(_) | Self::FloatOp(_) => Some(Type::Float),
            Self::ConstPtr(_) | Self::InputArgRef(_) | Self::RefOp(_) => Some(Type::Ref),
            Self::VoidOp(_) => Some(Type::Void),
        }
    }

    /// resoperation.py:47 `AbstractValue.is_constant()` returns False;
    /// history.py:213 `Const.is_constant()` returns True. The
    /// dispatch is class-based — typed body variants
    /// (`IntOp/RefOp/FloatOp/VoidOp/InputArg*`) correspond to
    /// `AbstractValue` subclasses and are NOT constants.
    ///
    /// A typed body variant with `CONST_BIT` in its payload is a
    /// namespace invariant violation — the constant namespace
    /// (`ConstInt/ConstFloat/ConstPtr`) and the body namespace must
    /// stay disjoint at construction time. Fail loud at the consumer
    /// rather than silently classifying as a constant.
    pub fn is_constant(self) -> bool {
        match self {
            Self::ConstInt(_) | Self::ConstFloat(_) | Self::ConstPtr(_) => true,
            // `TempVar` lives in the reserved `[SENTINEL_BASE, u32::MAX - 1]`
            // sentinel range. The raw payload DOES carry `CONST_BIT`
            // (`SENTINEL_BASE = 0xFFFF_0000 = CONST_BIT | 0x7FFF_0000`), but
            // variant-match returns `false` here, and the raw-bit helper
            // `raw_is_constant()` further rejects the sentinel range via
            // `raw < SENTINEL_BASE`.
            Self::None | Self::TempVar(_) => false,
            Self::IntOp(x)
            | Self::RefOp(x)
            | Self::FloatOp(x)
            | Self::VoidOp(x)
            | Self::InputArgInt(x)
            | Self::InputArgRef(x)
            | Self::InputArgFloat(x) => {
                debug_assert!(
                    !Self::raw_is_constant(x),
                    "typed body OpRef {:?} carries CONST_BIT payload {:#x}: \
                     namespace invariant violation — body and const namespaces must \
                     stay disjoint (history.py:213 vs resoperation.py:47)",
                    self,
                    x
                );
                false
            }
        }
    }

    /// Bit-helper variant of `is_constant()` for callers that hold a raw
    /// u32 from an index-keyed pool (constant pool key, opencoder tag,
    /// etc.) and only need to test the constant-namespace bit. Stays raw
    /// u32 because the underlying pool (`HashMap<u32, V>`) is genuinely
    /// index-keyed.
    pub const fn raw_is_constant(raw: u32) -> bool {
        raw & Self::CONST_BIT != 0 && raw < Self::SENTINEL_BASE
    }

    /// Bit-helper variant of `const_index()` for callers that hold a raw
    /// u32 known to be a constant-namespace key.  See `raw_is_constant`
    /// for context.
    pub const fn raw_const_index(raw: u32) -> u32 {
        debug_assert!(Self::raw_is_constant(raw));
        raw & !Self::CONST_BIT
    }

    // ── Typed constructors mirroring RPython AbstractValue variants ──
    //
    // Each factory produces the matching enum variant carrying the
    // raw u32 encoding. These are the canonical OpRef
    // construction entry points; the variant tag IS the RPython Box
    // class identity (history.py:182 / resoperation.py:29).

    /// resoperation.py:719 `InputArgInt` — `type = 'i'`.
    pub const fn input_arg_int(pos: u32) -> OpRef {
        OpRef::InputArgInt(pos)
    }

    /// resoperation.py:727 `InputArgFloat` — `type = 'f'`.
    pub const fn input_arg_float(pos: u32) -> OpRef {
        OpRef::InputArgFloat(pos)
    }

    /// resoperation.py:739 `InputArgRef` — `type = 'r'`.
    pub const fn input_arg_ref(pos: u32) -> OpRef {
        OpRef::InputArgRef(pos)
    }

    /// `AbstractResOp` + `IntOp` mixin — `type = 'i'`.
    pub const fn int_op(pos: u32) -> OpRef {
        OpRef::IntOp(pos)
    }

    /// `AbstractResOp` + `FloatOp` mixin — `type = 'f'`.
    pub const fn float_op(pos: u32) -> OpRef {
        OpRef::FloatOp(pos)
    }

    /// `AbstractResOp` + `RefOp` mixin — `type = 'r'`.
    pub const fn ref_op(pos: u32) -> OpRef {
        OpRef::RefOp(pos)
    }

    /// `AbstractResOp` default — `type = 'v'` (resoperation.py:260).
    pub const fn void_op(pos: u32) -> OpRef {
        OpRef::VoidOp(pos)
    }

    /// Lower a Const OpRef variant to its raw `i64` payload, matching the
    /// wire shape of the `set_constants(HashMap<u32, i64>)` backend
    /// boundary. Returns `None` for non-Const OpRefs.
    ///
    /// Backends use this as a guard before any `.raw()` call on a
    /// constant: Const variants carry the value directly
    /// (history.py:227/268/314) and have no u32 raw encoding.
    pub const fn inline_const_bits(self) -> Option<i64> {
        match self {
            Self::ConstInt(v) => Some(v),
            Self::ConstFloat(v) => Some(v.to_bits() as i64),
            Self::ConstPtr(v) => Some(v.0 as i64),
            _ => None,
        }
    }

    /// Integer value of an Int-typed constant — `ConstInt.getint()`
    /// (history.py:240). Returns `None` for every non-Int operand:
    /// `ConstFloat` / `ConstPtr` have no `getint` (history.py:268/314).
    ///
    /// Unlike `inline_const_bits`, this rejects `ConstFloat` and
    /// `ConstPtr`. Use it where the operand is statically an
    /// Int-typed constant — e.g. GuardClass / GuardNonnullClass /
    /// GuardSubclass class operands, which are read with `getint()` by
    /// RPython backends and carry vtable addresses as raw integers so the
    /// GC never traces them.
    pub const fn const_int_value(self) -> Option<i64> {
        match self {
            Self::ConstInt(v) => Some(v),
            _ => None,
        }
    }

    /// Reverse of `const_inline_from_value`: extract the typed `Value`
    /// from an inline-Const OpRef. Returns `None` for non-inline variants.
    pub fn inline_const_to_value(self) -> Option<Value> {
        match self {
            Self::ConstInt(v) => Some(Value::Int(v)),
            Self::ConstFloat(v) => Some(Value::Float(v)),
            Self::ConstPtr(v) => Some(Value::Ref(v)),
            _ => None,
        }
    }

    /// Allocate a typed `InputArg*` OpRef from a position. The type tag
    /// picks the matching variant (resoperation.py:719/727/739).
    /// `Type::Void` is rejected — RPython has no Void inputarg class.
    pub fn input_arg_typed(pos: u32, tp: Type) -> OpRef {
        match tp {
            Type::Int => OpRef::input_arg_int(pos),
            Type::Float => OpRef::input_arg_float(pos),
            Type::Ref => OpRef::input_arg_ref(pos),
            Type::Void => panic!("Void input args are not supported"),
        }
    }

    /// Build the `[InputArg*(0), InputArg*(1), ...]` vector for a trace
    /// whose inputarg types are `types`.  Position is the slot index.
    /// resoperation.py:719/727/739 InputArg{Int,Ref,Float}: RPython has
    /// no InputArgVoid class.
    ///
    /// # Panics
    ///
    /// Panics if `types` contains `Type::Void`.
    pub fn inputarg_refs(types: &[Type]) -> Vec<OpRef> {
        debug_assert!(
            types.iter().all(|t| *t != Type::Void),
            "inputarg_refs: Type::Void is not a valid InputArg type \
             (resoperation.py:719/727/739 has no InputArgVoid)"
        );
        types
            .iter()
            .enumerate()
            .map(|(i, t)| OpRef::input_arg_typed(i as u32, *t))
            .collect()
    }

    /// Allocate a typed `*Op` OpRef from a position. The type tag picks
    /// the matching mixin variant (resoperation.py:564-638).
    /// `Type::Void` lands on `VoidOp` — `AbstractResOp.type = 'v'`
    /// (resoperation.py:260), the default for ops with no result-type
    /// mixin.
    pub const fn op_typed(pos: u32, tp: Type) -> OpRef {
        match tp {
            Type::Int => OpRef::int_op(pos),
            Type::Float => OpRef::float_op(pos),
            Type::Ref => OpRef::ref_op(pos),
            Type::Void => OpRef::void_op(pos),
        }
    }

    /// RPython `TempVar()` / `TempInt()` parity
    /// (`rpython/jit/backend/llsupport/regalloc.py:18-23`,
    /// `x86/regalloc.py:470,514,521,605`,
    /// `aarch64/regalloc.py:990`). Upstream `TempVar.__init__` is
    /// `pass`, so each instance is a fresh Python object with unique
    /// `id(self)` identity and collision is structurally impossible.
    /// pyre's flat-OpRef encoding emulates that by minting a unique
    /// `OpRef::TempVar(SENTINEL_BASE | counter)` per call.
    ///
    /// `counter` must fit in 16 bits (`[0, 0xFFFE]`), giving 65535
    /// slots in the `[SENTINEL_BASE, u32::MAX - 1]` strip (with
    /// `u32::MAX` reserved for `OpRef::None`). The per-trace counter
    /// lives on `RegAlloc::temp_var_counter` and is incremented per
    /// call. Realistic `consider_*` bodies allocate one or two
    /// `TempVar()` each, well under the 65535 cap — but exhaustion
    /// would silently collide upstream-impossible state, so we panic
    /// loud to catch the bookkeeping bug.
    pub fn fresh_temp_var(counter: u32) -> OpRef {
        assert!(
            counter < 0xFFFF,
            "OpRef::fresh_temp_var counter exhausted (>= 0xFFFF); \
             reserved range is [0, 0xFFFE], raw = SENTINEL_BASE | counter. \
             RPython TempVar uses object identity so collision is impossible \
             upstream — pyre's flat-encoding cap would alias TempVars at this point."
        );
        OpRef::TempVar(Self::SENTINEL_BASE | counter)
    }

    /// True if this OpRef is a `TempVar` regalloc scratch box.
    pub fn is_temp_var(self) -> bool {
        matches!(self, Self::TempVar(_))
    }

    /// `resoperation.py:38 AbstractValue.same_box(other)` plus the
    /// `Const.same_box` override (history.py:211 → `same_constant`). A flat
    /// `OpRef` is a tagged handle and `==` compares variant + payload, so a
    /// single `==` already covers both: for non-Const variants it is
    /// position equality (the base `self is other`), and for
    /// `Const{Int,Float,Ptr}` it is inline value equality
    /// (`Const.same_constant`, including history.py:292 bitwise float so
    /// `0.0 != -0.0`). This is the explicit API name so callers don't reach
    /// for `==`.
    ///
    /// The same split is mirrored on `Operand::same_box` (identity for
    /// ResOp/InputArg, value for Const) reached via `OptContext::same_box`,
    /// for callers holding a stable `Rc<Box>` handle; callers comparing
    /// Const values may use that or `ConstOprefOracle` directly.
    #[inline]
    pub fn same_box(self, other: OpRef) -> bool {
        self == other
    }

    /// Re-encode this OpRef's variant with a fresh raw payload while
    /// preserving the type tag. Used by post-optimization remaps that
    /// renumber positions but keep RPython's `box.type` attached
    /// (history.py:802 record_same_as parity, where the remapped Box
    /// inherits the source Box's `.type`).
    ///
    /// `None` round-trips to `None` regardless of `new_raw`.
    pub const fn with_raw(self, new_raw: u32) -> OpRef {
        match self {
            Self::None => Self::None,
            Self::InputArgInt(_) => Self::InputArgInt(new_raw),
            Self::InputArgFloat(_) => Self::InputArgFloat(new_raw),
            Self::InputArgRef(_) => Self::InputArgRef(new_raw),
            Self::IntOp(_) => Self::IntOp(new_raw),
            Self::FloatOp(_) => Self::FloatOp(new_raw),
            Self::RefOp(_) => Self::RefOp(new_raw),
            Self::VoidOp(_) => Self::VoidOp(new_raw),
            Self::TempVar(_) => Self::TempVar(new_raw),
            Self::ConstInt(_) | Self::ConstFloat(_) | Self::ConstPtr(_) => {
                panic!(
                    "OpRef::with_raw() called on inline-Const variant; \
                     inline payload has no u32 raw encoding"
                )
            }
        }
    }

    // ── Const factories + accessors ──
    //
    // Strict RPython parity: `Const{Int,Float,Ptr}.value` are inline
    // attributes on the Box class (history.py:227/268/314). These
    // factories mint an OpRef variant carrying the value directly.

    /// history.py:227 `ConstInt.value: int` carried inline.
    pub const fn const_int(v: i64) -> OpRef {
        OpRef::ConstInt(v)
    }

    /// history.py:268 `ConstFloat.value: float` carried inline.
    pub const fn const_float(v: f64) -> OpRef {
        OpRef::ConstFloat(v)
    }

    /// history.py:314 `ConstPtr.value: gcref` carried inline.
    pub const fn const_ptr(v: GcRef) -> OpRef {
        OpRef::ConstPtr(v)
    }

    /// Mint an inline-Const OpRef from a `Value` per RPython
    /// `history.py:227/268/314` Const{Int,Float,Ptr}.value (the value
    /// lives inline on the Box). `Value::Void` panics — RPython has no
    /// `ConstVoid` (resoperation.py defines Const subclasses only for
    /// Int/Float/Ref).
    pub fn const_inline_from_value(value: &Value) -> OpRef {
        match value {
            Value::Int(i) => OpRef::ConstInt(*i),
            Value::Float(f) => OpRef::ConstFloat(*f),
            Value::Ref(r) => OpRef::ConstPtr(*r),
            Value::Void => {
                panic!("Value::Void has no Const subclass per resoperation.py / history.py")
            }
        }
    }

    /// Extract the inline `i64` from `ConstInt`; `None` for any
    /// other variant.
    pub const fn as_const_int(self) -> Option<i64> {
        match self {
            Self::ConstInt(v) => Some(v),
            _ => None,
        }
    }

    /// Extract the inline `f64` from `ConstFloat`; `None` for any
    /// other variant.
    pub const fn as_const_float(self) -> Option<f64> {
        match self {
            Self::ConstFloat(v) => Some(v),
            _ => None,
        }
    }

    /// Extract the inline `GcRef` from `ConstPtr`; `None` for any
    /// other variant.
    pub const fn as_const_ptr(self) -> Option<GcRef> {
        match self {
            Self::ConstPtr(v) => Some(v),
            _ => None,
        }
    }
}

// `#[derive(PartialEq, Eq, Hash)]` on `OpRef` enforces RPython's
// disjoint `Const` / `InputArg` / `ResOp` sub-hierarchies
// (resoperation.py:29, history.py:182): two variants compare unequal
// even when raw payloads coincide (`ConstInt(x) != ConstFloat(x) !=
// IntOp(x)`). Mirrors `AbstractValue.same_box` (resoperation.py:38
// `self is other`) and `ConstInt.same_constant` (history.py:244).

/// AbstractValue parity: rpython/jit/metainterp/resoperation.py:29
/// + history.py:182.
///
/// RPython's `AbstractValue` is the root of the value hierarchy that
/// carries `type` ('i' / 'r' / 'f') as a class-level constant. The
/// concrete subclasses split into three families:
///
/// 1. **`Const` family** (history.py:220 `ConstInt`, history.py:261
///    `ConstFloat`, history.py:307 `ConstPtr`).
/// 2. **`AbstractInputArg` family** (resoperation.py:719 `InputArgInt`,
///    resoperation.py:727 `InputArgFloat`, resoperation.py:739
///    `InputArgRef`).
/// 3. **`AbstractResOp`** (resoperation.py:250) mixed with one of the
///    `IntOp` / `FloatOp` / `RefOp` mixins (resoperation.py:564-638) —
///    every concrete ResOp subclass picks up its `type` attribute via
///    one of these three mixins.
///
/// In pyre, [`OpRef`] is the typed-variant model — each variant
/// (`ConstInt` / `ConstFloat` / `ConstPtr` / `InputArgInt` /
/// `InputArgFloat` / `InputArgRef` / `IntOp` / `FloatOp` / `RefOp` /
/// `VoidOp`) encodes the RPython class's `type` attribute, so an
/// `OpRef::IntOp(5)` and an `OpRef::RefOp(5)` are distinct identities
/// even when their `raw()` payload matches.  `AbstractValue` is the
/// Rust analogue of an instantiated RPython value: `type` is
/// intrinsic to the variant, matching the class-level `type` attribute
/// upstream.
///
/// The `None` variant is a Rust adaptation for missing/sentinel
/// references; in RPython missing values are Python `None` or absent
/// attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AbstractValue {
    None,
    /// history.py:220 `ConstInt` — `type = 'i'`.
    ConstInt(u32),
    /// history.py:261 `ConstFloat` — `type = 'f'`.
    ConstFloat(u32),
    /// history.py:307 `ConstPtr` — `type = 'r'`.
    ConstPtr(u32),
    /// resoperation.py:719 `InputArgInt` — `type = 'i'`. Payload: input
    /// arg slot position.
    InputArgInt(u32),
    /// resoperation.py:727 `InputArgFloat` — `type = 'f'`.
    InputArgFloat(u32),
    /// resoperation.py:739 `InputArgRef` — `type = 'r'`.
    InputArgRef(u32),
    /// `AbstractResOp` + `IntOp` mixin — `type = 'i'`. Payload: op
    /// result OpRef position.
    IntOp(u32),
    /// `AbstractResOp` + `FloatOp` mixin — `type = 'f'`.
    FloatOp(u32),
    /// `AbstractResOp` + `RefOp` mixin — `type = 'r'`.
    RefOp(u32),
    /// `AbstractResOp` default — `type = 'v'` (resoperation.py:260).
    /// Void-result ops (SETFIELD_GC, GUARD_*, JUMP, …).
    VoidOp(u32),
}

impl AbstractValue {
    /// Mirrors RPython `AbstractValue.type` access.
    pub fn ty(self) -> Option<Type> {
        match self {
            Self::None => None,
            Self::ConstInt(_) | Self::InputArgInt(_) | Self::IntOp(_) => Some(Type::Int),
            Self::ConstFloat(_) | Self::InputArgFloat(_) | Self::FloatOp(_) => Some(Type::Float),
            Self::ConstPtr(_) | Self::InputArgRef(_) | Self::RefOp(_) => Some(Type::Ref),
            Self::VoidOp(_) => Some(Type::Void),
        }
    }

    /// Mirrors RPython `isinstance(value, Const)`.
    pub fn is_constant(self) -> bool {
        matches!(
            self,
            Self::ConstInt(_) | Self::ConstFloat(_) | Self::ConstPtr(_)
        )
    }

    /// Mirrors RPython `isinstance(value, AbstractInputArg)`.
    pub fn is_input_arg(self) -> bool {
        matches!(
            self,
            Self::InputArgInt(_) | Self::InputArgFloat(_) | Self::InputArgRef(_)
        )
    }

    /// Mirrors RPython `isinstance(value, AbstractResOp)`.
    pub fn is_res_op(self) -> bool {
        matches!(
            self,
            Self::IntOp(_) | Self::FloatOp(_) | Self::RefOp(_) | Self::VoidOp(_)
        )
    }

    /// Returns the variant's raw u32 payload (input arg / op-result
    /// position; the `Const*` variants carry an opaque discriminant).
    /// `None` variant returns `None`.
    pub fn raw(self) -> Option<u32> {
        match self {
            Self::None => None,
            Self::ConstInt(x)
            | Self::ConstFloat(x)
            | Self::ConstPtr(x)
            | Self::InputArgInt(x)
            | Self::InputArgFloat(x)
            | Self::InputArgRef(x)
            | Self::IntOp(x)
            | Self::FloatOp(x)
            | Self::RefOp(x)
            | Self::VoidOp(x) => Some(x),
        }
    }
}

/// resume.py:576-860: virtual object serialization for rd_virtuals.
///
/// Each variant corresponds to a concrete virtual type in RPython's
/// resume.py:591-593 AbstractVirtualStructInfo.fielddescrs parity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldDescrInfo {
    pub index: u32,
    pub offset: usize,
    pub field_type: Type,
    pub field_size: usize,
}

/// Serializable snapshot of an ArrayDescr.
///
/// RPython's resume.py:692 VRawBufferInfo carries live ArrayDescr objects,
/// but we cannot put `Arc<dyn Descr>` in the IR serialization boundary.
/// This captures the fields needed by `_descrs_are_compatible()` (rawbuffer.py:83)
/// and `setrawbuffer_item()` dispatch (resume.py:1543).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayDescrInfo {
    /// Descriptor registry index.
    pub index: u32,
    /// descr.py:273 ArrayDescr.basesize.
    pub base_size: usize,
    /// descr.py:274 ArrayDescr.itemsize.
    pub item_size: usize,
    /// Item type: 0=ref, 1=int, 2=float.
    pub item_type: u8,
    /// descr.py:241-254 FLAG_SIGNED.
    pub is_signed: bool,
    /// descr.py:277 / descr.py:359-362 — `ArrayDescr.lendescr.offset`.
    /// `None` for the `nolength=True` shape (raw buffers); `Some(off)`
    /// for length-prefixed `Ptr(GcArray(T))`. Carries the live offset
    /// across the resume/materialization summary boundary so backends
    /// can read the length word from the same place the producer wrote.
    pub len_offset: Option<usize>,
}

/// AbstractVirtualInfo hierarchy (VirtualInfo, VStructInfo, VArrayInfo, etc.).
#[derive(Clone, Debug)]
pub enum RdVirtualInfo {
    /// resume.py:612 VirtualInfo(descr, fielddescrs).
    VirtualInfo {
        /// resume.py:615 self.descr — live SizeDescr reference.
        descr: Option<crate::DescrRef>,
        /// descr.tid — GC type identifier for allocation dispatch.
        type_id: u32,
        known_class: Option<i64>,
        fielddescrs: Vec<FieldDescrInfo>,
        fieldnums: Vec<i16>,
        descr_size: usize,
    },
    /// resume.py:628 VStructInfo(typedescr, fielddescrs).
    VStructInfo {
        /// resume.py:631 self.typedescr — live SizeDescr reference.
        typedescr: Option<crate::DescrRef>,
        /// typedescr.tid — GC type identifier (cached for serialization).
        type_id: u32,
        fielddescrs: Vec<FieldDescrInfo>,
        fieldnums: Vec<i16>,
        descr_size: usize,
    },
    /// resume.py:680: VArrayInfoClear (clear=True)
    VArrayInfoClear {
        /// resume.py:646 self.arraydescr — live ArrayDescr reference.
        arraydescr: Option<crate::DescrRef>,
        /// resume.py:656: arraydescr element kind (ref/int/float).
        kind: u8, // 0=ref, 1=int, 2=float (ArrayDescr.flag parity)
        fieldnums: Vec<i16>,
    },
    /// resume.py:683: VArrayInfoNotClear (clear=False)
    VArrayInfoNotClear {
        /// resume.py:646 self.arraydescr — live ArrayDescr reference.
        arraydescr: Option<crate::DescrRef>,
        /// resume.py:656: arraydescr element kind (ref/int/float).
        kind: u8, // 0=ref, 1=int, 2=float (ArrayDescr.flag parity)
        fieldnums: Vec<i16>,
    },
    /// resume.py:736: VArrayStructInfo
    VArrayStructInfo {
        /// resume.py:739 self.arraydescr — live ArrayDescr reference.
        arraydescr: Option<crate::DescrRef>,
        size: usize,
        /// resume.py:740: self.fielddescrs — live InteriorFieldDescr objects.
        fielddescrs: Vec<crate::DescrRef>,
        /// resume.py VArrayStructInfo.fielddescrs — per-field descriptor indices.
        fielddescr_indices: Vec<u32>,
        /// resume.py:757: fielddescrs[j].is_pointer_field/is_float_field dispatch.
        /// Per-field type within each element: 0=ref, 1=int, 2=float.
        field_types: Vec<u8>,
        /// descr.py:273 ArrayDescr.basesize — fixed header before array items.
        base_size: usize,
        /// llmodel.py:648: arraydescr.itemsize — bytes per struct element.
        item_size: usize,
        /// llmodel.py:649: fielddescr.offset — per-field byte offset within struct.
        field_offsets: Vec<usize>,
        /// llmodel.py:649: fielddescr.field_size — per-field byte width.
        field_sizes: Vec<usize>,
        fieldnums: Vec<i16>,
    },
    /// resume.py:692: VRawBufferInfo(func, size, offsets, descrs)
    VRawBufferInfo {
        /// resume.py:695: self.func — raw malloc function pointer.
        func: i64,
        size: usize,
        /// resume.py:696: self.offsets — byte offsets of stored values.
        /// Signed because rawbuffer.py:14 stores offsets as RPython
        /// unbounded ints; with `index < 0`, `basesize + itemsize*index`
        /// is negative.
        offsets: Vec<i64>,
        /// resume.py:697: self.descrs — per-entry ArrayDescr snapshots.
        /// RPython carries live ArrayDescr objects; we carry serializable snapshots.
        descrs: Vec<ArrayDescrInfo>,
        fieldnums: Vec<i16>,
    },
    /// resume.py:717: VRawSliceInfo
    VRawSliceInfo {
        /// info.py:460: signed slice base — `optimize_INT_ADD` folds the
        /// addend as a signed `getint()`.
        offset: i64,
        fieldnums: Vec<i16>,
    },
    /// resume.py:763 `VStrPlainInfo` — virtual byte-string built from
    /// character fieldnums. `fieldnums` length = string length.
    VStrPlainInfo {
        fieldnums: Vec<i16>,
    },
    /// resume.py:781 `VStrConcatInfo` — virtual concatenation of two
    /// strings. `fieldnums = [left, right]`. The OS_STR_CONCAT funcptr
    /// is resolved at materialization time via
    /// `callinfocollection.funcptr_for_oopspec(OS_STR_CONCAT)`
    /// (resume.py:1467-1468); the variant carries no funcptr itself.
    VStrConcatInfo {
        fieldnums: Vec<i16>,
    },
    /// resume.py:801 `VStrSliceInfo` — virtual slice of a larger string.
    /// `fieldnums = [largerstr, start, length]` (pyre stores `length`;
    /// the backend reader converts to RPython's `(start, start + length)`
    /// before calling the OS_STR_SLICE funcptr — see
    /// `resume.py:1479` and `resume.rs::ResumeDataDirectReader::slice_string`).
    /// OS_STR_SLICE funcptr is resolved via callinfocollection at
    /// materialization time.
    VStrSliceInfo {
        fieldnums: Vec<i16>,
    },
    /// resume.py:817 `VUniPlainInfo` — unicode counterpart of VStrPlain.
    VUniPlainInfo {
        fieldnums: Vec<i16>,
    },
    /// resume.py:836 `VUniConcatInfo` — unicode counterpart of VStrConcat.
    /// OS_UNI_CONCAT funcptr is resolved via callinfocollection at
    /// materialization time.
    VUniConcatInfo {
        fieldnums: Vec<i16>,
    },
    /// resume.py:856 `VUniSliceInfo` — unicode counterpart of `VStrSlice`
    /// (same length-vs-stop convention; backend reader adds
    /// `start + length` before calling the OS_UNI_SLICE funcptr).
    /// OS_UNI_SLICE funcptr is resolved via callinfocollection at
    /// materialization time.
    VUniSliceInfo {
        fieldnums: Vec<i16>,
    },
    Empty,
}

/// `history.py:125` `id(descr)` parity — Option<Arc<dyn Descr>> identity
/// compare.  Both `None` are equal (unset slots); two `Some` are equal
/// iff their Arcs share the underlying object (`Arc::ptr_eq`).  Backs
/// the `RdVirtualInfo` / `GuardPendingFieldEntry` `PartialEq` impls so
/// resume-info canonicalisation matches PyPy's `descr is other_descr`
/// rather than relying on the pyre-only `descr_index` serialization
/// handle.
#[inline]
fn opt_descr_ptr_eq(a: &Option<crate::DescrRef>, b: &Option<crate::DescrRef>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
        _ => false,
    }
}

// `PartialEq/Eq` parity: compare resume-info structurally + descr Arc
// identity (`history.py:125`); `descr_index` is a serialization handle,
// not identity.
impl PartialEq for RdVirtualInfo {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::VirtualInfo {
                    descr: a_descr,
                    type_id: a0,
                    known_class: a2,
                    fielddescrs: a3,
                    fieldnums: a4,
                    descr_size: a5,
                },
                Self::VirtualInfo {
                    descr: b_descr,
                    type_id: b0,
                    known_class: b2,
                    fielddescrs: b3,
                    fieldnums: b4,
                    descr_size: b5,
                },
            ) => {
                opt_descr_ptr_eq(a_descr, b_descr)
                    && a0 == b0
                    && a2 == b2
                    && a3 == b3
                    && a4 == b4
                    && a5 == b5
            }
            (
                Self::VStructInfo {
                    typedescr: a_descr,
                    type_id: a1,
                    fielddescrs: a3,
                    fieldnums: a4,
                    descr_size: a5,
                },
                Self::VStructInfo {
                    typedescr: b_descr,
                    type_id: b1,
                    fielddescrs: b3,
                    fieldnums: b4,
                    descr_size: b5,
                },
            ) => opt_descr_ptr_eq(a_descr, b_descr) && a1 == b1 && a3 == b3 && a4 == b4 && a5 == b5,
            (
                Self::VArrayInfoClear {
                    arraydescr: a_descr,
                    kind: a2,
                    fieldnums: a3,
                },
                Self::VArrayInfoClear {
                    arraydescr: b_descr,
                    kind: b2,
                    fieldnums: b3,
                },
            ) => opt_descr_ptr_eq(a_descr, b_descr) && a2 == b2 && a3 == b3,
            (
                Self::VArrayInfoNotClear {
                    arraydescr: a_descr,
                    kind: a2,
                    fieldnums: a3,
                },
                Self::VArrayInfoNotClear {
                    arraydescr: b_descr,
                    kind: b2,
                    fieldnums: b3,
                },
            ) => opt_descr_ptr_eq(a_descr, b_descr) && a2 == b2 && a3 == b3,
            (
                Self::VArrayStructInfo {
                    arraydescr: a_descr,
                    size: a2,
                    fielddescrs: a_fielddescrs,
                    fielddescr_indices: a3,
                    field_types: a4,
                    base_size: a4b,
                    item_size: a5,
                    field_offsets: a6,
                    field_sizes: a7,
                    fieldnums: a8,
                },
                Self::VArrayStructInfo {
                    arraydescr: b_descr,
                    size: b2,
                    fielddescrs: b_fielddescrs,
                    fielddescr_indices: b3,
                    field_types: b4,
                    base_size: b4b,
                    item_size: b5,
                    field_offsets: b6,
                    field_sizes: b7,
                    fieldnums: b8,
                },
            ) => {
                opt_descr_ptr_eq(a_descr, b_descr)
                    && a2 == b2
                    && a_fielddescrs.len() == b_fielddescrs.len()
                    && a_fielddescrs
                        .iter()
                        .zip(b_fielddescrs.iter())
                        .all(|(a, b)| std::sync::Arc::ptr_eq(a, b))
                    && a3 == b3
                    && a4 == b4
                    && a4b == b4b
                    && a5 == b5
                    && a6 == b6
                    && a7 == b7
                    && a8 == b8
            }
            (
                Self::VRawBufferInfo {
                    func: a0,
                    size: a1,
                    offsets: a2,
                    descrs: a3,
                    fieldnums: a4,
                },
                Self::VRawBufferInfo {
                    func: b0,
                    size: b1,
                    offsets: b2,
                    descrs: b3,
                    fieldnums: b4,
                },
            ) => a0 == b0 && a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4,
            (
                Self::VRawSliceInfo {
                    offset: a1,
                    fieldnums: a2,
                },
                Self::VRawSliceInfo {
                    offset: b1,
                    fieldnums: b2,
                },
            ) => a1 == b1 && a2 == b2,
            (Self::VStrPlainInfo { fieldnums: a }, Self::VStrPlainInfo { fieldnums: b }) => a == b,
            (Self::VStrConcatInfo { fieldnums: a }, Self::VStrConcatInfo { fieldnums: b }) => {
                a == b
            }
            (Self::VStrSliceInfo { fieldnums: a }, Self::VStrSliceInfo { fieldnums: b }) => a == b,
            (Self::VUniPlainInfo { fieldnums: a }, Self::VUniPlainInfo { fieldnums: b }) => a == b,
            (Self::VUniConcatInfo { fieldnums: a }, Self::VUniConcatInfo { fieldnums: b }) => {
                a == b
            }
            (Self::VUniSliceInfo { fieldnums: a }, Self::VUniSliceInfo { fieldnums: b }) => a == b,
            (Self::Empty, Self::Empty) => true,
            _ => false,
        }
    }
}
impl Eq for RdVirtualInfo {}

impl RdVirtualInfo {
    /// resume.py:584-585 `AbstractVirtualInfo.set_content` stores `fieldnums`
    /// onto every concrete vinfo. This accessor exposes that per-variant
    /// field for `equals` / caching.
    pub fn fieldnums(&self) -> Option<&[i16]> {
        match self {
            Self::VirtualInfo { fieldnums, .. }
            | Self::VStructInfo { fieldnums, .. }
            | Self::VArrayInfoClear { fieldnums, .. }
            | Self::VArrayInfoNotClear { fieldnums, .. }
            | Self::VArrayStructInfo { fieldnums, .. }
            | Self::VRawBufferInfo { fieldnums, .. }
            | Self::VRawSliceInfo { fieldnums, .. }
            | Self::VStrPlainInfo { fieldnums }
            | Self::VStrConcatInfo { fieldnums, .. }
            | Self::VStrSliceInfo { fieldnums, .. }
            | Self::VUniPlainInfo { fieldnums }
            | Self::VUniConcatInfo { fieldnums, .. }
            | Self::VUniSliceInfo { fieldnums, .. } => Some(fieldnums),
            Self::Empty => None,
        }
    }

    /// resume.py:581-582 `AbstractVirtualInfo.equals(fieldnums)`:
    ///
    /// ```python
    /// def equals(self, fieldnums):
    ///     return tagged_list_eq(self.fieldnums, fieldnums)
    /// ```
    ///
    /// Used by `ResumeDataVirtualAdder.make_virtual_info` (resume.py:310)
    /// to decide whether a cached `_cached_vinfo` can be reused verbatim.
    pub fn equals(&self, other_fieldnums: &[i16]) -> bool {
        self.fieldnums().is_some_and(|fns| fns == other_fieldnums)
    }

    /// resume.py:584-585 `AbstractVirtualInfo.set_content(fieldnums)`:
    ///
    /// ```python
    /// def set_content(self, fieldnums):
    ///     self.fieldnums = fieldnums
    /// ```
    ///
    /// Called by `ResumeDataVirtualAdder.make_virtual_info` (resume.py:313)
    /// after `info.visitor_dispatch_virtual_type(self)` produced a fresh
    /// vinfo — the visitor constructs the variant without fieldnums,
    /// and this method stores the caller-supplied `fieldnums` onto it
    /// before caching.
    pub fn set_content(&mut self, new_fieldnums: Vec<i16>) {
        match self {
            Self::VirtualInfo { fieldnums, .. }
            | Self::VStructInfo { fieldnums, .. }
            | Self::VArrayInfoClear { fieldnums, .. }
            | Self::VArrayInfoNotClear { fieldnums, .. }
            | Self::VArrayStructInfo { fieldnums, .. }
            | Self::VRawBufferInfo { fieldnums, .. }
            | Self::VRawSliceInfo { fieldnums, .. }
            | Self::VStrPlainInfo { fieldnums }
            | Self::VStrConcatInfo { fieldnums, .. }
            | Self::VStrSliceInfo { fieldnums, .. }
            | Self::VUniPlainInfo { fieldnums }
            | Self::VUniConcatInfo { fieldnums, .. }
            | Self::VUniSliceInfo { fieldnums, .. } => *fieldnums = new_fieldnums,
            Self::Empty => {}
        }
    }
}

/// resume.py:87-92 PENDINGFIELDSTRUCT parity: a deferred
/// SETFIELD_GC/SETARRAYITEM_GC where the stored value is virtual.
/// Encoded into the guard's resume data and replayed on guard failure
/// after virtual materialization.
///
/// Fields mirror PENDINGFIELDSTRUCT (lldescr / num / fieldnum / itemindex).
/// `target` / `value` are pyre-only (SSA position before resume numbering).
#[derive(Clone, Debug)]
pub struct GuardPendingFieldEntry {
    /// resume.py:88 `lldescr`: the field/array descriptor itself. Carries
    /// `field_offset` / `field_size` / `field_type` via the trait, so the
    /// consumer never needs a precomputed cache.
    pub descr: Option<DescrRef>,
    /// resume.py:91 `itemindex` — for SETARRAYITEM_GC the constant array
    /// index, -1 for SETFIELD_GC.
    pub item_index: i32,
    /// OpRef of the target struct/array (compile-time SSA position,
    /// pyre-only — RPython resolves this via Box identity).
    pub target: OpRef,
    /// OpRef of the value being stored (compile-time SSA position).
    pub value: OpRef,
    /// resume.py:89 `num` — tagged target (TAGBOX/TAGCONST/TAGVIRTUAL).
    /// Set by store_final_boxes_in_guard when resume numbering is available.
    pub target_tagged: i16,
    /// resume.py:90 `fieldnum` — tagged value (TAGBOX/TAGCONST/TAGVIRTUAL).
    pub value_tagged: i16,
}

/// resume.py:419-426 — virtual object field info discovered by
/// `visitor_walk_recursive` inside `finish()`.
#[derive(Debug, Clone)]
pub struct VirtualFieldsInfo {
    /// Type descriptor for the virtual object.
    pub descr: Option<DescrRef>,
    /// Known class pointer (for NewWithVtable) — an immortal vtable address
    /// carried as `ConstInt(ptr2int(typeptr))` (model.py:199-201), never a
    /// traced ref.
    pub known_class: Option<i64>,
    /// Field OpRefs (after get_box_replacement). Order matches the
    /// virtual's field descriptor list.
    pub field_oprefs: Vec<OpRef>,
}

/// resume.py:192-226 parity: box environment for _number_boxes.
///
/// Abstracts the operations RPython performs on boxes during snapshot
/// numbering. Used by ResumeDataLoopMemo.number() to tag each box.
pub trait BoxEnv {
    /// resume.py:202 — box.get_box_replacement()
    fn get_box_replacement(&self, opref: OpRef) -> OpRef;
    /// resoperation.py:58 get_box_replacement(not_const=True) — walk
    /// forwarding chains but stop before stepping into a Const target.
    ///
    /// Used after resume numbering has already classified Const boxes as
    /// TAGCONST, so backend liveboxes keep their runtime Box identity.
    fn get_box_replacement_not_const(&self, opref: OpRef) -> OpRef {
        self.get_box_replacement(opref)
    }
    /// resume.py:202 `box.get_box_replacement()` returning the canonical box
    /// OBJECT as an [`Operand`] (the producer `Op`/`InputArg` host), not just
    /// its OpRef. The resume-numbering maps (#160/S11 `LiveboxMap`,
    /// `cached_boxes`, `cached_virtuals`) key by box identity — RPython's
    /// dict-by-`is` — so two reaches of one box must compare equal. An
    /// `Operand`'s `==` / `Hash` route through its producer Rc, so two reaches
    /// of one logical box are `ptr_eq`. Const is classified by `is_const` /
    /// `getconst` before any map insert and never reaches here.
    fn get_box_replacement_operand(&self, opref: OpRef) -> Operand;
    /// resume.py:204 — isinstance(box, Const)
    fn is_const(&self, opref: OpRef) -> bool;
    /// Constant value + type. Only valid when is_const returns true.
    fn get_const(&self, opref: OpRef) -> (i64, Type);
    /// resume.py:211,214 — box.type
    fn get_type(&self, opref: OpRef) -> Type;
    /// resume.py:212-213 — getptrinfo(box) is not None and info.is_virtual()
    fn is_virtual_ref(&self, opref: OpRef) -> bool;
    /// resume.py:215-216 — getrawptrinfo(box) is not None and info.is_virtual()
    fn is_virtual_raw(&self, opref: OpRef) -> bool;
    /// resume.py:419-426 — getptrinfo(box).visitor_walk_recursive(box, self)
    ///
    /// Returns virtual field info for the given OpRef if it is a virtual
    /// object. Called by `finish()` to discover virtual fields inline,
    /// matching RPython's callback-based `visitor_walk_recursive` pattern.
    /// Default returns None (no virtual info available).
    fn get_virtual_fields(&self, _opref: OpRef) -> Option<VirtualFieldsInfo> {
        None
    }
    /// bridgeopt.py:79-80: getptrinfo(box).get_known_class(cpu) is not None.
    /// Returns true if the optimizer knows the class of the given OpRef.
    fn has_known_class(&self, _opref: OpRef) -> bool {
        false
    }
    /// resume.py:307-315 make_virtual_info(info, fieldnums) parity.
    ///
    /// Creates an `RdVirtualInfo` for a virtual OpRef with given fieldnums.
    /// Dispatches on the virtual type (Virtual, VStruct, VArray, etc.)
    /// to produce the correct variant — matching RPython's
    /// `info.visitor_dispatch_virtual_type(self)` + `vinfo.set_content(fieldnums)`.
    fn make_virtual_info(
        &self,
        _opref: OpRef,
        _fieldnums: Vec<i16>,
    ) -> Option<std::rc::Rc<RdVirtualInfo>> {
        None
    }
    /// resume.py:504-505 `if vinfo.fieldnums is not fieldnums: memo.nvreused += 1`.
    ///
    /// Returns true when `make_virtual_info()` would reuse an already-cached
    /// virtual info object for the given `(opref, fieldnums)` instead of
    /// allocating a fresh one.
    fn virtual_info_would_be_reused(&self, _opref: OpRef, _fieldnums: &[i16]) -> bool {
        false
    }
}

/// Shared-identity handle to an `Op`.
///
/// Mirrors RPython's object-identity model: `resoperation.py:250
/// AbstractResOp` instances are plain Python objects, so every consumer
/// (`history.py:528 TreeLoop.operations`, `optimizer.py:562 trace.next()`,
/// short preamble export, resume metadata, backend input lists) reaches
/// the **same** ResOperation object and reads/writes `_forwarded`
/// through that shared identity.  Pyre's analog: every consumer holds
/// the same `Rc<Op>` and reads/writes `forwarded`/`descr`/...  through
/// the interior-mutable slots.
///
/// This alias is the shared-identity handle for trace `Op` storage.
/// Most sites traffic in `OpRc`; the remaining `Vec<Op>` sites keep the
/// legacy clone-on-copy shape until they are migrated.
pub type OpRc = std::rc::Rc<Op>;

/// A single IR operation.
///
/// Mirrors `rpython/jit/metainterp/resoperation.py:250` `AbstractResOp`.
/// The `_forwarded` slot (`resoperation.py:233-242
/// AbstractResOpOrInputArg._forwarded`) lives directly on this struct in
/// the [`forwarded`](Op::forwarded) field, matching RPython's
/// object-identity model: every consumer holding the same `Rc<Op>` reads
/// and writes the same slot.
#[derive(Debug)]
pub struct Op {
    pub opcode: OpCode,
    /// `resoperation.py:281 AbstractResOp` operand list. `RefCell` so
    /// `setarg` / `initarglist` can mutate through a shared `Op` reached
    /// via `Rc<Op>` — RPython writes
    /// `op._args[i] = ...` on the same Python object the trace list,
    /// optimizer state, and backend input lists all observe.
    ///
    /// `#9` operand-union: each slot is an [`Operand`] — a bound producer
    /// carried by `Rc` (`Op` / `InputArg`) or an inline `Const`. The
    /// operand-keyed accessors (`arg`, `getarglist`, ...) hand out the stored
    /// [`Operand`] directly. Every source binds its producer, so an unbound
    /// position-only operand is never stored — that would be a #9 contract
    /// violation.
    pub args: std::cell::RefCell<SmallVec<[Operand; 3]>>,
    /// `resoperation.py:460 ResOpWithDescr._descr` parity.  `RefCell`
    /// so the optimizer can stamp a descr onto a shared `Op` reached
    /// through `Rc<Op>`: RPython's
    /// `op.setdescr(...)` writes through the same slot every observer
    /// sees.
    pub descr: std::cell::RefCell<Option<DescrRef>>,
    /// Index of this op in the trace (set by the trace builder). `Cell`
    /// so the position can be patched via `&Op` once the op is shared
    /// (the trace-iterator finalizer and unroll's resume-position
    /// retargeting both mutate `pos` after construction).
    pub pos: std::cell::Cell<OpRef>,
    /// resoperation.py:1693 `opclasses[opnum].type` parity (Box.type intrinsic).
    /// Mirrors RPython's `op.type` class attribute set by `optypes[opnum]`
    /// (`resoperation.py:1597`). Populated at construction from
    /// `opcode.result_type()`. Replaces side-table `value_types: HashMap<u32, Type>`.
    pub type_: Type,
    /// For guard ops: values to store in the dead frame on guard failure.
    /// Mirrors rpython/jit/metainterp/resoperation.py getfailargs/setfailargs.
    /// If None, the backend falls back to storing input args.  `RefCell` so
    /// the optimizer can rewrite fail_args on a shared `Op` reached
    /// through `Rc<Op>`: RPython writes
    /// `op._fail_args = [...]` on the same Python object the trace list,
    /// optimizer state, and backend input list all see.
    ///
    /// Stored as [`Operand`] (the same union as `args` — `GuardResOp.
    /// _fail_args` holds the Box objects themselves, resoperation.py:483).
    /// Each write sheds to a bound producer (`Operand::Op` / `InputArg`) or
    /// an inline `Const`; the position-remap passes skip bound operands and
    /// rewrite no longer-existing position-only snapshot, mirroring `args`.
    pub fail_args: std::cell::RefCell<Option<SmallVec<[Operand; 3]>>>,
    /// Types of fail_args, set by the optimizer (`set_fail_arg_types`)
    /// from each fail-arg operand's type.
    /// When present, the backend uses these instead of inferring types.
    /// `RefCell` so the optimizer can stamp types onto a shared `Op`
    /// reached through `Rc<Op>`: RPython
    /// writes `op.fail_arg_types = [...]` on the same Python object the
    /// trace/backend/short preamble all observe.
    pub fail_arg_types: std::cell::RefCell<Option<Vec<Type>>>,
    /// resoperation.py: GuardResOp.rd_resume_position — index of the
    /// guard in the trace for resume data lookup. Set by unroll when
    /// creating extra guards from short preamble / virtual state.
    /// -1 means unset. `Cell` so that mutators reachable via `&Op` (the
    /// shared-trace identity model from `Vec<Rc<Op>>`) can update the
    /// slot without requiring `&mut Op`.
    pub rd_resume_position: std::cell::Cell<i32>,
    /// resoperation.py:111-115 `VecOperationNew.__init__` stores
    /// `datatype` / `bytesize` / `signed` / `count` on the op instance
    /// itself. `resoperation.py:511-518 copy_and_change` propagates them
    /// across rewrites. The slot lives on every `Op` (not only on
    /// vector ops) because pyre collapses the upstream
    /// `VectorOp`/`VectorGuardOp` subclasses into the same struct;
    /// scalar ops keep this `None`. `RefCell` mirrors the in-place
    /// `op.bytesize = ...` overwrite in `VecOperation.__init__`.
    pub vecinfo: std::cell::RefCell<Option<std::boxed::Box<VectorizationInfo>>>,

    /// `resoperation.py:233-242 AbstractResOpOrInputArg._forwarded` parity
    /// slot — the canonical forwarding host for a bound ResOp box.
    /// `Forwarded::None` until a writer sets it; `set_forwarded_*`
    /// on a bound box routes here, and `get_forwarded` reads it back.
    pub forwarded: std::cell::RefCell<crate::forwarding::Forwarded>,

    /// `resoperation.py:566 IntOp._resint` / `:582 FloatOp._resfloat` /
    /// `:612 RefOp._resref` parity (`history.py:803-807 *FrontendOp(pos,
    /// value)`) — the concrete runtime value stamped onto this op
    /// identity at execute-time. The canonical per-identity concrete
    /// carrier for a bound ResOp box; the `get_value`/`set_value`
    /// accessors route here. `None` until a writer stamps it (trace-time
    /// `set_opref_concrete`); residual calls / guards keep it `None`
    /// until blackhole runs them.
    pub value: std::cell::Cell<Option<crate::value::Value>>,
}

impl Clone for Op {
    /// Cloning produces a fresh-identity `Op`. The `_forwarded` slot
    /// (`resoperation.py:233-242 AbstractResOpOrInputArg._forwarded`) is
    /// per-instance mutable state tied to that identity, and RPython
    /// resets it for every newly-constructed `ResOperation`
    /// (`resoperation.py:243-247 __init__`). Preserve identity-shared
    /// forwarding via `Rc::clone` on `OpRc` instead.
    fn clone(&self) -> Self {
        Op {
            opcode: self.opcode,
            args: std::cell::RefCell::new(self.args.borrow().clone()),
            descr: std::cell::RefCell::new(self.descr.borrow().clone()),
            pos: std::cell::Cell::new(self.pos.get()),
            type_: self.type_,
            fail_args: std::cell::RefCell::new(self.fail_args.borrow().clone()),
            fail_arg_types: std::cell::RefCell::new(self.fail_arg_types.borrow().clone()),
            rd_resume_position: std::cell::Cell::new(self.rd_resume_position.get()),
            // resoperation.py:511-518 VectorOp/VectorGuardOp.copy_and_change
            // copies datatype/bytesize/signed/count from the source.
            vecinfo: std::cell::RefCell::new(self.vecinfo.borrow().clone()),
            forwarded: std::cell::RefCell::new(Forwarded::None),
            value: std::cell::Cell::new(None),
        }
    }
}

/// resoperation.py:156-200: Per-op vector metadata for the vectorizer.
/// Tracks how a scalar op maps to SIMD lanes.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorizationInfo {
    /// 'i' for integer, 'f' for float, '\0' for unset
    pub datatype: char,
    /// Byte size per element (-1 = machine word)
    pub bytesize: i8,
    /// Whether the values are signed
    pub signed: bool,
    /// Number of SIMD lanes (-1 = unset)
    pub count: i16,
}

impl VectorizationInfo {
    /// resoperation.py:156-162: default values
    pub fn new() -> Self {
        VectorizationInfo {
            datatype: '\0',
            bytesize: -1,
            signed: true,
            count: -1,
        }
    }

    /// resoperation.py:163-169 `VectorizationInfo(op)` for Const/InputArg
    /// and the default result-type branch for regular ops.
    pub fn from_type(tp: Type) -> Self {
        let mut info = VectorizationInfo::new();
        info.setinfo(type_to_vector_datatype(tp), -1, tp == Type::Int);
        info
    }

    /// resoperation.py:214-230: setinfo — normalize bytesize by datatype.
    pub fn setinfo(&mut self, datatype: char, bytesize: i8, signed: bool) {
        self.datatype = datatype;
        self.bytesize = if bytesize == -1 {
            match datatype {
                'i' => Self::INT_WORD,
                'f' => Self::FLOAT_WORD,
                'r' => Self::INT_WORD,
                'v' => 0,
                'V' => Self::INT_WORD,
                _ => Self::INT_WORD, // safe default
            }
        } else {
            bytesize
        };
        self.signed = signed;
    }

    /// resoperation.py:219-222: getbytesize
    pub fn getbytesize(&self) -> usize {
        if self.bytesize == -1 {
            Self::INT_WORD as usize
        } else {
            self.bytesize as usize
        }
    }

    /// Machine word sizes (64-bit platform).
    const INT_WORD: i8 = 8;
    const FLOAT_WORD: i8 = 8;

    /// resoperation.py:224-227: getcount
    pub fn getcount(&self) -> usize {
        if self.count == -1 {
            1
        } else {
            self.count as usize
        }
    }
}

fn type_to_vector_datatype(tp: Type) -> char {
    match tp {
        Type::Int => 'i',
        Type::Float => 'f',
        Type::Ref => 'r',
        Type::Void => 'v',
    }
}

impl AsRef<Op> for Op {
    fn as_ref(&self) -> &Op {
        self
    }
}

impl Op {
    /// Visit every inline `ConstPtr.value` slot carried by this operation.
    ///
    /// RPython's GC traces `ResOperation.args` / `fail_args` object fields
    /// directly. Pyre stores `ConstPtr.value` in `OpRef::ConstPtr`, so
    /// long-lived op graphs must expose the actual mutable slots to the GC
    /// walker instead of copying the pointer values aside.
    pub fn walk_const_ptr_refs_mut(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        for arg in self.args.borrow().iter() {
            arg.walk_const_ptr_refs(visitor);
        }
        if let Some(fail_args) = self.fail_args.borrow().as_ref() {
            for arg in fail_args.iter() {
                arg.walk_const_ptr_refs(visitor);
            }
        }
    }

    pub fn new(opcode: OpCode, args: &[Operand]) -> Self {
        Op {
            opcode,
            args: std::cell::RefCell::new(args.iter().cloned().collect()),
            descr: std::cell::RefCell::new(None),
            pos: std::cell::Cell::new(OpRef::NONE),
            type_: opcode.result_type(),
            fail_args: std::cell::RefCell::new(None),
            fail_arg_types: std::cell::RefCell::new(None),
            rd_resume_position: std::cell::Cell::new(-1),
            vecinfo: std::cell::RefCell::new(None),
            forwarded: std::cell::RefCell::new(Forwarded::None),
            value: std::cell::Cell::new(None),
        }
    }

    pub fn with_descr(opcode: OpCode, args: &[Operand], descr: DescrRef) -> Self {
        Op {
            opcode,
            args: std::cell::RefCell::new(args.iter().cloned().collect()),
            descr: std::cell::RefCell::new(Some(descr)),
            pos: std::cell::Cell::new(OpRef::NONE),
            type_: opcode.result_type(),
            fail_args: std::cell::RefCell::new(None),
            fail_arg_types: std::cell::RefCell::new(None),
            rd_resume_position: std::cell::Cell::new(-1),
            vecinfo: std::cell::RefCell::new(None),
            forwarded: std::cell::RefCell::new(Forwarded::None),
            value: std::cell::Cell::new(None),
        }
    }

    pub fn arg(&self, idx: usize) -> Operand {
        self.args.borrow()[idx].clone()
    }

    /// True iff argument `idx` is a live-tracking bound operand
    /// (`Operand::Op` / `Operand::InputArg`) that reads its producer's
    /// current `op.pos`, rather than a `Const` / `None` slot. The
    /// position-remap passes skip these — they auto-track a renumbered
    /// producer and need no rewrite.
    pub fn arg_is_bound(&self, idx: usize) -> bool {
        self.args.borrow()[idx].is_bound()
    }

    pub fn num_args(&self) -> usize {
        self.args.borrow().len()
    }

    pub fn result_type(&self) -> Type {
        self.type_
    }

    /// Read the concrete runtime value stamped on this op identity
    /// (`history.py:680 *FrontendOp.getint()` for the `_resint`/
    /// `_resfloat`/`_resref` slot). `None` until a writer stamps it.
    pub fn get_value(&self) -> Option<crate::value::Value> {
        self.value.get()
    }

    /// Stamp the concrete runtime value on this op identity
    /// (`history.py:803 *FrontendOp(pos, value)`).
    pub fn set_value(&self, v: crate::value::Value) {
        self.value.set(Some(v));
    }

    /// resoperation.py:323-334 AbstractResOp.copy_and_change +
    /// resoperation.py:498-503 GuardResOp.copy_and_change parity.
    ///
    /// "shallow copy: the returned operation is meant to be used in place
    /// of self". For guard ops, copies fail_args AND rd_resume_position.
    /// `fail_arg_types` is the only resume-related cache still on `Op`;
    /// `rd_numb / rd_consts / rd_virtuals / rd_pendingfields` live on
    /// the descr (compile.py:855 `_attrs_`) and follow `descr` automatically
    /// when the same DescrRef is reused.
    ///
    /// `args=None` → reuse self.args (matches getarglist_copy()).
    /// `descr=None` → reuse self.descr.
    pub fn copy_and_change(
        &self,
        opcode: OpCode,
        args: Option<&[Operand]>,
        descr: Option<Option<DescrRef>>,
    ) -> Op {
        let new_args: SmallVec<[Operand; 3]> = match args {
            Some(a) => a.iter().cloned().collect(),
            None => self.args.borrow().clone(),
        };
        let new_descr = match descr {
            Some(d) => d,
            None => self.descr.borrow().clone(),
        };
        let newop = Op {
            opcode,
            args: std::cell::RefCell::new(new_args),
            descr: std::cell::RefCell::new(new_descr),
            pos: std::cell::Cell::new(self.pos.get()),
            type_: opcode.result_type(),
            fail_args: std::cell::RefCell::new(None),
            fail_arg_types: std::cell::RefCell::new(None),
            rd_resume_position: std::cell::Cell::new(-1),
            // resoperation.py:511-518 VectorGuardOp.copy_and_change +
            // :534-541 VectorOp.copy_and_change propagate
            // datatype/bytesize/signed/count from self. Scalar ops keep
            // `self.vecinfo` as `None`, so the clone is a no-op for them.
            vecinfo: std::cell::RefCell::new(self.vecinfo.borrow().clone()),
            forwarded: std::cell::RefCell::new(Forwarded::None),
            value: std::cell::Cell::new(None),
        };
        // resoperation.py:498-503 GuardResOp.copy_and_change:
        //   newop.setfailargs(self.getfailargs())
        //   newop.rd_resume_position = self.rd_resume_position
        // The check is on opcode.is_guard() because in RPython this lives
        // on the GuardResOp class hierarchy.  rd_* live on the descr
        // (compile.py:855 `_attrs_`); the descr Arc was already copied
        // above, so newop reads the same payload through descr.fail_descr().
        if opcode.is_guard() || self.opcode.is_guard() {
            *newop.fail_args.borrow_mut() = self.fail_args.borrow().clone();
            *newop.fail_arg_types.borrow_mut() = self.fail_arg_types.borrow().clone();
            newop.rd_resume_position.set(self.rd_resume_position.get());
            return newop;
        }
        newop
    }

    /// True iff the descr slot is populated. Matches
    /// `op.getdescr() is not None`.
    ///
    /// This sits in `resoperation.rs` (rather than the sibling
    /// `op_descr` module hosting the closure-bearing accessors) so the
    /// build-script source analyzer that reads this file can resolve
    /// the bool return type when callers in the same file write
    /// `!op.has_descr()`.
    pub fn has_descr(&self) -> bool {
        self.descr.borrow().is_some()
    }

    // `getdescr` / `setdescr` / `cleardescr` /
    // `project_descr` / `with_*_descr` / `resolved_rd_*` /
    // `getfailargs` / `setfailargs` / `getfailargs_copy` /
    // `get_fail_arg_types` / `set_fail_arg_types` /
    // `has_failargs` / `has_fail_arg_types` live in
    // `crate::op_descr` so the closure-bearing accessors don't have to
    // pass through the build-script source analyzer (which reads
    // `resoperation.rs` for the `RdVirtualInfo` enum and chokes on
    // `impl FnOnce` parameter types).
    /// compile.py: ResumeGuardDescr.store_final_boxes(guard_op, boxes, metainterp_sd)
    ///   guard_op.setfailargs(boxes)
    /// compile.py:874-876 store_final_boxes
    pub fn store_final_boxes(&self, boxes: Vec<Operand>) {
        // optimizer.py:745-749: check no duplicates (debug only).
        // history.py:213/251 — `Const.same_constant` defines Const equality
        // by value (e.g. `ConstInt(7) == ConstInt(7)`), not identity. The
        // duplicate detector uses `Operand::same_box` (ptr identity for
        // bound producers, `same_constant` for inline-Const operands), matching
        // the value-based semantics RPython's `op in seen` check relies on.
        #[cfg(debug_assertions)]
        {
            let mut seen: Vec<&Operand> = Vec::new();
            for b in &boxes {
                if !b.is_none() {
                    debug_assert!(
                        !seen.iter().any(|s| s.same_box(b)),
                        "duplicate box in fail_args: {b:?}"
                    );
                    seen.push(b);
                }
            }
        }
        *self.fail_args.borrow_mut() = Some(boxes.into_iter().collect());
    }
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // history.py:227/268/314 — Const operands carry their value inline.
        // Render those via `const_value()`; bound ResOp / InputArg operands
        // render as `v<pos>` from their producer position; `None` renders
        // as `_`.
        fn write_arg(
            f: &mut std::fmt::Formatter<'_>,
            arg: &crate::operand::Operand,
        ) -> std::fmt::Result {
            match arg.const_value() {
                Some(Value::Int(v)) => write!(f, "{v}"),
                Some(Value::Float(v)) => write!(f, "{v}"),
                Some(Value::Ref(v)) => write!(f, "ptr({:#x})", v.0),
                _ => match arg.to_opref() {
                    OpRef::None => write!(f, "_"),
                    r => write!(f, "v{}", r.raw()),
                },
            }
        }
        if self.opcode.is_guard() {
            write!(f, "{:?}(", self.opcode)?;
            for (i, arg) in self.getarglist().iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write_arg(f, arg)?;
            }
            write!(f, ")")?;
            if let Some(fa) = self.getfailargs() {
                write!(f, " [")?;
                for (i, arg) in fa.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write_arg(f, arg)?;
                }
                write!(f, "]")?;
            }
            Ok(())
        } else if self.result_type() != Type::Void {
            write!(f, "v{} = {:?}(", self.pos.get().raw(), self.opcode)?;
            for (i, arg) in self.getarglist().iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write_arg(f, arg)?;
            }
            write!(f, ")")
        } else {
            write!(f, "{:?}(", self.opcode)?;
            for (i, arg) in self.getarglist().iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write_arg(f, arg)?;
            }
            write!(f, ")")
        }
    }
}

/// Lookup-by-u32 abstraction so `format_trace` can accept any
/// constant-pool shape (`HashMap<u32, V>`, `IndexMap<u32, V>`, …)
/// without prescribing the underlying container.
pub trait ConstLookup<V> {
    fn lookup(&self, key: u32) -> Option<&V>;
}

impl<V> ConstLookup<V> for std::collections::HashMap<u32, V> {
    fn lookup(&self, key: u32) -> Option<&V> {
        self.get(&key)
    }
}

impl<V> ConstLookup<V> for indexmap::IndexMap<u32, V> {
    fn lookup(&self, key: u32) -> Option<&V> {
        self.get(&key)
    }
}

/// Format a trace (list of ops) with optional constants for debugging.
///
/// Generic over the constants value type so both the optimizer-side
/// typed `Value` pool and the backend-side legacy `i64` pool format
/// uniformly through their `Debug` impls.
pub fn format_trace<V: std::fmt::Debug, T: AsRef<Op>, C: ConstLookup<V>>(
    ops: &[T],
    constants: &C,
) -> String {
    use std::fmt::Write;
    // history.py:227/268/314 — inline-Const variants carry their value
    // directly; render via accessors instead of `.raw()` (which panics
    // on inline-Const). Body-namespace OpRefs continue to render via
    // raw position with constants-map lookup.
    fn render_arg<V: std::fmt::Debug, C: ConstLookup<V>>(
        out: &mut String,
        arg: &crate::operand::Operand,
        constants: &C,
    ) {
        use std::fmt::Write;
        match arg.const_value() {
            Some(Value::Int(v)) => write!(out, "{v}").unwrap(),
            Some(Value::Float(v)) => write!(out, "{v}").unwrap(),
            Some(Value::Ref(v)) => write!(out, "ptr({:#x})", v.0).unwrap(),
            _ => match arg.to_opref() {
                OpRef::None => write!(out, "_").unwrap(),
                r => {
                    let pos = r.raw();
                    if let Some(val) = constants.lookup(pos) {
                        write!(out, "{val:?}").unwrap();
                    } else {
                        write!(out, "v{pos}").unwrap();
                    }
                }
            },
        }
    }
    let mut out = String::new();
    for op in ops {
        let op: &Op = op.as_ref();
        // Replace known constants in display
        write!(out, "  ").unwrap();
        if op.opcode.is_guard() {
            write!(out, "{:?}(", op.opcode).unwrap();
        } else if op.type_ != Type::Void {
            write!(out, "v{} = {:?}(", op.pos.get().raw(), op.opcode).unwrap();
        } else {
            write!(out, "{:?}(", op.opcode).unwrap();
        }
        for (i, arg) in op.getarglist().iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            render_arg(&mut out, arg, constants);
        }
        write!(out, ")").unwrap();
        // Render descriptor if present (parity with RPython's logger repr_of_descr)
        if let Some(descr) = op.getdescr() {
            let repr = descr.repr();
            if !repr.is_empty() {
                write!(out, " descr=<{repr}>").unwrap();
            }
        }
        if let Some(fa) = op.getfailargs() {
            write!(out, " [").unwrap();
            for (i, arg) in fa.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                render_arg(&mut out, arg, constants);
            }
            write!(out, "]").unwrap();
        }
        writeln!(out).unwrap();
    }
    out
}

/// All JIT IR opcodes.
///
/// Faithfully mirrors rpython/jit/metainterp/resoperation.py `_oplist`.
/// Operations that produce typed results are expanded with suffixes:
///   _I (int), _R (ref/pointer), _F (float), _N (void/none)
///
/// Boundary markers (e.g., _GUARD_FIRST) are not included as enum variants;
/// instead, classification is done via methods on OpCode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum OpCode {
    // ── Final ──
    Jump = 0,
    Finish,

    Label,

    // ── Guards (foldable) ──
    GuardTrue,
    GuardFalse,
    VecGuardTrue,
    VecGuardFalse,
    GuardValue,
    GuardClass,
    GuardNonnull,
    GuardIsnull,
    GuardNonnullClass,
    GuardGcType,
    GuardIsObject,
    GuardSubclass,
    // ── Guards (non-foldable) ──
    GuardNoException,
    GuardException,
    GuardNoOverflow,
    GuardOverflow,
    GuardNotForced,
    GuardNotForced2,
    GuardNotInvalidated,
    GuardEvalBreaker,
    GuardFutureCondition,
    GuardAlwaysFails,

    // ── Always pure: integer arithmetic ──
    IntAdd,
    IntSub,
    IntMul,
    UintMulHigh,
    IntFloorDiv,
    IntMod,
    IntAnd,
    IntOr,
    IntXor,
    IntRshift,
    IntLshift,
    UintRshift,
    IntSignext,

    // ── Always pure: float arithmetic ──
    FloatAdd,
    FloatSub,
    FloatMul,
    FloatTrueDiv,
    FloatFloorDiv,
    FloatMod,
    FloatNeg,
    FloatAbs,

    // ── Always pure: casts ──
    CastFloatToInt,
    CastIntToFloat,
    CastFloatToSinglefloat,
    CastSinglefloatToFloat,
    ConvertFloatBytesToLonglong,
    ConvertLonglongBytesToFloat,

    // ── Always pure: vector arithmetic ──
    VecIntAdd,
    VecIntSub,
    VecIntMul,
    VecIntAnd,
    VecIntOr,
    VecIntXor,
    VecFloatAdd,
    VecFloatSub,
    VecFloatMul,
    VecFloatTrueDiv,
    VecFloatNeg,
    VecFloatAbs,

    // ── Always pure: vector comparisons / casts ──
    VecFloatEq,
    VecFloatNe,
    VecFloatXor,
    VecIntIsTrue,
    VecIntNe,
    VecIntEq,
    VecIntSignext,
    VecCastFloatToSinglefloat,
    VecCastSinglefloatToFloat,
    VecCastFloatToInt,
    VecCastIntToFloat,

    // ── Always pure: vector pack/unpack ──
    VecI,
    VecF,
    VecUnpackI,
    VecUnpackF,
    VecPackI,
    VecPackF,
    VecExpandI,
    VecExpandF,

    // ── Always pure: integer comparisons ──
    IntLt,
    IntLe,
    IntEq,
    IntNe,
    IntGt,
    IntGe,
    UintLt,
    UintLe,
    UintGt,
    UintGe,

    // ── Always pure: float comparisons ──
    FloatLt,
    FloatLe,
    FloatEq,
    FloatNe,
    FloatGt,
    FloatGe,

    // ── Always pure: unary int ──
    IntIsZero,
    IntIsTrue,
    IntNeg,
    IntInvert,
    IntForceGeZero,
    IntBetween,

    // ── Always pure: identity / cast ──
    SameAsI,
    SameAsR,
    SameAsF,
    CastPtrToInt,
    CastIntToPtr,
    CastOpaquePtr,

    // ── Always pure: pointer comparisons ──
    PtrEq,
    PtrNe,
    InstancePtrEq,
    InstancePtrNe,
    NurseryPtrIncrement,

    // ── Always pure: array/string length, getitem ──
    ArraylenGc,
    Strlen,
    Strgetitem,
    GetarrayitemGcPureI,
    GetarrayitemGcPureR,
    GetarrayitemGcPureF,
    Unicodelen,
    Unicodegetitem,

    // ── Always pure: backend-specific loads ──
    LoadFromGcTable,
    LoadEffectiveAddress,

    // ── Thread-local reference ──
    ThreadlocalrefGet,

    // ── No side effect (but not always pure) ──
    GcLoadI,
    GcLoadR,
    GcLoadF,
    GcLoadIndexedI,
    GcLoadIndexedR,
    GcLoadIndexedF,

    // ── Raw loads ──
    GetarrayitemGcI,
    GetarrayitemGcR,
    GetarrayitemGcF,
    GetarrayitemRawI,
    GetarrayitemRawR,
    GetarrayitemRawF,
    RawLoadI,
    RawLoadF,
    VecLoadI,
    VecLoadF,

    // ── No side effect: field/interior access ──
    GetinteriorfieldGcI,
    GetinteriorfieldGcR,
    GetinteriorfieldGcF,
    GetfieldGcI,
    GetfieldGcR,
    GetfieldGcF,
    GetfieldRawI,
    GetfieldRawR,
    GetfieldRawF,

    // ── No side effect: pure field access (immutable) ──
    GetfieldGcPureI,
    GetfieldGcPureR,
    GetfieldGcPureF,

    // ── Allocation ──
    New,
    NewWithVtable,
    NewArray,
    NewArrayClear,
    Newstr,
    Newunicode,

    // ── No side effect: misc ──
    ForceToken,
    VirtualRefI,
    VirtualRefR,
    Strhash,
    Unicodehash,

    // ── Side effects: GC stores ──
    GcStore,
    GcStoreIndexed,

    // ── Side effects: misc ──
    IncrementDebugCounter,

    // ── Raw stores ──
    SetarrayitemGc,
    SetarrayitemRaw,
    RawStore,
    VecStore,

    // ── Side effects: field/interior stores ──
    SetinteriorfieldGc,
    SetinteriorfieldRaw,
    SetfieldGc,
    ZeroArray,
    SetfieldRaw,
    Strsetitem,
    Unicodesetitem,

    // ── GC write barriers ──
    CondCallGcWb,
    CondCallGcWbArray,

    // ── Debug ──
    DebugMergePoint,
    EnterPortalFrame,
    LeavePortalFrame,
    JitDebug,

    // ── Testing only ──
    ForceSpill,

    // ── Misc side effects ──
    VirtualRefFinish,
    Copystrcontent,
    Copyunicodecontent,
    QuasiimmutField,
    AssertNotNone,
    RecordExactClass,
    RecordExactValueR,
    RecordExactValueI,
    Keepalive,
    SaveException,
    SaveExcClass,
    RestoreException,

    // ── Calls (can raise) ──
    CallI,
    CallR,
    CallF,
    CallN,
    CondCallN,
    CondCallValueI,
    CondCallValueR,
    CallAssemblerI,
    CallAssemblerR,
    CallAssemblerF,
    CallAssemblerN,
    CallMayForceI,
    CallMayForceR,
    CallMayForceF,
    CallMayForceN,
    CallLoopinvariantI,
    CallLoopinvariantR,
    CallLoopinvariantF,
    CallLoopinvariantN,
    CallReleaseGilI,
    // CallReleaseGilR intentionally absent: resoperation.py:1243-1244
    // (`# no such thing`) excludes CALL_RELEASE_GIL_R from the upstream
    // opcode table.
    CallReleaseGilF,
    CallReleaseGilN,
    CallPureI,
    CallPureR,
    CallPureF,
    CallPureN,
    CheckMemoryError,
    CallMallocNursery,
    CallMallocNurseryVarsize,
    CallMallocNurseryVarsizeFrame,
    RecordKnownResult,

    // ── Overflow ──
    IntAddOvf,
    IntSubOvf,
    IntMulOvf,
}

// ── Boundary constants for category classification ──
// These correspond to the _FIRST/_LAST markers in resoperation.py.

const FINAL_FIRST: u16 = OpCode::Jump as u16;
const FINAL_LAST: u16 = OpCode::Finish as u16;

const GUARD_FIRST: u16 = OpCode::GuardTrue as u16;
const GUARD_FOLDABLE_FIRST: u16 = OpCode::GuardTrue as u16;
const GUARD_FOLDABLE_LAST: u16 = OpCode::GuardSubclass as u16;
const GUARD_LAST: u16 = OpCode::GuardAlwaysFails as u16;

const ALWAYS_PURE_FIRST: u16 = OpCode::IntAdd as u16;
const ALWAYS_PURE_LAST: u16 = OpCode::LoadEffectiveAddress as u16;

const NOSIDEEFFECT_FIRST: u16 = OpCode::IntAdd as u16; // same as ALWAYS_PURE_FIRST
const NOSIDEEFFECT_LAST: u16 = OpCode::Unicodehash as u16;

const MALLOC_FIRST: u16 = OpCode::New as u16;
const MALLOC_LAST: u16 = OpCode::Newunicode as u16;

const RAW_LOAD_FIRST: u16 = OpCode::GetarrayitemGcI as u16;
const RAW_LOAD_LAST: u16 = OpCode::VecLoadF as u16;

const RAW_STORE_FIRST: u16 = OpCode::SetarrayitemGc as u16;
const RAW_STORE_LAST: u16 = OpCode::VecStore as u16;

const JIT_DEBUG_FIRST: u16 = OpCode::DebugMergePoint as u16;
const JIT_DEBUG_LAST: u16 = OpCode::JitDebug as u16;

const CALL_FIRST: u16 = OpCode::CallI as u16;
const CALL_LAST: u16 = OpCode::RecordKnownResult as u16;

const CANRAISE_FIRST: u16 = OpCode::CallI as u16;
const CANRAISE_LAST: u16 = OpCode::IntMulOvf as u16;

const OVF_FIRST: u16 = OpCode::IntAddOvf as u16;
const OVF_LAST: u16 = OpCode::IntMulOvf as u16;

impl OpCode {
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    /// Iterate over all defined OpCode variants (0..OPCODE_COUNT).
    pub fn all() -> impl Iterator<Item = OpCode> {
        (0..OPCODE_COUNT as u16).map(|i| unsafe { std::mem::transmute::<u16, OpCode>(i) })
    }

    /// Safe reverse of `as_u16` — bounds-checked conversion used by the
    /// byte-stream `ByteTraceIter` (opencoder.py:362-406 `next()` reads the
    /// opnum byte and looks it up in the `OP_*` registry; in RPython this
    /// is the `opnum` → class-table mapping). Returns `None` when `n`
    /// lies outside the defined `0..OPCODE_COUNT` range.
    pub fn from_u16(n: u16) -> Option<OpCode> {
        if (n as usize) < OPCODE_COUNT {
            // SAFETY: `OpCode` is `#[repr(u16)]` with contiguous variants
            // in `0..OPCODE_COUNT` — any value in that range is a
            // well-defined enum bit pattern (the same rationale used by
            // `OpCode::all` above).
            Some(unsafe { std::mem::transmute::<u16, OpCode>(n) })
        } else {
            None
        }
    }

    // ── Category classification (mirrors rop.is_* static methods) ──

    pub fn is_final(self) -> bool {
        let n = self.as_u16();
        FINAL_FIRST <= n && n <= FINAL_LAST
    }

    pub fn is_guard(self) -> bool {
        let n = self.as_u16();
        GUARD_FIRST <= n && n <= GUARD_LAST
    }

    pub fn is_foldable_guard(self) -> bool {
        let n = self.as_u16();
        GUARD_FOLDABLE_FIRST <= n && n <= GUARD_FOLDABLE_LAST
    }

    pub fn is_always_pure(self) -> bool {
        let n = self.as_u16();
        (ALWAYS_PURE_FIRST <= n && n <= ALWAYS_PURE_LAST)
            || matches!(
                self,
                OpCode::GetfieldGcPureI | OpCode::GetfieldGcPureR | OpCode::GetfieldGcPureF
            )
    }

    pub fn has_no_side_effect(self) -> bool {
        let n = self.as_u16();
        (NOSIDEEFFECT_FIRST <= n && n <= NOSIDEEFFECT_LAST)
            || matches!(
                self,
                OpCode::GetfieldGcPureI | OpCode::GetfieldGcPureR | OpCode::GetfieldGcPureF
            )
    }

    pub fn is_malloc(self) -> bool {
        let n = self.as_u16();
        MALLOC_FIRST <= n && n <= MALLOC_LAST
    }

    pub fn is_call(self) -> bool {
        let n = self.as_u16();
        CALL_FIRST <= n && n <= CALL_LAST
    }

    /// resoperation.py:1434 `OpHelpers.is_real_call`.
    pub fn is_real_call(self) -> bool {
        matches!(
            self,
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
        )
    }

    pub fn can_raise(self) -> bool {
        let n = self.as_u16();
        CANRAISE_FIRST <= n && n <= CANRAISE_LAST
    }

    pub fn can_malloc(self) -> bool {
        self.is_call() || self.is_malloc()
    }

    pub fn is_ovf(self) -> bool {
        let n = self.as_u16();
        OVF_FIRST <= n && n <= OVF_LAST
    }

    pub fn is_raw_load(self) -> bool {
        let n = self.as_u16();
        RAW_LOAD_FIRST < n && n < RAW_LOAD_LAST
    }

    pub fn is_raw_store(self) -> bool {
        let n = self.as_u16();
        RAW_STORE_FIRST < n && n < RAW_STORE_LAST
    }

    /// resoperation.py:1486-1491 `is_primitive_load` / `is_primitive_store`.
    /// Same opcode range as `is_raw_load` / `is_raw_store` (the upstream
    /// `_RAW_LOAD_FIRST` / `_RAW_LOAD_LAST` bracket — `is_primitive_*` and
    /// `is_raw_*` are the same predicate spelled twice).
    pub fn is_primitive_load(self) -> bool {
        self.is_raw_load()
    }

    pub fn is_primitive_store(self) -> bool {
        self.is_raw_store()
    }

    /// resoperation.py:407-417 `AbstractResOp.is_primitive_array_access`
    /// — opcode side of the check. The descr side
    /// (`descr.is_array_of_primitives()`) must still be tested by the
    /// caller because `Op.descr` lives outside `OpCode`.
    pub fn is_primitive_array_access_opcode(self) -> bool {
        self.is_primitive_load() || self.is_primitive_store()
    }

    /// resoperation.py:644-696 `CastOp`/`SignExtOp` mixin attachment
    /// (`resoperation.py:1682-1685`). Returns true exactly for the opcodes
    /// in `_cast_ops` (`resoperation.py:1177-1188`).
    pub fn is_typecast(self) -> bool {
        matches!(
            self,
            OpCode::CastFloatToInt
                | OpCode::CastIntToFloat
                | OpCode::CastFloatToSinglefloat
                | OpCode::CastSinglefloatToFloat
                | OpCode::IntSignext
                | OpCode::VecCastFloatToInt
                | OpCode::VecCastIntToFloat
                | OpCode::VecCastFloatToSinglefloat
                | OpCode::VecCastSinglefloatToFloat
                | OpCode::VecIntSignext,
        )
    }

    /// resoperation.py:434-435 `CastOp.cast_types`. Returns
    /// `(cls_casts[0], cls_casts[2])` — the (from_type, to_type) pair from
    /// `_cast_ops` (`resoperation.py:1177-1188`). Defaults to `('\0','\0')`
    /// (resoperation.py:264) for non-typecast opcodes.
    pub fn cast_types(self) -> (char, char) {
        match self {
            OpCode::CastFloatToInt | OpCode::VecCastFloatToInt => ('f', 'i'),
            OpCode::CastIntToFloat | OpCode::VecCastIntToFloat => ('i', 'f'),
            OpCode::CastFloatToSinglefloat | OpCode::VecCastFloatToSinglefloat => ('f', 'i'),
            OpCode::CastSinglefloatToFloat | OpCode::VecCastSinglefloatToFloat => ('i', 'f'),
            OpCode::IntSignext | OpCode::VecIntSignext => ('i', 'i'),
            _ => ('\0', '\0'),
        }
    }

    /// resoperation.py:437-438 `CastOp.cast_to_bytesize` — returns
    /// `cls_casts[3]`.  The base table at `resoperation.py:1177-1188`
    /// stores 4 for the float↔int casts; the non-x86 override at
    /// `resoperation.py:1190-1196` upgrades `CAST_FLOAT_TO_INT` /
    /// `VEC_CAST_FLOAT_TO_INT` (and the corresponding `cast_from`
    /// bytesize of `CAST_INT_TO_FLOAT` / `VEC_CAST_INT_TO_FLOAT`) to 8
    /// on architectures whose `platform.machine()` does not start with
    /// `x86`.  Mirror that here with a `cfg(target_arch)` switch:
    /// AArch64 / non-x86 builds return 8 for the float→int direction.
    /// `None` is returned for `INT_SIGNEXT` / `VEC_INT_SIGNEXT` where
    /// the `_cast_ops` entry stores 0 and the actual bytesize is the
    /// dynamic value of `arg1` (`SignExtOp.cast_to_bytesize` at
    /// `resoperation.py:682-686` reads `arg1.value`).  Callers must
    /// consult the const-pool to recover the bytesize for these two
    /// opcodes.
    pub fn cast_to_bytesize_static(self) -> Option<i32> {
        // resoperation.py:1190 `if not platform.machine().startswith('x86')`.
        // pyre is built per target arch; gate at compile time.
        const FLOAT_TO_INT_BYTESIZE: i32 = if cfg!(any(target_arch = "x86", target_arch = "x86_64"))
        {
            4
        } else {
            8
        };
        match self {
            OpCode::CastFloatToInt | OpCode::VecCastFloatToInt => Some(FLOAT_TO_INT_BYTESIZE),
            OpCode::CastIntToFloat | OpCode::VecCastIntToFloat => Some(8),
            OpCode::CastFloatToSinglefloat | OpCode::VecCastFloatToSinglefloat => Some(4),
            OpCode::CastSinglefloatToFloat | OpCode::VecCastSinglefloatToFloat => Some(8),
            OpCode::IntSignext | OpCode::VecIntSignext => None,
            _ => None,
        }
    }

    pub fn is_jit_debug(self) -> bool {
        let n = self.as_u16();
        JIT_DEBUG_FIRST <= n && n <= JIT_DEBUG_LAST
    }

    pub fn is_comparison(self) -> bool {
        self.is_always_pure() && self.returns_bool()
    }

    pub fn is_guard_exception(self) -> bool {
        matches!(self, OpCode::GuardException | OpCode::GuardNoException)
    }

    pub fn is_guard_overflow(self) -> bool {
        matches!(self, OpCode::GuardOverflow | OpCode::GuardNoOverflow)
    }

    pub fn is_same_as(self) -> bool {
        matches!(self, OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF)
    }

    pub fn is_getfield(self) -> bool {
        matches!(
            self,
            OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldGcPureI
                | OpCode::GetfieldGcPureR
                | OpCode::GetfieldGcPureF
        )
    }

    pub fn is_getarrayitem(self) -> bool {
        matches!(
            self,
            OpCode::GetarrayitemGcI
                | OpCode::GetarrayitemGcR
                | OpCode::GetarrayitemGcF
                | OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF
        )
    }

    pub fn is_setarrayitem(self) -> bool {
        matches!(self, OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw)
    }

    pub fn is_setfield(self) -> bool {
        matches!(self, OpCode::SetfieldGc | OpCode::SetfieldRaw)
    }

    pub fn is_getinteriorfield(self) -> bool {
        matches!(
            self,
            OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcR | OpCode::GetinteriorfieldGcF
        )
    }

    pub fn is_setinteriorfield(self) -> bool {
        matches!(self, OpCode::SetinteriorfieldGc)
    }

    pub fn is_plain_call(self) -> bool {
        matches!(
            self,
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
        )
    }

    pub fn is_call_assembler(self) -> bool {
        matches!(
            self,
            OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN
        )
    }

    pub fn is_call_may_force(self) -> bool {
        matches!(
            self,
            OpCode::CallMayForceI
                | OpCode::CallMayForceR
                | OpCode::CallMayForceF
                | OpCode::CallMayForceN
        )
    }

    pub fn is_call_pure(self) -> bool {
        matches!(
            self,
            OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN
        )
    }

    pub fn is_call_release_gil(self) -> bool {
        // resoperation.py:1238-1248 call_release_gil_for_descr maps
        // 'i'/'f'/'v' only; 'r' is `# no such thing`.
        matches!(
            self,
            OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN
        )
    }

    pub fn is_call_loopinvariant(self) -> bool {
        matches!(
            self,
            OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN
        )
    }

    pub fn is_cond_call_value(self) -> bool {
        matches!(self, OpCode::CondCallValueI | OpCode::CondCallValueR)
    }

    pub fn is_label(self) -> bool {
        matches!(self, OpCode::Label)
    }

    pub fn is_vector_arithmetic(self) -> bool {
        matches!(
            self,
            OpCode::VecIntAdd
                | OpCode::VecIntSub
                | OpCode::VecIntMul
                | OpCode::VecIntAnd
                | OpCode::VecIntOr
                | OpCode::VecIntXor
                | OpCode::VecFloatAdd
                | OpCode::VecFloatSub
                | OpCode::VecFloatMul
                | OpCode::VecFloatTrueDiv
                | OpCode::VecFloatNeg
                | OpCode::VecFloatAbs
        )
    }

    /// Expected number of arguments, or None for variadic.
    pub fn arity(self) -> Option<u8> {
        OPARITY[self.as_u16() as usize]
    }

    /// Whether this operation takes a descriptor.
    pub fn has_descr(self) -> bool {
        OPWITHDESCR[self.as_u16() as usize]
    }

    /// Whether this operation produces a boolean result.
    pub fn returns_bool(self) -> bool {
        OPBOOL[self.as_u16() as usize]
    }

    /// Result type of this operation.
    pub fn result_type(self) -> Type {
        OPRESTYPE[self.as_u16() as usize]
    }

    /// Name of this operation (for debugging).
    pub fn name(self) -> &'static str {
        OPNAME[self.as_u16() as usize]
    }
}

// ── Typed dispatch helpers (mirrors rop.*_for_descr) ──

impl OpCode {
    pub fn call_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallI,
            Type::Ref => OpCode::CallR,
            Type::Float => OpCode::CallF,
            Type::Void => OpCode::CallN,
        }
    }

    pub fn call_pure_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallPureI,
            Type::Ref => OpCode::CallPureR,
            Type::Float => OpCode::CallPureF,
            Type::Void => OpCode::CallPureN,
        }
    }

    pub fn call_may_force_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallMayForceI,
            Type::Ref => OpCode::CallMayForceR,
            Type::Float => OpCode::CallMayForceF,
            Type::Void => OpCode::CallMayForceN,
        }
    }

    pub fn call_assembler_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallAssemblerI,
            Type::Ref => OpCode::CallAssemblerR,
            Type::Float => OpCode::CallAssemblerF,
            Type::Void => OpCode::CallAssemblerN,
        }
    }

    pub fn call_loopinvariant_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallLoopinvariantI,
            Type::Ref => OpCode::CallLoopinvariantR,
            Type::Float => OpCode::CallLoopinvariantF,
            Type::Void => OpCode::CallLoopinvariantN,
        }
    }

    /// Mirrors `resoperation.py:1238-1248 call_release_gil_for_descr`:
    /// the `'r'` arm is explicitly commented out as `# no such thing`,
    /// so a `Type::Ref` result-typed release-gil callee has no upstream
    /// opcode mapping.  Panic rather than returning `CallReleaseGilR`,
    /// which has no producer in upstream and would record an IR op the
    /// optimizer/backend cannot consume.
    pub fn call_release_gil_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallReleaseGilI,
            Type::Ref => panic!(
                "call_release_gil_for_type: Type::Ref has no upstream counterpart \
                 (resoperation.py:1243-1244 `# no such thing`); CALL_RELEASE_GIL_R \
                 has no producer in RPython"
            ),
            Type::Float => OpCode::CallReleaseGilF,
            Type::Void => OpCode::CallReleaseGilN,
        }
    }

    pub fn same_as_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::SameAsI,
            Type::Ref => OpCode::SameAsR,
            Type::Float => OpCode::SameAsF,
            Type::Void => unreachable!("same_as has no void variant"),
        }
    }

    pub fn getfield_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Ref => OpCode::GetfieldGcR,
            Type::Float => OpCode::GetfieldGcF,
            _ => OpCode::GetfieldGcI,
        }
    }

    pub fn getarrayitem_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Ref => OpCode::GetarrayitemGcR,
            Type::Float => OpCode::GetarrayitemGcF,
            _ => OpCode::GetarrayitemGcI,
        }
    }

    pub fn getfield_raw_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Ref => OpCode::GetfieldRawR,
            Type::Float => OpCode::GetfieldRawF,
            _ => OpCode::GetfieldRawI,
        }
    }

    pub fn getarrayitem_raw_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Ref => OpCode::GetarrayitemRawR,
            Type::Float => OpCode::GetarrayitemRawF,
            _ => OpCode::GetarrayitemRawI,
        }
    }
}

// ── Boolean inverse/reflex tables (from resoperation.py) ──

impl OpCode {
    /// Returns the boolean inverse of a comparison, e.g. INT_EQ -> INT_NE.
    pub fn bool_inverse(self) -> Option<OpCode> {
        match self {
            OpCode::IntEq => Some(OpCode::IntNe),
            OpCode::IntNe => Some(OpCode::IntEq),
            OpCode::IntLt => Some(OpCode::IntGe),
            OpCode::IntGe => Some(OpCode::IntLt),
            OpCode::IntGt => Some(OpCode::IntLe),
            OpCode::IntLe => Some(OpCode::IntGt),
            OpCode::UintLt => Some(OpCode::UintGe),
            OpCode::UintGe => Some(OpCode::UintLt),
            OpCode::UintGt => Some(OpCode::UintLe),
            OpCode::UintLe => Some(OpCode::UintGt),
            OpCode::FloatEq => Some(OpCode::FloatNe),
            OpCode::FloatNe => Some(OpCode::FloatEq),
            OpCode::FloatLt => Some(OpCode::FloatGe),
            OpCode::FloatGe => Some(OpCode::FloatLt),
            OpCode::FloatGt => Some(OpCode::FloatLe),
            OpCode::FloatLe => Some(OpCode::FloatGt),
            OpCode::PtrEq => Some(OpCode::PtrNe),
            OpCode::PtrNe => Some(OpCode::PtrEq),
            _ => None,
        }
    }

    /// Returns the reflexive form of a comparison (swap operands),
    /// e.g. INT_LT -> INT_GT.
    pub fn bool_reflex(self) -> Option<OpCode> {
        match self {
            OpCode::IntEq => Some(OpCode::IntEq),
            OpCode::IntNe => Some(OpCode::IntNe),
            OpCode::IntLt => Some(OpCode::IntGt),
            OpCode::IntGe => Some(OpCode::IntLe),
            OpCode::IntGt => Some(OpCode::IntLt),
            OpCode::IntLe => Some(OpCode::IntGe),
            OpCode::UintLt => Some(OpCode::UintGt),
            OpCode::UintGe => Some(OpCode::UintLe),
            OpCode::UintGt => Some(OpCode::UintLt),
            OpCode::UintLe => Some(OpCode::UintGe),
            OpCode::FloatEq => Some(OpCode::FloatEq),
            OpCode::FloatNe => Some(OpCode::FloatNe),
            OpCode::FloatLt => Some(OpCode::FloatGt),
            OpCode::FloatGe => Some(OpCode::FloatLe),
            OpCode::FloatGt => Some(OpCode::FloatLt),
            OpCode::FloatLe => Some(OpCode::FloatGe),
            OpCode::PtrEq => Some(OpCode::PtrEq),
            OpCode::PtrNe => Some(OpCode::PtrNe),
            _ => None,
        }
    }

    /// Maps a scalar op to its vector equivalent, e.g. INT_ADD -> VEC_INT_ADD.
    pub fn to_vector(self) -> Option<OpCode> {
        match self {
            // resoperation.py:1746-1759 `_opvector`: memory loads/stores map to
            // VEC_LOAD/VEC_STORE. The `_R` (ref) array loads have no vector
            // form upstream and are intentionally omitted. There is no `_PURE`
            // vector op; the pure getarrayitem loads still map to VEC_LOAD_I/F.
            OpCode::RawLoadI => Some(OpCode::VecLoadI),
            OpCode::RawLoadF => Some(OpCode::VecLoadF),
            OpCode::GetarrayitemRawI => Some(OpCode::VecLoadI),
            OpCode::GetarrayitemRawF => Some(OpCode::VecLoadF),
            OpCode::GetarrayitemGcI => Some(OpCode::VecLoadI),
            OpCode::GetarrayitemGcF => Some(OpCode::VecLoadF),
            OpCode::GetarrayitemGcPureI => Some(OpCode::VecLoadI),
            OpCode::GetarrayitemGcPureF => Some(OpCode::VecLoadF),
            OpCode::RawStore => Some(OpCode::VecStore),
            OpCode::SetarrayitemRaw => Some(OpCode::VecStore),
            OpCode::SetarrayitemGc => Some(OpCode::VecStore),
            OpCode::IntAdd => Some(OpCode::VecIntAdd),
            OpCode::IntSub => Some(OpCode::VecIntSub),
            OpCode::IntMul => Some(OpCode::VecIntMul),
            OpCode::IntAnd => Some(OpCode::VecIntAnd),
            OpCode::IntOr => Some(OpCode::VecIntOr),
            OpCode::IntXor => Some(OpCode::VecIntXor),
            OpCode::FloatAdd => Some(OpCode::VecFloatAdd),
            OpCode::FloatSub => Some(OpCode::VecFloatSub),
            OpCode::FloatMul => Some(OpCode::VecFloatMul),
            OpCode::FloatTrueDiv => Some(OpCode::VecFloatTrueDiv),
            OpCode::FloatAbs => Some(OpCode::VecFloatAbs),
            OpCode::FloatNeg => Some(OpCode::VecFloatNeg),
            OpCode::FloatEq => Some(OpCode::VecFloatEq),
            OpCode::FloatNe => Some(OpCode::VecFloatNe),
            OpCode::IntIsTrue => Some(OpCode::VecIntIsTrue),
            OpCode::IntEq => Some(OpCode::VecIntEq),
            OpCode::IntNe => Some(OpCode::VecIntNe),
            OpCode::IntSignext => Some(OpCode::VecIntSignext),
            OpCode::CastFloatToSinglefloat => Some(OpCode::VecCastFloatToSinglefloat),
            OpCode::CastSinglefloatToFloat => Some(OpCode::VecCastSinglefloatToFloat),
            OpCode::CastIntToFloat => Some(OpCode::VecCastIntToFloat),
            OpCode::CastFloatToInt => Some(OpCode::VecCastFloatToInt),
            OpCode::GuardTrue => Some(OpCode::VecGuardTrue),
            OpCode::GuardFalse => Some(OpCode::VecGuardFalse),
            _ => None,
        }
    }

    /// The non-overflow version of an overflow op, e.g. INT_ADD_OVF -> INT_ADD.
    pub fn without_overflow(self) -> Option<OpCode> {
        match self {
            OpCode::IntAddOvf => Some(OpCode::IntAdd),
            OpCode::IntSubOvf => Some(OpCode::IntSub),
            OpCode::IntMulOvf => Some(OpCode::IntMul),
            _ => None,
        }
    }

    /// Whether this opcode accesses memory (load/store).
    pub fn is_memory_access(self) -> bool {
        matches!(
            self,
            // Typed getfield
            OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldRawI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldRawF
                | OpCode::GetfieldGcPureI
                | OpCode::GetfieldGcPureR
                | OpCode::GetfieldGcPureF
                // Untyped setfield
                | OpCode::SetfieldGc
                | OpCode::SetfieldRaw
                // Typed getarrayitem
                | OpCode::GetarrayitemGcI
                | OpCode::GetarrayitemGcR
                | OpCode::GetarrayitemGcF
                | OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF
                | OpCode::GetarrayitemRawI
                | OpCode::GetarrayitemRawR
                | OpCode::GetarrayitemRawF
                // Untyped setarrayitem
                | OpCode::SetarrayitemGc
                | OpCode::SetarrayitemRaw
                // Raw load/store
                | OpCode::RawLoadI
                | OpCode::RawLoadF
                | OpCode::RawStore
                // GC load (typed)
                | OpCode::GcLoadI
                | OpCode::GcLoadR
                | OpCode::GcLoadF
                | OpCode::GcLoadIndexedI
                | OpCode::GcLoadIndexedR
                | OpCode::GcLoadIndexedF
                // GC store (untyped)
                | OpCode::GcStore
                | OpCode::GcStoreIndexed
        )
    }

    /// dependency.py:207-208: loads_from_complex_object
    /// (ALWAYS_PURE_LAST <= opnum < MALLOC_FIRST in RPython)
    pub fn is_complex_load(self) -> bool {
        matches!(
            self,
            OpCode::GetarrayitemGcI
                | OpCode::GetarrayitemGcR
                | OpCode::GetarrayitemGcF
                | OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF
                | OpCode::GetarrayitemRawI
                | OpCode::GetarrayitemRawF
                | OpCode::RawLoadI
                | OpCode::RawLoadF
                | OpCode::VecLoadI
                | OpCode::VecLoadF
                | OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldRawI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldRawF
                | OpCode::GetinteriorfieldGcI
                | OpCode::GetinteriorfieldGcF
                | OpCode::GetinteriorfieldGcR
        )
    }

    /// dependency.py:210-211: modifies_complex_object
    /// (SETARRAYITEM_GC <= opnum <= UNICODESETITEM)
    pub fn is_complex_modify(self) -> bool {
        matches!(
            self,
            OpCode::SetarrayitemGc
                | OpCode::SetarrayitemRaw
                | OpCode::RawStore
                | OpCode::VecStore
                | OpCode::SetinteriorfieldGc
                | OpCode::SetinteriorfieldRaw
                | OpCode::SetfieldGc
                | OpCode::SetfieldRaw
                | OpCode::ZeroArray
                | OpCode::Strsetitem
                | OpCode::Unicodesetitem
        )
    }
}

// ── Metadata tables ──
// These are generated to match the setup() function in resoperation.py.
// Format: arity (None = variadic), has_descr, returns_bool, result_type, name.

macro_rules! opcode_count {
    () => {
        OpCode::IntMulOvf as usize + 1
    };
}

/// Number of defined opcodes.
pub const OPCODE_COUNT: usize = opcode_count!();

// We use include! or manual arrays. For now, manual tables.
// These tables are indexed by OpCode as u16.

/// Arity: Some(n) for fixed arity, None for variadic.
static OPARITY: [Option<u8>; OPCODE_COUNT] = {
    let mut t = [None; OPCODE_COUNT];
    use OpCode::*;
    // Variadic ops (arity = *)
    // Jump, Finish, Label, DebugMergePoint, JitDebug, Escape*, all Calls, CondCall*, RecordKnownResult
    // are variadic -> None (already default)

    // Fixed arity ops
    macro_rules! set {
        ($op:ident, $a:expr) => {
            t[$op as usize] = Some($a);
        };
    }
    // Guards
    set!(GuardTrue, 1);
    set!(GuardFalse, 1);
    set!(VecGuardTrue, 1);
    set!(VecGuardFalse, 1);
    set!(GuardValue, 2);
    set!(GuardClass, 2);
    set!(GuardNonnull, 1);
    set!(GuardIsnull, 1);
    set!(GuardNonnullClass, 2);
    set!(GuardGcType, 2);
    set!(GuardIsObject, 1);
    set!(GuardSubclass, 2);
    set!(GuardNoException, 0);
    set!(GuardException, 1);
    set!(GuardNoOverflow, 0);
    set!(GuardOverflow, 0);
    set!(GuardNotForced, 0);
    set!(GuardNotForced2, 0);
    set!(GuardNotInvalidated, 0);
    set!(GuardEvalBreaker, 0);
    set!(GuardFutureCondition, 0);
    set!(GuardAlwaysFails, 0);
    // Arithmetic (binary)
    set!(IntAdd, 2);
    set!(IntSub, 2);
    set!(IntMul, 2);
    set!(UintMulHigh, 2);
    set!(IntFloorDiv, 2);
    set!(IntMod, 2);
    set!(IntAnd, 2);
    set!(IntOr, 2);
    set!(IntXor, 2);
    set!(IntRshift, 2);
    set!(IntLshift, 2);
    set!(UintRshift, 2);
    set!(IntSignext, 2);
    set!(FloatAdd, 2);
    set!(FloatSub, 2);
    set!(FloatMul, 2);
    set!(FloatTrueDiv, 2);
    set!(FloatFloorDiv, 2);
    set!(FloatMod, 2);
    set!(FloatNeg, 1);
    set!(FloatAbs, 1);
    // Casts (unary)
    set!(CastFloatToInt, 1);
    set!(CastIntToFloat, 1);
    set!(CastFloatToSinglefloat, 1);
    set!(CastSinglefloatToFloat, 1);
    set!(ConvertFloatBytesToLonglong, 1);
    set!(ConvertLonglongBytesToFloat, 1);
    // Vector arithmetic (binary/unary)
    set!(VecIntAdd, 2);
    set!(VecIntSub, 2);
    set!(VecIntMul, 2);
    set!(VecIntAnd, 2);
    set!(VecIntOr, 2);
    set!(VecIntXor, 2);
    set!(VecFloatAdd, 2);
    set!(VecFloatSub, 2);
    set!(VecFloatMul, 2);
    set!(VecFloatTrueDiv, 2);
    set!(VecFloatNeg, 1);
    set!(VecFloatAbs, 1);
    set!(VecFloatEq, 2);
    set!(VecFloatNe, 2);
    set!(VecFloatXor, 2);
    set!(VecIntIsTrue, 1);
    set!(VecIntNe, 2);
    set!(VecIntEq, 2);
    set!(VecIntSignext, 2);
    set!(VecCastFloatToSinglefloat, 1);
    set!(VecCastSinglefloatToFloat, 1);
    set!(VecCastFloatToInt, 1);
    set!(VecCastIntToFloat, 1);
    set!(VecI, 0);
    set!(VecF, 0);
    set!(VecUnpackI, 3);
    set!(VecUnpackF, 3);
    set!(VecPackI, 4);
    set!(VecPackF, 4);
    set!(VecExpandI, 1);
    set!(VecExpandF, 1);
    // Comparisons
    set!(IntLt, 2);
    set!(IntLe, 2);
    set!(IntEq, 2);
    set!(IntNe, 2);
    set!(IntGt, 2);
    set!(IntGe, 2);
    set!(UintLt, 2);
    set!(UintLe, 2);
    set!(UintGt, 2);
    set!(UintGe, 2);
    set!(FloatLt, 2);
    set!(FloatLe, 2);
    set!(FloatEq, 2);
    set!(FloatNe, 2);
    set!(FloatGt, 2);
    set!(FloatGe, 2);
    // Unary int
    set!(IntIsZero, 1);
    set!(IntIsTrue, 1);
    set!(IntNeg, 1);
    set!(IntInvert, 1);
    set!(IntForceGeZero, 1);
    set!(IntBetween, 3);
    // Identity/cast
    set!(SameAsI, 1);
    set!(SameAsR, 1);
    set!(SameAsF, 1);
    set!(CastPtrToInt, 1);
    set!(CastIntToPtr, 1);
    set!(CastOpaquePtr, 1);
    // Pointer comparisons
    set!(PtrEq, 2);
    set!(PtrNe, 2);
    set!(InstancePtrEq, 2);
    set!(InstancePtrNe, 2);
    set!(NurseryPtrIncrement, 2);
    // Array/string length
    set!(ArraylenGc, 1);
    set!(Strlen, 1);
    set!(Strgetitem, 2);
    set!(GetarrayitemGcPureI, 2);
    set!(GetarrayitemGcPureR, 2);
    set!(GetarrayitemGcPureF, 2);
    set!(Unicodelen, 1);
    set!(Unicodegetitem, 2);
    set!(LoadFromGcTable, 1);
    set!(LoadEffectiveAddress, 4);
    // Thread-local
    set!(ThreadlocalrefGet, 0);
    // GC load
    set!(GcLoadI, 3);
    set!(GcLoadR, 3);
    set!(GcLoadF, 3);
    set!(GcLoadIndexedI, 5);
    set!(GcLoadIndexedR, 5);
    set!(GcLoadIndexedF, 5);
    // Array/field get
    set!(GetarrayitemGcI, 2);
    set!(GetarrayitemGcR, 2);
    set!(GetarrayitemGcF, 2);
    set!(GetarrayitemRawI, 2);
    set!(GetarrayitemRawR, 2);
    set!(GetarrayitemRawF, 2);
    set!(RawLoadI, 2);
    set!(RawLoadF, 2);
    set!(VecLoadI, 4);
    set!(VecLoadF, 4);
    set!(GetinteriorfieldGcI, 2);
    set!(GetinteriorfieldGcR, 2);
    set!(GetinteriorfieldGcF, 2);
    set!(GetfieldGcI, 1);
    set!(GetfieldGcR, 1);
    set!(GetfieldGcF, 1);
    set!(GetfieldRawI, 1);
    set!(GetfieldRawR, 1);
    set!(GetfieldRawF, 1);
    set!(GetfieldGcPureI, 1);
    set!(GetfieldGcPureR, 1);
    set!(GetfieldGcPureF, 1);
    // Allocation
    set!(New, 0);
    set!(NewWithVtable, 0);
    set!(NewArray, 1);
    set!(NewArrayClear, 1);
    set!(Newstr, 1);
    set!(Newunicode, 1);
    // Misc no-side-effect
    set!(ForceToken, 0);
    set!(VirtualRefI, 2);
    set!(VirtualRefR, 2);
    set!(Strhash, 1);
    set!(Unicodehash, 1);
    // GC store
    set!(GcStore, 4);
    set!(GcStoreIndexed, 6);
    set!(IncrementDebugCounter, 1);
    // Array/field set
    set!(SetarrayitemGc, 3);
    set!(SetarrayitemRaw, 3);
    set!(RawStore, 3);
    set!(VecStore, 5);
    set!(SetinteriorfieldGc, 3);
    set!(SetinteriorfieldRaw, 3);
    set!(SetfieldGc, 2);
    set!(ZeroArray, 5);
    set!(SetfieldRaw, 2);
    set!(Strsetitem, 3);
    set!(Unicodesetitem, 3);
    // GC write barriers
    set!(CondCallGcWb, 1);
    set!(CondCallGcWbArray, 2);
    // Debug (variadic) - already None
    // Portal frames
    set!(EnterPortalFrame, 2);
    set!(LeavePortalFrame, 1);
    // Misc
    set!(ForceSpill, 1);
    set!(VirtualRefFinish, 2);
    set!(Copystrcontent, 5);
    set!(Copyunicodecontent, 5);
    set!(QuasiimmutField, 1);
    set!(AssertNotNone, 1);
    set!(RecordExactClass, 2);
    set!(RecordExactValueR, 2);
    set!(RecordExactValueI, 2);
    set!(Keepalive, 1);
    set!(SaveException, 0);
    set!(SaveExcClass, 0);
    set!(RestoreException, 2);
    // Calls: all variadic (None) - default
    set!(CheckMemoryError, 1);
    set!(CallMallocNursery, 1);
    set!(CallMallocNurseryVarsizeFrame, 1);
    // Overflow
    set!(IntAddOvf, 2);
    set!(IntSubOvf, 2);
    set!(IntMulOvf, 2);
    t
};

/// Whether the operation takes a descriptor.
static OPWITHDESCR: [bool; OPCODE_COUNT] = {
    let mut t = [false; OPCODE_COUNT];
    use OpCode::*;
    macro_rules! set {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = true;)+
        };
    }
    set!(
        Jump,
        Finish,
        Label,
        // Guards
        GuardTrue,
        GuardFalse,
        VecGuardTrue,
        VecGuardFalse,
        GuardValue,
        GuardClass,
        GuardNonnull,
        GuardIsnull,
        GuardNonnullClass,
        GuardGcType,
        GuardIsObject,
        GuardSubclass,
        GuardNoException,
        GuardException,
        GuardNoOverflow,
        GuardOverflow,
        GuardNotForced,
        GuardNotForced2,
        GuardNotInvalidated,
        GuardEvalBreaker,
        GuardFutureCondition,
        GuardAlwaysFails,
        // Array/field access
        ArraylenGc,
        GetarrayitemGcPureI,
        GetarrayitemGcPureR,
        GetarrayitemGcPureF,
        GetarrayitemGcI,
        GetarrayitemGcR,
        GetarrayitemGcF,
        GetarrayitemRawI,
        GetarrayitemRawR,
        GetarrayitemRawF,
        RawLoadI,
        RawLoadF,
        VecLoadI,
        VecLoadF,
        GetinteriorfieldGcI,
        GetinteriorfieldGcR,
        GetinteriorfieldGcF,
        GetfieldGcI,
        GetfieldGcR,
        GetfieldGcF,
        GetfieldRawI,
        GetfieldRawR,
        GetfieldRawF,
        GetfieldGcPureI,
        GetfieldGcPureR,
        GetfieldGcPureF,
        // Allocation
        New,
        NewWithVtable,
        NewArray,
        NewArrayClear,
        // Stores
        GcStore,
        GcStoreIndexed,
        SetarrayitemGc,
        SetarrayitemRaw,
        RawStore,
        VecStore,
        SetinteriorfieldGc,
        SetinteriorfieldRaw,
        SetfieldGc,
        ZeroArray,
        SetfieldRaw,
        // GC barriers
        CondCallGcWb,
        CondCallGcWbArray,
        // Misc
        QuasiimmutField,
        // Calls
        CallI,
        CallR,
        CallF,
        CallN,
        CondCallN,
        CondCallValueI,
        CondCallValueR,
        CallAssemblerI,
        CallAssemblerR,
        CallAssemblerF,
        CallAssemblerN,
        CallMayForceI,
        CallMayForceR,
        CallMayForceF,
        CallMayForceN,
        CallLoopinvariantI,
        CallLoopinvariantR,
        CallLoopinvariantF,
        CallLoopinvariantN,
        CallReleaseGilI,
        CallReleaseGilF,
        CallReleaseGilN,
        CallPureI,
        CallPureR,
        CallPureF,
        CallPureN,
        CallMallocNurseryVarsize,
        ThreadlocalrefGet,
        RecordKnownResult
    );
    t
};

/// Whether the operation returns a boolean result.
static OPBOOL: [bool; OPCODE_COUNT] = {
    let mut t = [false; OPCODE_COUNT];
    use OpCode::*;
    macro_rules! set {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = true;)+
        };
    }
    set!(
        IntLt,
        IntLe,
        IntEq,
        IntNe,
        IntGt,
        IntGe,
        UintLt,
        UintLe,
        UintGt,
        UintGe,
        FloatLt,
        FloatLe,
        FloatEq,
        FloatNe,
        FloatGt,
        FloatGe,
        IntIsZero,
        IntIsTrue,
        IntBetween,
        PtrEq,
        PtrNe,
        InstancePtrEq,
        InstancePtrNe,
        VecFloatEq,
        VecFloatNe,
        VecIntIsTrue,
        VecIntNe,
        VecIntEq
    );
    t
};

/// Result type of each operation.
static OPRESTYPE: [Type; OPCODE_COUNT] = {
    let mut t = [Type::Void; OPCODE_COUNT];
    use OpCode::*;

    macro_rules! int {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = Type::Int;)+
        };
    }
    macro_rules! float {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = Type::Float;)+
        };
    }
    macro_rules! ref_ {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = Type::Ref;)+
        };
    }

    int!(
        IntAdd,
        IntSub,
        IntMul,
        UintMulHigh,
        IntFloorDiv,
        IntMod,
        IntAnd,
        IntOr,
        IntXor,
        IntRshift,
        IntLshift,
        UintRshift,
        IntSignext,
        CastFloatToInt,
        CastFloatToSinglefloat,
        ConvertFloatBytesToLonglong,
        // Vector int
        VecIntAdd,
        VecIntSub,
        VecIntMul,
        VecIntAnd,
        VecIntOr,
        VecIntXor,
        VecFloatEq,
        VecFloatNe,
        VecIntIsTrue,
        VecIntNe,
        VecIntEq,
        VecIntSignext,
        VecCastFloatToSinglefloat,
        VecCastFloatToInt,
        // Comparisons (all return int)
        IntLt,
        IntLe,
        IntEq,
        IntNe,
        IntGt,
        IntGe,
        UintLt,
        UintLe,
        UintGt,
        UintGe,
        FloatLt,
        FloatLe,
        FloatEq,
        FloatNe,
        FloatGt,
        FloatGe,
        IntIsZero,
        IntIsTrue,
        IntNeg,
        IntInvert,
        IntForceGeZero,
        IntBetween,
        SameAsI,
        CastPtrToInt,
        PtrEq,
        PtrNe,
        InstancePtrEq,
        InstancePtrNe,
        ArraylenGc,
        Strlen,
        Strgetitem,
        GetarrayitemGcPureI,
        Unicodelen,
        Unicodegetitem,
        LoadEffectiveAddress,
        GcLoadI,
        GcLoadIndexedI,
        GetarrayitemGcI,
        GetarrayitemRawI,
        RawLoadI,
        GetinteriorfieldGcI,
        GetfieldGcI,
        GetfieldRawI,
        GetfieldGcPureI,
        Strhash,
        Unicodehash,
        CondCallValueI,
        CallI,
        CallPureI,
        CallMayForceI,
        CallAssemblerI,
        CallLoopinvariantI,
        CallReleaseGilI,
        SaveExcClass,
        RecordExactValueI,
        IntAddOvf,
        IntSubOvf,
        IntMulOvf
    );

    float!(
        FloatAdd,
        FloatSub,
        FloatMul,
        FloatTrueDiv,
        FloatFloorDiv,
        FloatMod,
        FloatNeg,
        FloatAbs,
        CastIntToFloat,
        CastSinglefloatToFloat,
        ConvertLonglongBytesToFloat,
        VecFloatAdd,
        VecFloatSub,
        VecFloatMul,
        VecFloatTrueDiv,
        VecFloatNeg,
        VecFloatAbs,
        VecFloatXor,
        VecCastSinglefloatToFloat,
        VecCastIntToFloat,
        SameAsF,
        GetarrayitemGcPureF,
        GcLoadF,
        GcLoadIndexedF,
        GetarrayitemGcF,
        GetarrayitemRawF,
        RawLoadF,
        GetinteriorfieldGcF,
        GetfieldGcF,
        GetfieldRawF,
        GetfieldGcPureF,
        CallF,
        CallPureF,
        CallMayForceF,
        CallAssemblerF,
        CallLoopinvariantF,
        CallReleaseGilF
    );

    ref_!(
        CastIntToPtr,
        CastOpaquePtr,
        SameAsR,
        NurseryPtrIncrement,
        GetarrayitemGcPureR,
        LoadFromGcTable,
        GcLoadR,
        GcLoadIndexedR,
        GetarrayitemGcR,
        GetarrayitemRawR,
        GetinteriorfieldGcR,
        GetfieldGcR,
        GetfieldRawR,
        GetfieldGcPureR,
        New,
        NewWithVtable,
        NewArray,
        NewArrayClear,
        Newstr,
        Newunicode,
        ForceToken,
        VirtualRefR,
        GuardException,
        CondCallValueR,
        CallR,
        CallPureR,
        CallMayForceR,
        CallAssemblerR,
        CallLoopinvariantR,
        ThreadlocalrefGet,
        CallMallocNursery,
        CallMallocNurseryVarsize,
        CallMallocNurseryVarsizeFrame,
        SaveException
    );

    // VecI/VecF, VecUnpack*, VecPack*, VecExpand* can be either int or float
    // depending on usage. Default to int for I variants, float for F variants.
    int!(VecI, VecUnpackI, VecPackI, VecExpandI, VecLoadI);
    float!(VecF, VecUnpackF, VecPackF, VecExpandF, VecLoadF);
    int!(VirtualRefI);
    t
};

/// Operation names for debugging.
static OPNAME: [&str; OPCODE_COUNT] = {
    let mut t = [""; OPCODE_COUNT];
    use OpCode::*;
    macro_rules! name {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = stringify!($op);)+
        };
    }
    name!(
        Jump,
        Finish,
        Label,
        GuardTrue,
        GuardFalse,
        VecGuardTrue,
        VecGuardFalse,
        GuardValue,
        GuardClass,
        GuardNonnull,
        GuardIsnull,
        GuardNonnullClass,
        GuardGcType,
        GuardIsObject,
        GuardSubclass,
        GuardNoException,
        GuardException,
        GuardNoOverflow,
        GuardOverflow,
        GuardNotForced,
        GuardNotForced2,
        GuardNotInvalidated,
        GuardEvalBreaker,
        GuardFutureCondition,
        GuardAlwaysFails,
        IntAdd,
        IntSub,
        IntMul,
        UintMulHigh,
        IntFloorDiv,
        IntMod,
        IntAnd,
        IntOr,
        IntXor,
        IntRshift,
        IntLshift,
        UintRshift,
        IntSignext,
        FloatAdd,
        FloatSub,
        FloatMul,
        FloatTrueDiv,
        FloatFloorDiv,
        FloatMod,
        FloatNeg,
        FloatAbs,
        CastFloatToInt,
        CastIntToFloat,
        CastFloatToSinglefloat,
        CastSinglefloatToFloat,
        ConvertFloatBytesToLonglong,
        ConvertLonglongBytesToFloat,
        VecIntAdd,
        VecIntSub,
        VecIntMul,
        VecIntAnd,
        VecIntOr,
        VecIntXor,
        VecFloatAdd,
        VecFloatSub,
        VecFloatMul,
        VecFloatTrueDiv,
        VecFloatNeg,
        VecFloatAbs,
        VecFloatEq,
        VecFloatNe,
        VecFloatXor,
        VecIntIsTrue,
        VecIntNe,
        VecIntEq,
        VecIntSignext,
        VecCastFloatToSinglefloat,
        VecCastSinglefloatToFloat,
        VecCastFloatToInt,
        VecCastIntToFloat,
        VecI,
        VecF,
        VecUnpackI,
        VecUnpackF,
        VecPackI,
        VecPackF,
        VecExpandI,
        VecExpandF,
        IntLt,
        IntLe,
        IntEq,
        IntNe,
        IntGt,
        IntGe,
        UintLt,
        UintLe,
        UintGt,
        UintGe,
        FloatLt,
        FloatLe,
        FloatEq,
        FloatNe,
        FloatGt,
        FloatGe,
        IntIsZero,
        IntIsTrue,
        IntNeg,
        IntInvert,
        IntForceGeZero,
        IntBetween,
        SameAsI,
        SameAsR,
        SameAsF,
        CastPtrToInt,
        CastIntToPtr,
        CastOpaquePtr,
        PtrEq,
        PtrNe,
        InstancePtrEq,
        InstancePtrNe,
        NurseryPtrIncrement,
        ArraylenGc,
        Strlen,
        Strgetitem,
        GetarrayitemGcPureI,
        GetarrayitemGcPureR,
        GetarrayitemGcPureF,
        Unicodelen,
        Unicodegetitem,
        LoadFromGcTable,
        LoadEffectiveAddress,
        ThreadlocalrefGet,
        GcLoadI,
        GcLoadR,
        GcLoadF,
        GcLoadIndexedI,
        GcLoadIndexedR,
        GcLoadIndexedF,
        GetarrayitemGcI,
        GetarrayitemGcR,
        GetarrayitemGcF,
        GetarrayitemRawI,
        GetarrayitemRawR,
        GetarrayitemRawF,
        RawLoadI,
        RawLoadF,
        VecLoadI,
        VecLoadF,
        GetinteriorfieldGcI,
        GetinteriorfieldGcR,
        GetinteriorfieldGcF,
        GetfieldGcI,
        GetfieldGcR,
        GetfieldGcF,
        GetfieldRawI,
        GetfieldRawR,
        GetfieldRawF,
        GetfieldGcPureI,
        GetfieldGcPureR,
        GetfieldGcPureF,
        New,
        NewWithVtable,
        NewArray,
        NewArrayClear,
        Newstr,
        Newunicode,
        ForceToken,
        VirtualRefI,
        VirtualRefR,
        Strhash,
        Unicodehash,
        GcStore,
        GcStoreIndexed,
        IncrementDebugCounter,
        SetarrayitemGc,
        SetarrayitemRaw,
        RawStore,
        VecStore,
        SetinteriorfieldGc,
        SetinteriorfieldRaw,
        SetfieldGc,
        ZeroArray,
        SetfieldRaw,
        Strsetitem,
        Unicodesetitem,
        CondCallGcWb,
        CondCallGcWbArray,
        DebugMergePoint,
        EnterPortalFrame,
        LeavePortalFrame,
        JitDebug,
        ForceSpill,
        VirtualRefFinish,
        Copystrcontent,
        Copyunicodecontent,
        QuasiimmutField,
        AssertNotNone,
        RecordExactClass,
        RecordExactValueR,
        RecordExactValueI,
        Keepalive,
        SaveException,
        SaveExcClass,
        RestoreException,
        CallI,
        CallR,
        CallF,
        CallN,
        CondCallN,
        CondCallValueI,
        CondCallValueR,
        CallAssemblerI,
        CallAssemblerR,
        CallAssemblerF,
        CallAssemblerN,
        CallMayForceI,
        CallMayForceR,
        CallMayForceF,
        CallMayForceN,
        CallLoopinvariantI,
        CallLoopinvariantR,
        CallLoopinvariantF,
        CallLoopinvariantN,
        CallReleaseGilI,
        CallReleaseGilF,
        CallReleaseGilN,
        CallPureI,
        CallPureR,
        CallPureF,
        CallPureN,
        CheckMemoryError,
        CallMallocNursery,
        CallMallocNurseryVarsize,
        CallMallocNurseryVarsizeFrame,
        RecordKnownResult,
        IntAddOvf,
        IntSubOvf,
        IntMulOvf
    );
    t
};

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! op {
        ($($field:tt)*) => {{
            let mut __op = Op {
                $($field)*
                type_: Type::Void,
                forwarded: std::cell::RefCell::new(crate::forwarding::Forwarded::None),
                value: std::cell::Cell::new(None),
            };
            __op.type_ = __op.opcode.result_type();
            __op
        }};
    }

    /// Iterate over all defined OpCode variants.
    fn all_opcodes() -> impl Iterator<Item = OpCode> {
        (0..OPCODE_COUNT as u16).map(|i| unsafe { std::mem::transmute::<u16, OpCode>(i) })
    }

    // ══════════════════════════════════════════════════════════════════
    // Resoperation parity tests
    // Ported from rpython/jit/metainterp/test/test_resoperation.py
    // ══════════════════════════════════════════════════════════════════

    // ── OpRef Eq/Hash sentinel coverage ──

    #[test]
    fn opref_typed_variants_disjoint_from_none() {
        // Variant-aware Eq: `OpRef::None` and any typed variant are
        // disjoint identities. RPython parity (resoperation.py:38
        // same_box: self is other) — Python `None` vs a Box object are
        // never identical.
        let none = OpRef::NONE;
        assert!(none.is_none());
        assert_ne!(none, OpRef::int_op(0));
        assert_ne!(OpRef::int_op(0), none);
        assert_ne!(OpRef::int_op(0), OpRef::ref_op(0));
        assert_ne!(OpRef::int_op(0), OpRef::float_op(0));
        assert_ne!(OpRef::ref_op(0), OpRef::float_op(0));
    }

    // ── Metadata table coverage ──

    #[test]
    fn test_every_opcode_has_name() {
        for op in all_opcodes() {
            let name = op.name();
            assert!(
                !name.is_empty(),
                "OpCode {:?} (u16={}) has empty name",
                op,
                op.as_u16()
            );
        }
    }

    /// `from_u16` is the left-inverse of `as_u16` over the defined
    /// `0..OPCODE_COUNT` range, and returns `None` everywhere outside it.
    #[test]
    fn test_opcode_from_u16_roundtrip() {
        for op in all_opcodes() {
            let n = op.as_u16();
            assert_eq!(OpCode::from_u16(n), Some(op));
        }
        assert_eq!(OpCode::from_u16(OPCODE_COUNT as u16), None);
        assert_eq!(OpCode::from_u16(u16::MAX), None);
    }

    #[test]
    fn test_every_opcode_has_result_type() {
        for op in all_opcodes() {
            let _tp = op.result_type();
        }
    }

    #[test]
    fn test_every_opcode_has_arity_entry() {
        for op in all_opcodes() {
            let _arity = op.arity();
        }
    }

    #[test]
    fn test_every_opcode_has_descr_entry() {
        for op in all_opcodes() {
            let _has_descr = op.has_descr();
        }
    }

    #[test]
    fn test_every_opcode_has_bool_entry() {
        for op in all_opcodes() {
            let _returns_bool = op.returns_bool();
        }
    }

    // ── Arity: nullary / unary / binary / variadic ──

    #[test]
    fn test_arity_nullary() {
        let nullary_ops = [
            OpCode::New,
            OpCode::NewWithVtable,
            OpCode::ForceToken,
            OpCode::GuardNoException,
            OpCode::GuardNoOverflow,
            OpCode::GuardOverflow,
            OpCode::GuardNotForced,
            OpCode::GuardNotForced2,
            OpCode::GuardNotInvalidated,
            OpCode::GuardEvalBreaker,
            OpCode::GuardFutureCondition,
            OpCode::GuardAlwaysFails,
            OpCode::VecI,
            OpCode::VecF,
            OpCode::ThreadlocalrefGet,
            OpCode::SaveException,
            OpCode::SaveExcClass,
        ];
        for op in &nullary_ops {
            assert_eq!(op.arity(), Some(0), "{:?} should have arity 0", op);
        }
    }

    #[test]
    fn test_arity_unary() {
        let unary_ops = [
            OpCode::GuardTrue,
            OpCode::GuardFalse,
            OpCode::GuardNonnull,
            OpCode::GuardIsnull,
            OpCode::GuardIsObject,
            OpCode::GuardException,
            OpCode::FloatNeg,
            OpCode::FloatAbs,
            OpCode::CastFloatToInt,
            OpCode::CastIntToFloat,
            OpCode::IntIsZero,
            OpCode::IntIsTrue,
            OpCode::IntNeg,
            OpCode::IntInvert,
            OpCode::IntForceGeZero,
            OpCode::SameAsI,
            OpCode::SameAsR,
            OpCode::SameAsF,
            OpCode::CastPtrToInt,
            OpCode::CastIntToPtr,
            OpCode::CastOpaquePtr,
            OpCode::ArraylenGc,
            OpCode::Strlen,
            OpCode::Unicodelen,
            OpCode::GetfieldGcI,
            OpCode::GetfieldGcR,
            OpCode::GetfieldGcF,
            OpCode::GetfieldRawI,
            OpCode::GetfieldRawR,
            OpCode::GetfieldRawF,
            OpCode::GetfieldGcPureI,
            OpCode::GetfieldGcPureR,
            OpCode::GetfieldGcPureF,
            OpCode::NewArray,
            OpCode::NewArrayClear,
            OpCode::Newstr,
            OpCode::Newunicode,
            OpCode::Strhash,
            OpCode::Unicodehash,
            OpCode::CheckMemoryError,
            OpCode::ForceSpill,
            OpCode::QuasiimmutField,
            OpCode::AssertNotNone,
            OpCode::Keepalive,
            OpCode::CondCallGcWb,
            OpCode::LoadFromGcTable,
            OpCode::IncrementDebugCounter,
            OpCode::LeavePortalFrame,
            OpCode::CallMallocNursery,
            OpCode::CallMallocNurseryVarsizeFrame,
        ];
        for op in &unary_ops {
            assert_eq!(op.arity(), Some(1), "{:?} should have arity 1", op);
        }
    }

    #[test]
    fn test_arity_binary() {
        let binary_ops = [
            OpCode::IntAdd,
            OpCode::IntSub,
            OpCode::IntMul,
            OpCode::UintMulHigh,
            OpCode::IntFloorDiv,
            OpCode::IntMod,
            OpCode::IntAnd,
            OpCode::IntOr,
            OpCode::IntXor,
            OpCode::IntRshift,
            OpCode::IntLshift,
            OpCode::UintRshift,
            OpCode::IntSignext,
            OpCode::FloatAdd,
            OpCode::FloatSub,
            OpCode::FloatMul,
            OpCode::FloatTrueDiv,
            OpCode::FloatFloorDiv,
            OpCode::FloatMod,
            OpCode::IntLt,
            OpCode::IntLe,
            OpCode::IntEq,
            OpCode::IntNe,
            OpCode::IntGt,
            OpCode::IntGe,
            OpCode::UintLt,
            OpCode::UintLe,
            OpCode::UintGt,
            OpCode::UintGe,
            OpCode::FloatLt,
            OpCode::FloatLe,
            OpCode::FloatEq,
            OpCode::FloatNe,
            OpCode::FloatGt,
            OpCode::FloatGe,
            OpCode::PtrEq,
            OpCode::PtrNe,
            OpCode::InstancePtrEq,
            OpCode::InstancePtrNe,
            OpCode::NurseryPtrIncrement,
            OpCode::Strgetitem,
            OpCode::Unicodegetitem,
            OpCode::GuardValue,
            OpCode::GuardClass,
            OpCode::GuardNonnullClass,
            OpCode::GuardGcType,
            OpCode::GuardSubclass,
            OpCode::SetfieldGc,
            OpCode::SetfieldRaw,
            OpCode::CondCallGcWbArray,
            OpCode::VirtualRefI,
            OpCode::VirtualRefR,
            OpCode::VirtualRefFinish,
            OpCode::RecordExactClass,
            OpCode::RecordExactValueR,
            OpCode::RecordExactValueI,
            OpCode::EnterPortalFrame,
            OpCode::RestoreException,
            OpCode::RawLoadI,
            OpCode::RawLoadF,
            OpCode::GetarrayitemGcI,
            OpCode::GetarrayitemGcR,
            OpCode::GetarrayitemGcF,
            OpCode::GetarrayitemGcPureI,
            OpCode::GetarrayitemGcPureR,
            OpCode::GetarrayitemGcPureF,
            OpCode::GetarrayitemRawI,
            OpCode::GetarrayitemRawR,
            OpCode::GetarrayitemRawF,
            OpCode::GetinteriorfieldGcI,
            OpCode::GetinteriorfieldGcR,
            OpCode::GetinteriorfieldGcF,
            OpCode::IntAddOvf,
            OpCode::IntSubOvf,
            OpCode::IntMulOvf,
        ];
        for op in &binary_ops {
            assert_eq!(op.arity(), Some(2), "{:?} should have arity 2", op);
        }
    }

    #[test]
    fn test_arity_variadic() {
        let variadic_ops = [
            OpCode::Jump,
            OpCode::Finish,
            OpCode::Label,
            OpCode::DebugMergePoint,
            OpCode::JitDebug,
            OpCode::CallI,
            OpCode::CallR,
            OpCode::CallF,
            OpCode::CallN,
            OpCode::CondCallN,
            OpCode::CondCallValueI,
            OpCode::CondCallValueR,
            OpCode::CallAssemblerI,
            OpCode::CallAssemblerR,
            OpCode::CallAssemblerF,
            OpCode::CallAssemblerN,
            OpCode::CallMayForceI,
            OpCode::CallMayForceR,
            OpCode::CallMayForceF,
            OpCode::CallMayForceN,
            OpCode::CallLoopinvariantI,
            OpCode::CallLoopinvariantR,
            OpCode::CallLoopinvariantF,
            OpCode::CallLoopinvariantN,
            OpCode::CallReleaseGilI,
            OpCode::CallReleaseGilF,
            OpCode::CallReleaseGilN,
            OpCode::CallPureI,
            OpCode::CallPureR,
            OpCode::CallPureF,
            OpCode::CallPureN,
            OpCode::CallMallocNurseryVarsize,
            OpCode::RecordKnownResult,
        ];
        for op in &variadic_ops {
            assert_eq!(op.arity(), None, "{:?} should be variadic (arity=None)", op);
        }
    }

    // ── Result type exhaustive checks ──

    #[test]
    fn test_int_result_types() {
        let int_ops = [
            OpCode::IntAdd,
            OpCode::IntSub,
            OpCode::IntMul,
            OpCode::IntFloorDiv,
            OpCode::IntMod,
            OpCode::IntAnd,
            OpCode::IntOr,
            OpCode::IntXor,
            OpCode::IntRshift,
            OpCode::IntLshift,
            OpCode::UintRshift,
            OpCode::IntSignext,
            OpCode::CastFloatToInt,
            OpCode::IntLt,
            OpCode::IntLe,
            OpCode::IntEq,
            OpCode::IntNe,
            OpCode::IntGt,
            OpCode::IntGe,
            OpCode::IntIsZero,
            OpCode::IntIsTrue,
            OpCode::IntNeg,
            OpCode::IntInvert,
            OpCode::IntForceGeZero,
            OpCode::SameAsI,
            OpCode::CastPtrToInt,
            OpCode::PtrEq,
            OpCode::PtrNe,
            OpCode::IntAddOvf,
            OpCode::IntSubOvf,
            OpCode::IntMulOvf,
            OpCode::GetfieldGcI,
            OpCode::GetfieldRawI,
            OpCode::GetfieldGcPureI,
            OpCode::GetarrayitemGcI,
            OpCode::GetarrayitemRawI,
            OpCode::GetarrayitemGcPureI,
            OpCode::CallI,
            OpCode::CallPureI,
            OpCode::CallMayForceI,
            OpCode::CallAssemblerI,
            OpCode::CallLoopinvariantI,
            OpCode::CallReleaseGilI,
            OpCode::SaveExcClass,
        ];
        for op in &int_ops {
            assert_eq!(op.result_type(), Type::Int, "{:?} should return Int", op);
        }
    }

    #[test]
    fn test_float_result_types() {
        let float_ops = [
            OpCode::FloatAdd,
            OpCode::FloatSub,
            OpCode::FloatMul,
            OpCode::FloatTrueDiv,
            OpCode::FloatFloorDiv,
            OpCode::FloatMod,
            OpCode::FloatNeg,
            OpCode::FloatAbs,
            OpCode::CastIntToFloat,
            OpCode::CastSinglefloatToFloat,
            OpCode::SameAsF,
            OpCode::GetfieldGcF,
            OpCode::GetfieldRawF,
            OpCode::GetfieldGcPureF,
            OpCode::GetarrayitemGcF,
            OpCode::GetarrayitemRawF,
            OpCode::GetarrayitemGcPureF,
            OpCode::CallF,
            OpCode::CallPureF,
            OpCode::CallMayForceF,
            OpCode::CallAssemblerF,
            OpCode::CallLoopinvariantF,
            OpCode::CallReleaseGilF,
        ];
        for op in &float_ops {
            assert_eq!(
                op.result_type(),
                Type::Float,
                "{:?} should return Float",
                op
            );
        }
    }

    #[test]
    fn test_ref_result_types() {
        let ref_ops = [
            OpCode::CastIntToPtr,
            OpCode::CastOpaquePtr,
            OpCode::SameAsR,
            OpCode::NurseryPtrIncrement,
            OpCode::LoadFromGcTable,
            OpCode::New,
            OpCode::NewWithVtable,
            OpCode::NewArray,
            OpCode::NewArrayClear,
            OpCode::Newstr,
            OpCode::Newunicode,
            OpCode::ForceToken,
            OpCode::VirtualRefR,
            OpCode::GuardException,
            OpCode::GetfieldGcR,
            OpCode::GetfieldRawR,
            OpCode::GetfieldGcPureR,
            OpCode::GetarrayitemGcR,
            OpCode::GetarrayitemRawR,
            OpCode::GetarrayitemGcPureR,
            OpCode::CallR,
            OpCode::CallPureR,
            OpCode::CallMayForceR,
            OpCode::CallAssemblerR,
            OpCode::CallLoopinvariantR,
            OpCode::CondCallValueR,
            OpCode::ThreadlocalrefGet,
            OpCode::CallMallocNursery,
            OpCode::CallMallocNurseryVarsize,
            OpCode::CallMallocNurseryVarsizeFrame,
            OpCode::SaveException,
        ];
        for op in &ref_ops {
            assert_eq!(op.result_type(), Type::Ref, "{:?} should return Ref", op);
        }
    }

    #[test]
    fn test_void_result_types() {
        let void_ops = [
            OpCode::Jump,
            OpCode::Finish,
            OpCode::Label,
            OpCode::SetfieldGc,
            OpCode::SetfieldRaw,
            OpCode::SetarrayitemGc,
            OpCode::SetarrayitemRaw,
            OpCode::SetinteriorfieldGc,
            OpCode::SetinteriorfieldRaw,
            OpCode::RawStore,
            OpCode::GcStore,
            OpCode::GcStoreIndexed,
            OpCode::Strsetitem,
            OpCode::Unicodesetitem,
            OpCode::CondCallGcWb,
            OpCode::CondCallGcWbArray,
            OpCode::DebugMergePoint,
            OpCode::EnterPortalFrame,
            OpCode::LeavePortalFrame,
            OpCode::JitDebug,
            OpCode::CallN,
            OpCode::CondCallN,
            OpCode::CallAssemblerN,
            OpCode::CallMayForceN,
            OpCode::CallLoopinvariantN,
            OpCode::CallReleaseGilN,
            OpCode::CallPureN,
            OpCode::ForceSpill,
            OpCode::VirtualRefFinish,
            OpCode::Copystrcontent,
            OpCode::Copyunicodecontent,
            OpCode::QuasiimmutField,
            OpCode::AssertNotNone,
            OpCode::RecordExactClass,
            OpCode::Keepalive,
            OpCode::RestoreException,
            OpCode::ZeroArray,
            OpCode::VecStore,
            OpCode::IncrementDebugCounter,
        ];
        for op in &void_ops {
            assert_eq!(op.result_type(), Type::Void, "{:?} should return Void", op);
        }
    }

    // ── Classification methods ──

    #[test]
    fn test_category_classification() {
        assert!(OpCode::Jump.is_final());
        assert!(OpCode::Finish.is_final());
        assert!(!OpCode::Label.is_final());

        assert!(OpCode::GuardTrue.is_guard());
        assert!(OpCode::GuardAlwaysFails.is_guard());
        assert!(!OpCode::IntAdd.is_guard());

        assert!(OpCode::IntAdd.is_always_pure());
        assert!(OpCode::FloatMul.is_always_pure());
        assert!(OpCode::GetfieldGcPureI.is_always_pure());
        assert!(!OpCode::SetfieldGc.is_always_pure());

        assert!(OpCode::IntAddOvf.is_ovf());
        assert!(!OpCode::IntAdd.is_ovf());

        assert!(OpCode::CallI.is_call());
        assert!(OpCode::CallPureN.is_call());
        assert!(!OpCode::IntAdd.is_call());
    }

    #[test]
    fn test_guard_classification_exhaustive() {
        let all_guards: Vec<OpCode> = all_opcodes().filter(|op| op.is_guard()).collect();
        assert!(
            all_guards.len() >= 20,
            "expected at least 20 guard ops, got {}",
            all_guards.len()
        );
        let expected_guards = [
            OpCode::GuardTrue,
            OpCode::GuardFalse,
            OpCode::VecGuardTrue,
            OpCode::VecGuardFalse,
            OpCode::GuardValue,
            OpCode::GuardClass,
            OpCode::GuardNonnull,
            OpCode::GuardIsnull,
            OpCode::GuardNonnullClass,
            OpCode::GuardGcType,
            OpCode::GuardIsObject,
            OpCode::GuardSubclass,
            OpCode::GuardNoException,
            OpCode::GuardException,
            OpCode::GuardNoOverflow,
            OpCode::GuardOverflow,
            OpCode::GuardNotForced,
            OpCode::GuardNotForced2,
            OpCode::GuardNotInvalidated,
            OpCode::GuardEvalBreaker,
            OpCode::GuardFutureCondition,
            OpCode::GuardAlwaysFails,
        ];
        for op in &expected_guards {
            assert!(op.is_guard(), "{:?} should be a guard", op);
        }
    }

    #[test]
    fn test_foldable_guard_subset() {
        let foldable_guards = [
            OpCode::GuardTrue,
            OpCode::GuardFalse,
            OpCode::VecGuardTrue,
            OpCode::VecGuardFalse,
            OpCode::GuardValue,
            OpCode::GuardClass,
            OpCode::GuardNonnull,
            OpCode::GuardIsnull,
            OpCode::GuardNonnullClass,
            OpCode::GuardGcType,
            OpCode::GuardIsObject,
            OpCode::GuardSubclass,
        ];
        for op in &foldable_guards {
            assert!(op.is_foldable_guard(), "{:?} should be foldable", op);
            assert!(
                op.is_guard(),
                "foldable guard {:?} must also be a guard",
                op
            );
        }
        let non_foldable = [
            OpCode::GuardNoException,
            OpCode::GuardNotForced,
            OpCode::GuardNotInvalidated,
            OpCode::GuardEvalBreaker,
            OpCode::GuardAlwaysFails,
        ];
        for op in &non_foldable {
            assert!(!op.is_foldable_guard(), "{:?} should NOT be foldable", op);
            assert!(op.is_guard(), "{:?} should still be a guard", op);
        }
    }

    #[test]
    fn test_pure_ops_no_side_effect() {
        for op in all_opcodes() {
            if op.is_always_pure() {
                assert!(
                    op.has_no_side_effect(),
                    "{:?} is pure but does not claim no_side_effect",
                    op
                );
            }
        }
    }

    #[test]
    fn test_no_side_effect_superset_of_pure() {
        let extra_nosideeffect = [
            OpCode::GcLoadI,
            OpCode::GcLoadR,
            OpCode::GcLoadF,
            OpCode::GetarrayitemGcI,
            OpCode::GetarrayitemGcR,
            OpCode::GetarrayitemGcF,
            OpCode::GetfieldGcI,
            OpCode::GetfieldGcR,
            OpCode::GetfieldGcF,
            OpCode::New,
            OpCode::NewWithVtable,
            OpCode::NewArray,
            OpCode::ForceToken,
            OpCode::Strhash,
            OpCode::Unicodehash,
        ];
        for op in &extra_nosideeffect {
            assert!(
                op.has_no_side_effect(),
                "{:?} should have no_side_effect",
                op
            );
        }
    }

    #[test]
    fn test_can_malloc() {
        assert!(OpCode::New.can_malloc());
        assert!(OpCode::NewWithVtable.can_malloc());
        assert!(OpCode::NewArray.can_malloc());
        assert!(OpCode::CallN.can_malloc());
        assert!(OpCode::CallI.can_malloc());
        assert!(OpCode::CallMayForceI.can_malloc());
        assert!(!OpCode::IntAdd.can_malloc());
        assert!(!OpCode::GuardTrue.can_malloc());
    }

    #[test]
    fn test_is_comparison() {
        let comparisons = [
            OpCode::IntLt,
            OpCode::IntLe,
            OpCode::IntEq,
            OpCode::IntNe,
            OpCode::IntGt,
            OpCode::IntGe,
            OpCode::UintLt,
            OpCode::UintLe,
            OpCode::UintGt,
            OpCode::UintGe,
            OpCode::FloatLt,
            OpCode::FloatLe,
            OpCode::FloatEq,
            OpCode::FloatNe,
            OpCode::FloatGt,
            OpCode::FloatGe,
            OpCode::PtrEq,
            OpCode::PtrNe,
            OpCode::InstancePtrEq,
            OpCode::InstancePtrNe,
            OpCode::IntIsZero,
            OpCode::IntIsTrue,
            OpCode::IntBetween,
        ];
        for op in &comparisons {
            assert!(op.is_comparison(), "{:?} should be a comparison", op);
            assert!(op.is_always_pure(), "comparison {:?} must be pure", op);
            assert!(op.returns_bool(), "comparison {:?} must return bool", op);
        }
        assert!(!OpCode::IntAdd.is_comparison());
        assert!(!OpCode::FloatAdd.is_comparison());
    }

    #[test]
    fn test_guard_exception_classification() {
        assert!(OpCode::GuardException.is_guard_exception());
        assert!(OpCode::GuardNoException.is_guard_exception());
        assert!(!OpCode::GuardTrue.is_guard_exception());
    }

    #[test]
    fn test_guard_overflow_classification() {
        assert!(OpCode::GuardOverflow.is_guard_overflow());
        assert!(OpCode::GuardNoOverflow.is_guard_overflow());
        assert!(!OpCode::GuardTrue.is_guard_overflow());
    }

    #[test]
    fn test_call_subcategories() {
        for op in all_opcodes() {
            if op.is_plain_call()
                || op.is_call_assembler()
                || op.is_call_may_force()
                || op.is_call_pure()
                || op.is_call_release_gil()
                || op.is_call_loopinvariant()
                || op.is_cond_call_value()
            {
                assert!(
                    op.is_call(),
                    "{:?} is a call subcategory but not is_call()",
                    op
                );
            }
        }
    }

    #[test]
    fn test_is_same_as() {
        assert!(OpCode::SameAsI.is_same_as());
        assert!(OpCode::SameAsR.is_same_as());
        assert!(OpCode::SameAsF.is_same_as());
        assert!(!OpCode::IntAdd.is_same_as());
    }

    // ── Typed dispatch ──

    #[test]
    fn test_call_for_type() {
        assert_eq!(OpCode::call_for_type(Type::Int), OpCode::CallI);
        assert_eq!(OpCode::call_for_type(Type::Ref), OpCode::CallR);
        assert_eq!(OpCode::call_for_type(Type::Float), OpCode::CallF);
        assert_eq!(OpCode::call_for_type(Type::Void), OpCode::CallN);
    }

    #[test]
    fn test_call_pure_for_type() {
        assert_eq!(OpCode::call_pure_for_type(Type::Int), OpCode::CallPureI);
        assert_eq!(OpCode::call_pure_for_type(Type::Float), OpCode::CallPureF);
    }

    #[test]
    fn test_same_as_for_type() {
        assert_eq!(OpCode::same_as_for_type(Type::Int), OpCode::SameAsI);
        assert_eq!(OpCode::same_as_for_type(Type::Ref), OpCode::SameAsR);
        assert_eq!(OpCode::same_as_for_type(Type::Float), OpCode::SameAsF);
    }

    #[test]
    fn test_getfield_for_type() {
        assert_eq!(OpCode::getfield_for_type(Type::Int), OpCode::GetfieldGcI);
        assert_eq!(OpCode::getfield_for_type(Type::Ref), OpCode::GetfieldGcR);
        assert_eq!(OpCode::getfield_for_type(Type::Float), OpCode::GetfieldGcF);
    }

    // ── bool_inverse / bool_reflex ──

    #[test]
    fn test_bool_inverse() {
        assert_eq!(OpCode::IntEq.bool_inverse(), Some(OpCode::IntNe));
        assert_eq!(OpCode::IntNe.bool_inverse(), Some(OpCode::IntEq));
        assert_eq!(OpCode::IntLt.bool_inverse(), Some(OpCode::IntGe));
        assert_eq!(OpCode::IntGe.bool_inverse(), Some(OpCode::IntLt));
        assert_eq!(OpCode::IntGt.bool_inverse(), Some(OpCode::IntLe));
        assert_eq!(OpCode::IntLe.bool_inverse(), Some(OpCode::IntGt));
        assert_eq!(OpCode::FloatEq.bool_inverse(), Some(OpCode::FloatNe));
        assert_eq!(OpCode::FloatLt.bool_inverse(), Some(OpCode::FloatGe));
        assert_eq!(OpCode::UintLt.bool_inverse(), Some(OpCode::UintGe));
        assert_eq!(OpCode::PtrEq.bool_inverse(), Some(OpCode::PtrNe));
        assert_eq!(OpCode::IntAdd.bool_inverse(), None);
    }

    #[test]
    fn test_bool_inverse_is_involution() {
        for op in all_opcodes() {
            if let Some(inv) = op.bool_inverse() {
                assert_eq!(
                    inv.bool_inverse(),
                    Some(op),
                    "bool_inverse should be an involution for {:?}",
                    op
                );
            }
        }
    }

    #[test]
    fn test_bool_reflex() {
        assert_eq!(OpCode::IntLt.bool_reflex(), Some(OpCode::IntGt));
        assert_eq!(OpCode::IntGt.bool_reflex(), Some(OpCode::IntLt));
        assert_eq!(OpCode::IntEq.bool_reflex(), Some(OpCode::IntEq));
        assert_eq!(OpCode::IntNe.bool_reflex(), Some(OpCode::IntNe));
        assert_eq!(OpCode::FloatLt.bool_reflex(), Some(OpCode::FloatGt));
        assert_eq!(OpCode::PtrEq.bool_reflex(), Some(OpCode::PtrEq));
        assert_eq!(OpCode::IntAdd.bool_reflex(), None);
    }

    #[test]
    fn test_bool_reflex_is_involution() {
        for op in all_opcodes() {
            if let Some(refl) = op.bool_reflex() {
                assert_eq!(
                    refl.bool_reflex(),
                    Some(op),
                    "bool_reflex should be an involution for {:?}",
                    op
                );
            }
        }
    }

    // ── without_overflow / to_vector ──

    #[test]
    fn test_without_overflow() {
        assert_eq!(OpCode::IntAddOvf.without_overflow(), Some(OpCode::IntAdd));
        assert_eq!(OpCode::IntSubOvf.without_overflow(), Some(OpCode::IntSub));
        assert_eq!(OpCode::IntMulOvf.without_overflow(), Some(OpCode::IntMul));
        assert_eq!(OpCode::IntAdd.without_overflow(), None);
    }

    #[test]
    fn test_to_vector() {
        assert_eq!(OpCode::IntAdd.to_vector(), Some(OpCode::VecIntAdd));
        assert_eq!(OpCode::FloatAdd.to_vector(), Some(OpCode::VecFloatAdd));
        assert_eq!(OpCode::GuardTrue.to_vector(), Some(OpCode::VecGuardTrue));
        assert_eq!(OpCode::SetfieldGc.to_vector(), None);
        // resoperation.py:1746-1759 `_opvector`: memory loads/stores.
        assert_eq!(OpCode::RawLoadI.to_vector(), Some(OpCode::VecLoadI));
        assert_eq!(OpCode::RawLoadF.to_vector(), Some(OpCode::VecLoadF));
        assert_eq!(OpCode::GetarrayitemRawI.to_vector(), Some(OpCode::VecLoadI));
        assert_eq!(
            OpCode::GetarrayitemGcPureF.to_vector(),
            Some(OpCode::VecLoadF)
        );
        assert_eq!(OpCode::RawStore.to_vector(), Some(OpCode::VecStore));
        assert_eq!(OpCode::SetarrayitemGc.to_vector(), Some(OpCode::VecStore));
        // `_R` (ref) array loads have no vector form upstream.
        assert_eq!(OpCode::GetarrayitemGcR.to_vector(), None);
    }

    // ── Name table ──

    #[test]
    fn test_opname_matches_debug_name() {
        for op in all_opcodes() {
            let name = op.name();
            let debug = format!("{:?}", op);
            assert_eq!(name, debug, "name() and Debug should match for {:?}", op);
        }
    }

    #[test]
    fn test_specific_opnames() {
        assert_eq!(OpCode::IntAdd.name(), "IntAdd");
        assert_eq!(OpCode::GuardTrue.name(), "GuardTrue");
        assert_eq!(OpCode::CallI.name(), "CallI");
        assert_eq!(OpCode::Jump.name(), "Jump");
        assert_eq!(OpCode::Finish.name(), "Finish");
        assert_eq!(OpCode::New.name(), "New");
        assert_eq!(OpCode::SetfieldGc.name(), "SetfieldGc");
    }

    // ── Op construction ──

    #[test]
    fn test_op_new() {
        // IntAdd takes two Int operands (resoperation.py:1693
        // `opclasses[INT_ADD].arity` = 2). The operands bind to synthetic
        // producers (kept alive by `_lp`/`_rp`) so `Op::new` carries
        // `Operand::Op`, not an unbound position-only box.
        let lhs_op = crate::forwarding::test_support::bound_resop_operand(Type::Int, 0);
        let rhs_op = crate::forwarding::test_support::bound_resop_operand(Type::Int, 1);
        let lhs = lhs_op.to_opref();
        let rhs = rhs_op.to_opref();
        let op = Op::new(OpCode::IntAdd, &[lhs_op.clone(), rhs_op.clone()]);
        assert_eq!(op.opcode, OpCode::IntAdd);
        assert_eq!(op.num_args(), 2);
        assert_eq!(op.arg(0).to_opref(), lhs);
        assert_eq!(op.arg(1).to_opref(), rhs);
        assert!(op.getdescr().is_none());
        assert!(op.getfailargs().is_none());
        assert_eq!(op.result_type(), Type::Int);
        assert_eq!(op.num_args(), 2);
    }

    #[test]
    fn test_op_getarg() {
        let lhs_op = crate::forwarding::test_support::bound_resop_operand(Type::Int, 10);
        let rhs_op = crate::forwarding::test_support::bound_resop_operand(Type::Int, 20);
        let lhs = lhs_op.to_opref();
        let rhs = rhs_op.to_opref();
        let op = Op::new(OpCode::IntAdd, &[lhs_op.clone(), rhs_op.clone()]);
        assert_eq!(op.arg(0).to_opref(), lhs);
        assert_eq!(op.arg(1).to_opref(), rhs);
    }

    // ── Descriptor requirements ──

    #[test]
    fn test_guards_have_descr() {
        for op in all_opcodes() {
            if op.is_guard() {
                assert!(op.has_descr(), "guard {:?} should have has_descr=true", op);
            }
        }
    }

    #[test]
    fn test_calls_have_descr() {
        // All call subcategories (plain calls, call_assembler, call_may_force,
        // call_pure, call_release_gil, call_loopinvariant, cond_call_value)
        // must have descriptors. Backend helpers like CheckMemoryError and
        // CallMallocNursery* are in the call range but don't need descriptors.
        for op in all_opcodes() {
            if op.is_plain_call()
                || op.is_call_assembler()
                || op.is_call_may_force()
                || op.is_call_pure()
                || op.is_call_release_gil()
                || op.is_call_loopinvariant()
                || op.is_cond_call_value()
            {
                assert!(op.has_descr(), "call {:?} should have has_descr=true", op);
            }
        }
    }

    // ── ovf alignment ──

    #[test]
    fn test_ovf_to_non_ovf_alignment() {
        let add_ovf_offset = OpCode::IntAddOvf as u16 - OVF_FIRST;
        let add_offset = OpCode::IntAdd as u16 - ALWAYS_PURE_FIRST;
        assert_eq!(add_ovf_offset, add_offset);

        let sub_ovf_offset = OpCode::IntSubOvf as u16 - OVF_FIRST;
        let sub_offset = OpCode::IntSub as u16 - ALWAYS_PURE_FIRST;
        assert_eq!(sub_ovf_offset, sub_offset);

        let mul_ovf_offset = OpCode::IntMulOvf as u16 - OVF_FIRST;
        let mul_offset = OpCode::IntMul as u16 - ALWAYS_PURE_FIRST;
        assert_eq!(mul_ovf_offset, mul_offset);
    }

    // ── is_getfield / is_getarrayitem / is_memory_access ──

    #[test]
    fn test_is_getfield() {
        assert!(OpCode::GetfieldGcI.is_getfield());
        assert!(OpCode::GetfieldGcR.is_getfield());
        assert!(OpCode::GetfieldGcF.is_getfield());
        assert!(OpCode::GetfieldGcPureI.is_getfield());
        assert!(OpCode::GetfieldGcPureR.is_getfield());
        assert!(OpCode::GetfieldGcPureF.is_getfield());
        assert!(!OpCode::GetfieldRawI.is_getfield());
        assert!(!OpCode::IntAdd.is_getfield());
    }

    #[test]
    fn test_is_getarrayitem() {
        assert!(OpCode::GetarrayitemGcI.is_getarrayitem());
        assert!(OpCode::GetarrayitemGcPureI.is_getarrayitem());
        assert!(!OpCode::IntAdd.is_getarrayitem());
    }

    #[test]
    fn test_memory_access_includes_fields_and_arrays() {
        let memory_ops = [
            OpCode::GetfieldGcI,
            OpCode::SetfieldGc,
            OpCode::GetarrayitemGcI,
            OpCode::SetarrayitemGc,
            OpCode::RawLoadI,
            OpCode::RawStore,
            OpCode::GcLoadI,
            OpCode::GcStore,
        ];
        for op in &memory_ops {
            assert!(op.is_memory_access(), "{:?} should be memory access", op);
        }
        assert!(!OpCode::IntAdd.is_memory_access());
        assert!(!OpCode::CallI.is_memory_access());
    }

    // ── can_raise ──

    #[test]
    fn test_can_raise() {
        assert!(OpCode::CallI.can_raise());
        assert!(OpCode::CallMayForceN.can_raise());
        assert!(OpCode::IntAddOvf.can_raise());
        assert!(OpCode::IntSubOvf.can_raise());
        assert!(OpCode::IntMulOvf.can_raise());
        assert!(!OpCode::IntAdd.can_raise());
        assert!(!OpCode::GuardTrue.can_raise());
        assert!(!OpCode::New.can_raise());
    }

    // ── is_label / is_jit_debug / is_malloc / is_vector_arithmetic ──

    #[test]
    fn test_is_label() {
        assert!(OpCode::Label.is_label());
        assert!(!OpCode::Jump.is_label());
    }

    #[test]
    fn test_is_jit_debug() {
        assert!(OpCode::DebugMergePoint.is_jit_debug());
        assert!(OpCode::EnterPortalFrame.is_jit_debug());
        assert!(OpCode::LeavePortalFrame.is_jit_debug());
        assert!(OpCode::JitDebug.is_jit_debug());
        assert!(!OpCode::IntAdd.is_jit_debug());
    }

    #[test]
    fn test_is_malloc() {
        let malloc_ops = [
            OpCode::New,
            OpCode::NewWithVtable,
            OpCode::NewArray,
            OpCode::NewArrayClear,
            OpCode::Newstr,
            OpCode::Newunicode,
        ];
        for op in &malloc_ops {
            assert!(op.is_malloc(), "{:?} should be malloc", op);
        }
        assert!(!OpCode::IntAdd.is_malloc());
        assert!(!OpCode::CallI.is_malloc());
    }

    #[test]
    fn test_is_vector_arithmetic() {
        let vec_arith = [
            OpCode::VecIntAdd,
            OpCode::VecIntSub,
            OpCode::VecIntMul,
            OpCode::VecFloatAdd,
            OpCode::VecFloatMul,
            OpCode::VecFloatNeg,
            OpCode::VecFloatAbs,
        ];
        for op in &vec_arith {
            assert!(
                op.is_vector_arithmetic(),
                "{:?} should be vec arithmetic",
                op
            );
        }
        assert!(!OpCode::IntAdd.is_vector_arithmetic());
    }

    // ── Consistency invariants ──

    #[test]
    fn test_guard_and_call_disjoint() {
        for op in all_opcodes() {
            assert!(
                !(op.is_guard() && op.is_call()),
                "{:?} is both guard and call",
                op
            );
        }
    }

    #[test]
    fn test_final_and_guard_disjoint() {
        for op in all_opcodes() {
            assert!(
                !(op.is_final() && op.is_guard()),
                "{:?} is both final and guard",
                op
            );
        }
    }

    #[test]
    fn test_guards_not_pure() {
        for op in all_opcodes() {
            if op.is_guard() {
                assert!(
                    !op.is_always_pure(),
                    "{:?} is a guard and should not be always_pure",
                    op
                );
            }
        }
    }

    // ── Logger parity tests (rpython/jit/metainterp/test/test_logger.py) ──

    #[test]
    fn test_format_trace_readable_output() {
        let ops = vec![
            op! {
                opcode: OpCode::IntAdd,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 1), crate::forwarding::test_support::bound_resop_operand(Type::Int, 2)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::int_op(3)),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::IntAdd,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 3), crate::forwarding::test_support::bound_resop_operand(Type::Int, 10_000)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::int_op(4)),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::Jump,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 4), crate::forwarding::test_support::bound_resop_operand(Type::Int, 3)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(None),


                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
        ];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 3);
        let output = format_trace(&ops, &constants);
        assert!(output.contains("v3 = IntAdd(v1, v2)"));
        assert!(output.contains("v4 = IntAdd(v3, 3)"));
        assert!(output.contains("Jump(v0, v4, v3)"));
    }

    #[test]
    fn test_op_display_int_result() {
        let op = op! {
            opcode: OpCode::IntAdd,
            args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 1), crate::forwarding::test_support::bound_resop_operand(Type::Int, 2)]),
            descr: std::cell::RefCell::new(None),
            pos: std::cell::Cell::new(OpRef::int_op(6)),
            fail_args: std::cell::RefCell::new(None),

            fail_arg_types: std::cell::RefCell::new(None),
            rd_resume_position: std::cell::Cell::new(-1),
            vecinfo: std::cell::RefCell::new(None),
        };
        let s = format!("{op}");
        assert_eq!(s, "v6 = IntAdd(v1, v2)");
    }

    #[test]
    fn test_op_display_void() {
        let op = op! {
            opcode: OpCode::SetfieldGc,
            args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 1)]),
            descr: std::cell::RefCell::new(None),
            pos: std::cell::Cell::new(OpRef::NONE),
            fail_args: std::cell::RefCell::new(None),

            fail_arg_types: std::cell::RefCell::new(None),
            rd_resume_position: std::cell::Cell::new(-1),
            vecinfo: std::cell::RefCell::new(None),
        };
        let s = format!("{op}");
        assert_eq!(s, "SetfieldGc(v0, v1)");
    }

    #[test]
    fn test_op_display_guard_with_fail_args() {
        let op = op! {
            opcode: OpCode::GuardTrue,
            args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0)]),
            descr: std::cell::RefCell::new(None),
            pos: std::cell::Cell::new(OpRef::NONE),
            fail_args: std::cell::RefCell::new(Some(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 1)])),


            fail_arg_types: std::cell::RefCell::new(None),
            rd_resume_position: std::cell::Cell::new(-1),
            vecinfo: std::cell::RefCell::new(None),
        };
        let s = format!("{op}");
        assert_eq!(s, "GuardTrue(v0) [v0, v1]");
    }

    #[test]
    fn test_op_display_guard_without_fail_args() {
        let op = op! {
            opcode: OpCode::GuardTrue,
            args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0)]),
            descr: std::cell::RefCell::new(None),
            pos: std::cell::Cell::new(OpRef::NONE),
            fail_args: std::cell::RefCell::new(None),

            fail_arg_types: std::cell::RefCell::new(None),
            rd_resume_position: std::cell::Cell::new(-1),
            vecinfo: std::cell::RefCell::new(None),
        };
        let s = format!("{op}");
        assert_eq!(s, "GuardTrue(v0)");
    }

    #[test]
    fn test_format_trace_constants_rendered_with_values() {
        let ops = vec![op! {
            opcode: OpCode::IntAdd,
            args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 10_000)]),
            descr: std::cell::RefCell::new(None),
            pos: std::cell::Cell::new(OpRef::int_op(1)),
            fail_args: std::cell::RefCell::new(None),

            fail_arg_types: std::cell::RefCell::new(None),
            rd_resume_position: std::cell::Cell::new(-1),
            vecinfo: std::cell::RefCell::new(None),
        }];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 42);
        let output = format_trace(&ops, &constants);
        assert!(output.contains("v1 = IntAdd(v0, 42)"));
        assert!(!output.contains("v10000"));
    }

    #[test]
    fn test_format_trace_guards_show_fail_args() {
        let ops = vec![
            op! {
                opcode: OpCode::IntAdd,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 10_000)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::int_op(1)),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::GuardTrue,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(Some(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 1)])),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::Finish,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 1)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(None),


                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
        ];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 1);
        let output = format_trace(&ops, &constants);
        assert!(output.contains("GuardTrue(v0) [v0, v1]"));
    }

    #[test]
    fn test_format_trace_constants_in_fail_args() {
        let ops = vec![op! {
            opcode: OpCode::GuardTrue,
            args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0)]),
            descr: std::cell::RefCell::new(None),
            pos: std::cell::Cell::new(OpRef::NONE),
            fail_args: std::cell::RefCell::new(Some(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 10_000)])),


            fail_arg_types: std::cell::RefCell::new(None),
            rd_resume_position: std::cell::Cell::new(-1),
            vecinfo: std::cell::RefCell::new(None),
        }];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 99);
        let output = format_trace(&ops, &constants);
        assert!(output.contains("GuardTrue(v0) [v0, 99]"));
    }

    #[test]
    fn test_format_trace_empty() {
        let ops: Vec<Op> = vec![];
        let constants: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
        let output = format_trace(&ops, &constants);
        assert!(output.is_empty());
    }

    // ── Extended logger parity tests (rpython/jit/metainterp/test/test_logger.py) ──

    #[test]
    fn test_format_trace_full_loop_label_to_jump() {
        // Parity with test_simple: a full loop trace from Label to Jump
        // should format each op on its own line with readable names and args.
        let ops = vec![
            op! {
                opcode: OpCode::Label,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 1), crate::forwarding::test_support::bound_resop_operand(Type::Int, 2)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::IntAdd,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 1), crate::forwarding::test_support::bound_resop_operand(Type::Int, 2)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::int_op(3)),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::IntAdd,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 3), crate::forwarding::test_support::bound_resop_operand(Type::Int, 10_000)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::int_op(4)),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::Jump,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 4), crate::forwarding::test_support::bound_resop_operand(Type::Int, 3)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(None),


                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
        ];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 3);
        let output = format_trace(&ops, &constants);
        // Label opens, Jump closes
        assert!(output.contains("Label(v0, v1, v2)"));
        assert!(output.contains("v3 = IntAdd(v1, v2)"));
        assert!(output.contains("v4 = IntAdd(v3, 3)"));
        assert!(output.contains("Jump(v0, v4, v3)"));
        // Each line is indented with 2 spaces
        for line in output.lines() {
            assert!(
                line.starts_with("  "),
                "each line should be indented: {line}"
            );
        }
    }

    #[test]
    fn test_format_trace_bridge_guard_to_finish() {
        // Parity with test_guard: a bridge trace starts with ops and ends with Finish.
        let ops = vec![
            op! {
                opcode: OpCode::IntSub,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 10_000)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::int_op(1)),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::IntGt,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 1), crate::forwarding::test_support::bound_resop_operand(Type::Int, 10_001)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::int_op(2)),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::GuardTrue,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 2)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(Some(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 1)])),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::Finish,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 1)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(None),


                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
        ];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 1);
        constants.insert(10_001, 0);
        let output = format_trace(&ops, &constants);
        assert!(output.contains("v1 = IntSub(v0, 1)"));
        assert!(output.contains("v2 = IntGt(v1, 0)"));
        assert!(output.contains("GuardTrue(v2) [v0, v1]"));
        assert!(output.contains("Finish(v1)"));
    }

    #[test]
    fn test_format_trace_descr_repr_in_output() {
        // Parity with test_descr: descriptors are rendered in the output
        // via repr_of_descr.
        use crate::descr::{DebugMergePointDescr, DebugMergePointInfo};
        let descr: crate::DescrRef = std::sync::Arc::new(DebugMergePointDescr::new(
            DebugMergePointInfo::new("testdriver", "bytecode ADD at 5", 5, 0),
        ));
        let ops = vec![op! {
            opcode: OpCode::DebugMergePoint,
            args: std::cell::RefCell::new(smallvec::smallvec![]),
            descr: std::cell::RefCell::new(Some(descr)),
            pos: std::cell::Cell::new(OpRef::NONE),
            fail_args: std::cell::RefCell::new(None),

            fail_arg_types: std::cell::RefCell::new(None),
            rd_resume_position: std::cell::Cell::new(-1),
            vecinfo: std::cell::RefCell::new(None),
        }];
        let constants: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
        let output = format_trace(&ops, &constants);
        assert!(
            output.contains("descr=<"),
            "output should contain 'descr=<': {output}"
        );
        assert!(
            output.contains("testdriver"),
            "descr repr should contain driver name: {output}"
        );
        assert!(
            output.contains("bytecode ADD at 5"),
            "descr repr should contain source repr: {output}"
        );
    }

    #[test]
    fn test_format_trace_complex_with_guards_and_constants() {
        // Parity with test_guard: complex trace with mixed ops, guards, constants,
        // and fail_args all render correctly and can be round-tripped.
        let ops = vec![
            op! {
                opcode: OpCode::Label,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 1)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::IntAdd,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 1)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::int_op(2)),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::IntLt,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 2), crate::forwarding::test_support::bound_resop_operand(Type::Int, 10_000)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::int_op(3)),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::GuardTrue,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 3)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(Some(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 2)])),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::IntSub,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 10_001)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::int_op(4)),
                fail_args: std::cell::RefCell::new(None),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::Jump,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 4), crate::forwarding::test_support::bound_resop_operand(Type::Int, 2)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(None),


                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
        ];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 100);
        constants.insert(10_001, 1);
        let output = format_trace(&ops, &constants);

        // Verify every op is present
        assert!(output.contains("Label(v0, v1)"));
        assert!(output.contains("v2 = IntAdd(v0, v1)"));
        assert!(output.contains("v3 = IntLt(v2, 100)"));
        assert!(output.contains("GuardTrue(v3) [v0, v2]"));
        assert!(output.contains("v4 = IntSub(v0, 1)"));
        assert!(output.contains("Jump(v4, v2)"));

        // Verify line count (6 ops = 6 lines)
        assert_eq!(output.lines().count(), 6);
    }

    #[test]
    fn test_format_trace_multiple_guards_with_different_fail_args() {
        // Multiple guards in a single trace, each with distinct fail_args.
        let ops = vec![
            op! {
                opcode: OpCode::GuardTrue,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(Some(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0)])),

                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
            op! {
                opcode: OpCode::GuardFalse,
                args: std::cell::RefCell::new(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 1)]),
                descr: std::cell::RefCell::new(None),
                pos: std::cell::Cell::new(OpRef::NONE),
                fail_args: std::cell::RefCell::new(Some(smallvec::smallvec![crate::forwarding::test_support::bound_resop_operand(Type::Int, 0), crate::forwarding::test_support::bound_resop_operand(Type::Int, 1), crate::forwarding::test_support::bound_resop_operand(Type::Int, 2)])),


                fail_arg_types: std::cell::RefCell::new(None),
                rd_resume_position: std::cell::Cell::new(-1),
                vecinfo: std::cell::RefCell::new(None),
            },
        ];
        let constants: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
        let output = format_trace(&ops, &constants);
        assert!(output.contains("GuardTrue(v0) [v0]"));
        assert!(output.contains("GuardFalse(v1) [v0, v1, v2]"));
    }

    #[test]
    fn test_is_setarrayitem() {
        assert!(OpCode::SetarrayitemGc.is_setarrayitem());
        assert!(OpCode::SetarrayitemRaw.is_setarrayitem());
        assert!(!OpCode::GetarrayitemGcI.is_setarrayitem());
        assert!(!OpCode::IntAdd.is_setarrayitem());
    }

    #[test]
    fn test_is_setfield() {
        assert!(OpCode::SetfieldGc.is_setfield());
        assert!(OpCode::SetfieldRaw.is_setfield());
        assert!(!OpCode::GetfieldGcI.is_setfield());
    }

    #[test]
    fn test_is_getinteriorfield() {
        assert!(OpCode::GetinteriorfieldGcI.is_getinteriorfield());
        assert!(OpCode::GetinteriorfieldGcR.is_getinteriorfield());
        assert!(OpCode::GetinteriorfieldGcF.is_getinteriorfield());
        assert!(!OpCode::SetinteriorfieldGc.is_getinteriorfield());
    }

    #[test]
    fn test_is_setinteriorfield() {
        assert!(OpCode::SetinteriorfieldGc.is_setinteriorfield());
        assert!(!OpCode::GetinteriorfieldGcI.is_setinteriorfield());
    }

    // ══════════════════════════════════════════════════════════════════
    // AbstractValue parity tests
    // Mirror rpython/jit/metainterp/test/test_history.py and
    // rpython/jit/metainterp/test/test_resoperation.py for the
    // AbstractValue / Const / InputArg / ResOp class hierarchy.
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_abstract_value_ty_const() {
        assert_eq!(AbstractValue::ConstInt(0).ty(), Some(Type::Int));
        assert_eq!(AbstractValue::ConstFloat(0).ty(), Some(Type::Float));
        assert_eq!(AbstractValue::ConstPtr(0).ty(), Some(Type::Ref));
    }

    #[test]
    fn test_abstract_value_ty_input_arg() {
        assert_eq!(AbstractValue::InputArgInt(0).ty(), Some(Type::Int));
        assert_eq!(AbstractValue::InputArgFloat(0).ty(), Some(Type::Float));
        assert_eq!(AbstractValue::InputArgRef(0).ty(), Some(Type::Ref));
    }

    #[test]
    fn test_abstract_value_ty_res_op() {
        assert_eq!(AbstractValue::IntOp(0).ty(), Some(Type::Int));
        assert_eq!(AbstractValue::FloatOp(0).ty(), Some(Type::Float));
        assert_eq!(AbstractValue::RefOp(0).ty(), Some(Type::Ref));
        assert_eq!(AbstractValue::VoidOp(0).ty(), Some(Type::Void));
    }

    #[test]
    fn test_abstract_value_ty_none() {
        assert_eq!(AbstractValue::None.ty(), None);
    }

    #[test]
    fn test_opref_ty_temp_var() {
        // `regalloc.py:18 TempVar(AbstractResOpOrInputArg)` has no
        // `.type` attribute; `_check_type` at `regalloc.py:405-407` skips
        // it via `isinstance(v, TempVar)`. `OpRef::ty()` must mirror by
        // returning `None` — projecting `Type::Int` would make a temp box
        // indistinguishable from an `IntOp` to anyone holding only the
        // OpRef.
        assert_eq!(OpRef::fresh_temp_var(0).ty(), None);
        assert_eq!(OpRef::fresh_temp_var(1).ty(), None);
    }

    #[test]
    fn test_abstract_value_is_constant() {
        assert!(AbstractValue::ConstInt(7).is_constant());
        assert!(AbstractValue::ConstFloat(7).is_constant());
        assert!(AbstractValue::ConstPtr(7).is_constant());
        assert!(!AbstractValue::InputArgInt(7).is_constant());
        assert!(!AbstractValue::IntOp(7).is_constant());
        assert!(!AbstractValue::None.is_constant());
    }

    #[test]
    fn test_abstract_value_is_input_arg() {
        assert!(AbstractValue::InputArgInt(7).is_input_arg());
        assert!(AbstractValue::InputArgFloat(7).is_input_arg());
        assert!(AbstractValue::InputArgRef(7).is_input_arg());
        assert!(!AbstractValue::ConstInt(7).is_input_arg());
        assert!(!AbstractValue::IntOp(7).is_input_arg());
        assert!(!AbstractValue::None.is_input_arg());
    }

    #[test]
    fn test_abstract_value_is_res_op() {
        assert!(AbstractValue::IntOp(7).is_res_op());
        assert!(AbstractValue::FloatOp(7).is_res_op());
        assert!(AbstractValue::RefOp(7).is_res_op());
        assert!(!AbstractValue::ConstInt(7).is_res_op());
        assert!(!AbstractValue::InputArgInt(7).is_res_op());
        assert!(!AbstractValue::None.is_res_op());
    }

    #[test]
    fn test_abstract_value_raw() {
        assert_eq!(AbstractValue::ConstInt(11).raw(), Some(11));
        assert_eq!(AbstractValue::InputArgRef(22).raw(), Some(22));
        assert_eq!(AbstractValue::IntOp(33).raw(), Some(33));
        assert_eq!(AbstractValue::None.raw(), None);
    }

    #[test]
    fn test_abstract_value_disjoint_categories() {
        // RPython parity: Const / AbstractInputArg / AbstractResOp
        // are disjoint sub-hierarchies under AbstractValue.
        let const_iv = AbstractValue::ConstInt(0);
        let input_iv = AbstractValue::InputArgInt(0);
        let res_iv = AbstractValue::IntOp(0);
        assert!(const_iv.is_constant() && !const_iv.is_input_arg() && !const_iv.is_res_op());
        assert!(!input_iv.is_constant() && input_iv.is_input_arg() && !input_iv.is_res_op());
        assert!(!res_iv.is_constant() && !res_iv.is_input_arg() && res_iv.is_res_op());
    }

    // ── Typed OpRef constructors (Phase 2A) ──

    #[test]
    fn inline_const_constructors_keep_variant_distinct() {
        // history.py:244 `ConstInt.same_constant` rejects `ConstFloat` /
        // `ConstPtr` — Const sub-classes are disjoint identities.
        for v in [0i64, 1, 7, 100] {
            assert_ne!(OpRef::const_int(v), OpRef::const_float(v as f64));
            assert_ne!(OpRef::const_int(v), OpRef::const_ptr(GcRef(v as usize)));
            assert_ne!(
                OpRef::const_float(v as f64),
                OpRef::const_ptr(GcRef(v as usize))
            );
            assert_eq!(OpRef::const_int(v), OpRef::const_int(v));
        }
    }

    #[test]
    fn typed_input_arg_constructors_keep_variant_distinct() {
        // resoperation.py:719/727/739 `InputArg{Int,Float,Ref}` are
        // disjoint Box classes; the enum discriminant rejects
        // cross-variant identity even at matching raw payloads.
        for pos in [0u32, 1, 7, 100] {
            assert_ne!(OpRef::input_arg_int(pos), OpRef::input_arg_float(pos));
            assert_ne!(OpRef::input_arg_int(pos), OpRef::input_arg_ref(pos));
            assert_ne!(OpRef::input_arg_float(pos), OpRef::input_arg_ref(pos));
            assert_eq!(OpRef::input_arg_int(pos), OpRef::input_arg_int(pos));
        }
    }

    #[test]
    fn typed_op_result_constructors_keep_variant_distinct() {
        // resoperation.py:564-638 `IntOp` / `FloatOp` / `RefOp` mixins:
        // each ResOp's `.type` is fixed by the mixin class.
        for pos in [0u32, 1, 7, 100, 1_000_000] {
            assert_ne!(OpRef::int_op(pos), OpRef::float_op(pos));
            assert_ne!(OpRef::int_op(pos), OpRef::ref_op(pos));
            assert_ne!(OpRef::float_op(pos), OpRef::ref_op(pos));
            assert_eq!(OpRef::int_op(pos), OpRef::int_op(pos));
        }
    }

    #[test]
    fn test_typed_constructors_classification() {
        // is_constant() distinguishes Const family from the rest.
        assert!(OpRef::const_int(0).is_constant());
        assert!(OpRef::const_float(0.0).is_constant());
        assert!(OpRef::const_ptr(GcRef(0)).is_constant());
        assert!(!OpRef::input_arg_int(0).is_constant());
        assert!(!OpRef::input_arg_float(0).is_constant());
        assert!(!OpRef::input_arg_ref(0).is_constant());
        assert!(!OpRef::int_op(0).is_constant());
        assert!(!OpRef::float_op(0).is_constant());
        assert!(!OpRef::ref_op(0).is_constant());
    }

    /// Pins the raw bit-helpers used by callers that hold a raw u32 from
    /// an index-keyed constant pool: the high bit marks the constant
    /// namespace and `raw_const_index` strips it.
    #[test]
    fn test_raw_is_constant_matches_opref_path() {
        // High bit set ↔ constant-namespace pool key (see `CONST_BIT`).
        const CONST_BIT: u32 = 1 << 31;
        for idx in [0u32, 1, 7, 100, 0x0FFF_FFFF] {
            let raw = idx | CONST_BIT;
            assert!(OpRef::raw_is_constant(raw));
            assert_eq!(OpRef::raw_const_index(raw), idx);
        }
        // Non-constant raw values: op positions, inputarg positions, plain numbers.
        for raw in [0u32, 1, 7, 100, 0x0FFF_FFFF] {
            assert!(!OpRef::raw_is_constant(raw));
        }
        // Sentinel range stays out of the constant namespace.
        assert!(!OpRef::raw_is_constant(u32::MAX));
        assert!(!OpRef::raw_is_constant(u32::MAX - 1));
    }
}
