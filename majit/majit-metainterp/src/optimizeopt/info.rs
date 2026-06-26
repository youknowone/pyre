use crate::optimizeopt::intutils::{IntBound, IntBoundMakeGuards};
pub use crate::optimizeopt::rawbuffer::{
    InvalidRawOperation, InvalidRawRead, InvalidRawWrite, RawBuffer,
};
/// Abstract information attached to operations during optimization.
///
/// Translated from rpython/jit/metainterp/optimizeopt/info.py.
/// Each operation can have associated analysis info (e.g., known integer bounds,
/// pointer info, virtual object state).
use majit_ir::operand::Operand;
use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRef, Type, Value};

fn lookup_field_descr(field_descrs: &[DescrRef], field_idx: u32) -> Option<DescrRef> {
    field_descrs.get(field_idx as usize).cloned()
}

pub use majit_ir::field_entry::{FieldEntry, PreambleOp};
pub use majit_ir::op_info::{EmptyInfo, FloatConstInfo, OpInfo};
pub use majit_ir::ptr_info::reasonable_array_index;
pub use majit_ir::ptr_info::{PtrInfo, StrPtrInfo};

pub use majit_ir::ptr_info::{
    StringConstantAllocator, StringContentResolver, StringLengthResolver,
};

/// Result of `OptContext::ensure_ptr_info_arg0(op)` — direct line-by-line
/// equivalent of PyPy's `ensure_ptr_info_arg0` return value
/// (`optimizer.py:461-499`).
///
/// PyPy returns a Python `PtrInfo` object that the caller invokes methods on
/// (`structinfo.setfield(...)`, `arrayinfo.getlenbound(None).make_gt_const(...)`).
/// The Rust port can't expose `&mut PtrInfo` directly when the arg0 is a
/// constant — there's no `Forwarded::Info` slot to borrow from — so the enum
/// distinguishes the two cases:
///
/// - **`Constant { gcref, .. }`** — `arg0.is_constant()`
///   (`optimizer.py:464-466`). PyPy returns a freshly-constructed
///   `info.ConstPtrInfo(arg0)`. The Rust variant carries the resolved
///   `GcRef` so methods like `getlenbound` can synthesize the same answer
///   on demand. The optional `string_length_resolver` Arc allows
///   `getlenbound(Some(mode))` to return an exact constant length when the
///   runtime can read the underlying string object — matching PyPy's
///   `getstrlen1(mode)` path through `_unpack_str(mode)`.
///
/// - **`Forwarded(&mut PtrInfo)`** — `arg0.get_forwarded()` returns either an
///   existing `AbstractVirtualPtrInfo` subclass (early-return path) or a
///   freshly-installed Instance/Struct/Array/Str etc. (`optimizer.py:475-498`).
///   The mutable reference is backed by the bound `Op`/`InputArg`'s
///   `_forwarded` slot, so `info.setfield()` / `info.setitem()` mutate the
///   canonical PtrInfo in-place — matching PyPy's
///   `arg0.set_forwarded(opinfo)` followed by `opinfo.setfield(...)`.
pub enum EnsuredPtrInfo {
    /// `info.ConstPtrInfo(arg0)` — synthesized from a constant Ref / raw-pointer
    /// Int OpRef. Read-only by construction.
    Constant {
        gcref: GcRef,
        /// Optional runtime hook for `getstrlen1(mode)` lookups.
        string_length_resolver: Option<StringLengthResolver>,
    },
    /// `arg0.get_forwarded()` — BoxRef-routed mutable handle. Each
    /// `as_mut()` call re-borrows the inner `RefCell`. Produced when the
    /// opref resolves to a bound `Op`/`InputArg`.
    ForwardedBox(majit_ir::box_ref::BoxRef),
}

impl EnsuredPtrInfo {
    /// `info.py PtrInfo.getlenbound(mode)` — direct delegation to the underlying
    /// PtrInfo. For `Constant` the call routes through the optional
    /// `string_length_resolver` so an exact constant length can be returned
    /// when the runtime knows it (PyPy `ConstPtrInfo.getlenbound` →
    /// `getstrlen1(mode)` → `_unpack_str(mode)` at info.py:796-822).
    pub fn getlenbound(&mut self, mode: Option<u8>) -> Option<IntBound> {
        match self {
            EnsuredPtrInfo::Constant {
                gcref,
                string_length_resolver,
            } => {
                // info.py:796-802 ConstPtrInfo.getlenbound(mode):
                //
                //     def getlenbound(self, mode):
                //         length = self.getstrlen1(mode)
                //         if length < 0:
                //             return IntBound.nonnegative()
                //         return IntBound.from_constant(length)
                //
                // info.py:810-824 ConstPtrInfo.getstrlen1(mode):
                //
                //     def getstrlen1(self, mode):
                //         if mode is vstring.mode_string:    ...
                //         elif mode is vstring.mode_unicode: ...
                //         else:
                //             return -1
                //
                // PyPy returns `IntBound.nonnegative()` regardless of
                // mode whenever `getstrlen1` cannot supply an exact
                // length. The Rust port mirrors that:
                //   * mode == None        → getstrlen1 returns -1 →
                //                           nonnegative()
                //   * mode == Some(0|1)   → resolver returns Some(len) →
                //                           from_constant(len);
                //                           else nonnegative()
                let length = match mode {
                    Some(mode_value) => {
                        if gcref.is_null() {
                            -1
                        } else if let Some(resolver) = string_length_resolver.as_deref() {
                            resolver(*gcref, mode_value).unwrap_or(-1)
                        } else {
                            -1
                        }
                    }
                    // info.py:823-824 `else: return -1` for mode == None.
                    None => -1,
                };
                if length < 0 {
                    Some(IntBound::nonnegative())
                } else {
                    Some(IntBound::from_constant(length))
                }
            }
            EnsuredPtrInfo::ForwardedBox(bx) => {
                bx.ptr_info_mut().and_then(|mut p| p.getlenbound(mode))
            }
        }
    }

    /// Mutable access to the underlying `PtrInfo`. Returns `None` for the
    /// `Constant` variant — PyPy's `ConstPtrInfo.setfield/setitem` route
    /// through `optheap.const_infos`, not through the constant box's own
    /// info slot (info.py:738-752). The `ForwardedBox` variant returns
    /// `None` if the BoxRef's `_forwarded` slot does not currently hold
    /// `Forwarded::Info(OpInfo::Ptr(_))`. The returned guard owns an `Rc`
    /// clone of the live `Rc<RefCell<PtrInfo>>` cell and an exclusive
    /// `RefCell` borrow — drop it before any sibling write to the same
    /// box's `_forwarded` slot.
    pub fn as_mut(&mut self) -> Option<majit_ir::box_ref::PtrInfoBorrowMut> {
        match self {
            EnsuredPtrInfo::Constant { .. } => None,
            EnsuredPtrInfo::ForwardedBox(bx) => bx.ptr_info_mut(),
        }
    }

    /// Whether the helper produced a synthesized `ConstPtrInfo` rather than a
    /// real forwarded entry. Mirrors `isinstance(opinfo, ConstPtrInfo)` at
    /// the call site.
    pub fn is_constant(&self) -> bool {
        matches!(self, EnsuredPtrInfo::Constant { .. })
    }
}

pub use majit_ir::ptr_info::{
    VStringConcatInfo, VStringPlainInfo, VStringSliceInfo, VStringVariant,
};

/// Extension trait carrying the OptContext-coupled methods that used
/// to live on `impl StrPtrInfo` in metainterp. The data type itself
/// is in `majit-ir`; only methods that depend on metainterp-side
/// helpers (`get_constant_int_or_bound`, `get_box_replacement_box`,
/// `getptrinfo`) stay here as a trait.
pub trait StrPtrInfoExt {
    fn getstrlen(&self, ctx: &crate::optimizeopt::OptContext, mode: u8) -> Option<i64>;
    fn get_constant_string_spec(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<Vec<i64>>;
    fn strgetitem(&self, index: i64, ctx: &crate::optimizeopt::OptContext) -> Option<OpRef>;
}

impl StrPtrInfoExt for StrPtrInfo {
    /// vstring.py:110/171/251/281 `getstrlen()` on the string ptrinfo classes.
    ///
    /// Returns the structurally-known constant length for virtual variants
    /// (Plain/Slice/Concat). For the non-virtual `Ptr` variant, returns
    /// `None` — RPython's base StrPtrInfo.getstrlen() (vstring.py:110-119)
    /// always emits STRLEN and attaches lenbound as metadata; it never
    /// extracts a constant from lenbound directly.
    fn getstrlen(&self, ctx: &crate::optimizeopt::OptContext, mode: u8) -> Option<i64> {
        // vstring.py:112: if self.lgtop is not None: return self.lgtop — the
        // length BOX. Callers fold only when it `isinstance(ConstInt)`, so a
        // length variable whose IntBound merely happens to be constant is NOT
        // a known constant length: read only an actual ConstInt, not the bound.
        if let Some(lgtop) = self.lgtop.as_ref() {
            return ctx
                .resolve_operand_operand_opt(lgtop)
                .and_then(|b| ctx.get_constant_int_box(&b));
        }
        match &self.variant {
            // vstring.py:110-119: base StrPtrInfo.getstrlen always emits
            // STRLEN and caches in lgtop; never returns a constant from
            // lenbound. The caller (getstrlen_opref) handles STRLEN emission.
            VStringVariant::Ptr => None,
            // vstring.py:171-175: VStringPlainInfo.getstrlen
            VStringVariant::Plain(info) => Some(info._chars.len() as i64),
            // vstring.py:251-253: VStringSliceInfo.getstrlen → self.lgtop;
            // constant only when it is an actual ConstInt (isinstance check).
            VStringVariant::Slice(info) => {
                let b = ctx.resolve_operand_operand_opt(&info.lgtop)?;
                ctx.get_constant_int_box(&b)
            }
            // vstring.py:281-295: VStringConcatInfo.getstrlen
            VStringVariant::Concat(info) => {
                let vleft_box = ctx.resolve_operand_operand_opt(&info.vleft);
                let vright_box = ctx.resolve_operand_operand_opt(&info.vright);
                let left = vleft_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let right = vright_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let len1 = left.get_known_str_length(ctx, mode)?;
                let len2 = right.get_known_str_length(ctx, mode)?;
                Some(len1 + len2)
            }
        }
    }

    /// vstring.py:161 / 172 / 298 `get_constant_string_spec()` on the string
    /// ptrinfo classes.
    ///
    /// The upstream method returns either a low-level string or unicode object.
    /// majit keeps the same recursive shape but represents the constant string
    /// as character/codepoint integers until a runtime string allocator is
    /// wired in.
    fn get_constant_string_spec(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<Vec<i64>> {
        let _ = mode;
        match &self.variant {
            VStringVariant::Ptr => None,
            VStringVariant::Plain(info) => {
                let mut chars = Vec::with_capacity(info._chars.len());
                for ch in &info._chars {
                    let ch_box = ch.as_ref()?;
                    // vstring.py:179-180: `if c is None or not c.is_constant()`.
                    // RPython stores the resolved char box in `_chars`, so a
                    // bare `c.is_constant()` suffices. Pyre stores the operand's
                    // position box (which forwards to its Const), so resolving
                    // the chain first is the faithful equivalent of RPython's
                    // direct check. `const_int()` (not `get_constant_int_box`)
                    // keeps parity strict: a Const terminal folds, a box with
                    // only a constant IntBound does NOT (no synthesized
                    // ConstInt).
                    chars.push(
                        ctx.resolve_operand_operand_opt(ch_box)
                            .and_then(|cb| cb.const_int())?,
                    );
                }
                Some(chars)
            }
            VStringVariant::Slice(info) => {
                // vstring.py:236-248: use getintbound().is_constant()
                let s_box = ctx.resolve_operand_operand_opt(&info.s);
                let source = s_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let source_chars = source.get_constant_string_spec(ctx, mode)?;
                let start_box = ctx.resolve_operand_operand_opt(&info.start)?;
                let start = usize::try_from(ctx.get_constant_int_or_bound_box(&start_box)?).ok()?;
                let lgtop_box = ctx.resolve_operand_operand_opt(&info.lgtop)?;
                let length =
                    usize::try_from(ctx.get_constant_int_or_bound_box(&lgtop_box)?).ok()?;
                let stop = start.checked_add(length)?;
                if stop > source_chars.len() {
                    return None;
                }
                Some(source_chars[start..stop].to_vec())
            }
            VStringVariant::Concat(info) => {
                let vleft_box = ctx.resolve_operand_operand_opt(&info.vleft);
                let vright_box = ctx.resolve_operand_operand_opt(&info.vright);
                let left = vleft_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let right = vright_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let mut chars = left.get_constant_string_spec(ctx, mode)?;
                chars.extend(right.get_constant_string_spec(ctx, mode)?);
                Some(chars)
            }
        }
    }

    /// vstring.py:158 / 172 / 230 `strgetitem()` shape, collapsed into a single
    /// variant-dispatch method on the Rust side.
    fn strgetitem(&self, index: i64, ctx: &crate::optimizeopt::OptContext) -> Option<OpRef> {
        let index = usize::try_from(index).ok()?;
        match &self.variant {
            VStringVariant::Ptr => None,
            VStringVariant::Plain(info) => info
                ._chars
                .get(index)
                .and_then(|o| o.as_ref())
                .map(|b| b.to_opref()),
            VStringVariant::Slice(info) => {
                // vstring.py:490-497: index = _int_add(sinfo.start, index), then
                // the fold proceeds on `getintbound(index).is_constant()` — the
                // IntBound of the COMPUTED index, not `isinstance(index, ConstInt)`.
                // So a start whose IntBound is constant (even if the box is not a
                // literal ConstInt) still yields a constant index and folds. Read
                // the start via the intbound-aware accessor, not ConstInt-only.
                let start_box = ctx.resolve_operand_operand_opt(&info.start)?;
                let start = ctx.get_constant_int_or_bound_box(&start_box)?;
                let s_box = ctx.resolve_operand_operand_opt(&info.s);
                let source = s_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                source.strgetitem(index as i64 + start, ctx)
            }
            VStringVariant::Concat(info) => {
                let vleft_box = ctx.resolve_operand_operand_opt(&info.vleft);
                let left = vleft_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let left_len = usize::try_from(left.get_known_str_length(ctx, self.mode)?).ok()?;
                if index < left_len {
                    left.strgetitem(index as i64, ctx)
                } else {
                    let vright_box = ctx.resolve_operand_operand_opt(&info.vright);
                    let right = vright_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                    right.strgetitem((index - left_len) as i64, ctx)
                }
            }
        }
    }
}

/// Extension trait carrying the `OptContext` / `majit_gc` /
/// `VirtualVisitor`-coupled methods that used to live on `impl PtrInfo`.
/// The data type and pure-leaf methods live in `majit-ir::ptr_info`; only
/// methods that depend on metainterp-side helpers stay here as a trait.
pub trait PtrInfoExt {
    /// info.py:763-772 `ConstPtrInfo.get_known_class(cpu)` + the other
    /// PtrInfo subclasses' `_known_class` accessors. Probes
    /// `majit_gc::supports_guard_gc_type` / `majit_gc::check_is_object`
    /// when discriminating constant pointers. The `cpu` argument routes
    /// `cls_of_box` through the `Cpu` trait so backends overriding the
    /// typeptr-at-offset-0 read (`gcremovetypeptr`) are honored.
    fn get_known_class(&self, cpu: &dyn crate::cpu::Cpu) -> Option<i64>;

    /// info.py:83 `make_guards(op, short, optimizer)`.
    fn make_guards(&self, op: OpRef, short: &mut Vec<Op>, ctx: &mut crate::optimizeopt::OptContext);

    /// info.py:74-75 / vstring.py:103-105 / 249-258 — common string-length
    /// query across `ConstPtrInfo` and `StrPtrInfo`.
    fn get_known_str_length(&self, ctx: &crate::optimizeopt::OptContext, mode: u8) -> Option<i64>;

    /// info.py:793 ConstPtrInfo.get_constant_string_spec and
    /// vstring.py:178 / 236 / 298 — recursive constant string extraction.
    fn get_constant_string_spec(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<Vec<i64>>;

    /// vstring.py:172 / 230 `strgetitem()` on string ptrinfo —
    /// virtual dispatch only.
    fn strgetitem(&self, index: i64, ctx: &crate::optimizeopt::OptContext) -> Option<OpRef>;

    /// info.py:331 / 369 / 376 / 445 / 485 / 598 / 701 +
    /// vstring.py:211 / 263 / 333 `visitor_dispatch_virtual_type`.
    fn visitor_dispatch_virtual_type<V: crate::walkvirtual::VirtualVisitor>(
        &self,
        visitor: &mut V,
    ) -> Option<V::VInfo>;

    /// info.py:137-160 / 222-226: force_box() emits the allocation and
    /// field writes via emit_extra(), recursively forcing child virtuals.
    /// `op` is the (bound) operand of the virtual being forced — RPython
    /// `force_box(self, op, optforce)` passes the op directly
    /// (info.py:148-152), so the make_equal_to / set_forwarded receiver
    /// needs no lookup.
    fn force_box(&mut self, op: &Operand, ctx: &mut crate::optimizeopt::OptContext) -> OpRef;

    /// info.py:273-303: `_is_immutable_and_filled_with_constants`
    /// — used by `force_box` to decide whether a virtual can be
    /// constant-folded.
    fn is_immutable_and_filled_with_constants(&self, ctx: &crate::optimizeopt::OptContext) -> bool;
}

impl PtrInfoExt for PtrInfo {
    /// info.py:763-772 `ConstPtrInfo.get_known_class(cpu)` +
    /// the other PtrInfo subclasses' `_known_class` accessors:
    ///
    /// ```text
    /// def get_known_class(self, cpu):
    ///     if not self._const.nonnull():
    ///         return None
    ///     if cpu.supports_guard_gc_type:
    ///         if not cpu.check_is_object(self._const.getref_base()):
    ///             return None
    ///     return cpu.cls_of_box(self._const)
    /// ```
    ///
    /// - `Instance`/`Virtual`: return the stored `known_class` field
    ///   (PyPy `InstancePtrInfo._known_class`). A class-only result of
    ///   `make_constant_class` is also stored as `Instance(descr=None,
    ///   known_class=Some(...))`.
    /// - `Constant`: null constants → `None`; otherwise, when the
    ///   backend supports `guard_gc_type` (`majit_gc::supports_guard_gc_type`),
    ///   gate `cls_of_box` on `majit_gc::check_is_object` so that
    ///   non-object constant pointers are rejected and the optimizer
    ///   does not read garbage at offset 0. When the backend does
    ///   not support `guard_gc_type`, RPython skips the
    ///   `check_is_object` call entirely and still returns
    ///   `cls_of_box(self._const)`; this port follows that.
    /// - Everything else: `None`.
    fn get_known_class(&self, cpu: &dyn crate::cpu::Cpu) -> Option<i64> {
        match self {
            PtrInfo::Instance(v) => v.known_class,
            PtrInfo::Virtual(v) => v.known_class,
            PtrInfo::Constant(gcref) => {
                // info.py:764: `if not self._const.nonnull(): return None`
                if gcref.is_null() {
                    return None;
                }
                // info.py:765-767: gate the `check_is_object` call on
                // `supports_guard_gc_type`. When the backend doesn't
                // support guard_gc_type, RPython simply skips the
                // `check_is_object` step and still calls `cls_of_box`.
                if majit_gc::supports_guard_gc_type() && !majit_gc::check_is_object(*gcref) {
                    return None;
                }
                // info.py:768 `return cpu.cls_of_box(self._const)` —
                // routes through the Cpu trait so backends that override
                // the typeptr-at-offset-0 read (e.g. `gcremovetypeptr`)
                // are honored here too.
                let vtable = cpu.cls_of_gcref(*gcref);
                if vtable == 0 { None } else { Some(vtable) }
            }
            _ => None,
        }
    }

    /// info.py:83: make_guards(op, short, optimizer)
    /// info.py: make_guards(self, op, short, optimizer)
    ///
    /// Append guard operations to `short` that check this PtrInfo's
    /// properties hold for `op`. Used by use_box (shortpreamble.py:382).
    /// `ctx` plays the role of `optimizer` in the upstream signature:
    /// constant-pool allocation goes through `reserve_const_ref` +
    /// `seed_constant`, and producer-result identity through
    /// `alloc_op_position_typed`.
    fn make_guards(
        &self,
        op: OpRef,
        short: &mut Vec<Op>,
        ctx: &mut crate::optimizeopt::OptContext,
    ) {
        let mut alloc_const = |ctx: &mut crate::optimizeopt::OptContext, value: Value| {
            // history.py:227/268/314 Const{Int,Float,Ptr}.value inline.
            let pos = match value {
                Value::Int(v) => OpRef::const_int(v),
                Value::Float(v) => OpRef::const_float(v),
                Value::Ref(v) => OpRef::const_ptr(v),
                Value::Void => panic!("alloc_const: ConstVoid not allowed"),
            };
            // ConstInt/Float/Ptr value rides inline on `pos`
            // (history.py:227/268/314); no `seed_constant` (const arm no-op).
            ctx.materialize_operand_at(pos)
        };
        // info.py make_guards receives `op` as a Box object; bind the
        // caller-resolved producer once. Guard args referencing ops pushed
        // into `short` itself (lenop / eq_op below) stay position-only —
        // their producer lives in `short`, not in ctx's registries.
        let op_b = ctx.materialize_operand_at(op);
        match self {
            // info.py:83-84: PtrInfo base — no-op
            PtrInfo::NonNull { .. } => {
                // info.py:120-122: NonNullPtrInfo.make_guards
                short.push(Op::new(OpCode::GuardNonnull, &[op_b.clone()]));
            }
            PtrInfo::Instance(info) => {
                // info.py:336-353 InstancePtrInfo.make_guards line-by-line.
                //
                //   def make_guards(self, op, short, optimizer):
                //       if self._known_class is not None:
                //           if not optimizer.cpu.remove_gctypeptr:
                //               short.append(GUARD_NONNULL[op])
                //               short.append(GUARD_IS_OBJECT[op])
                //               short.append(GUARD_CLASS[op, self._known_class])
                //           else:
                //               short.append(GUARD_NONNULL_CLASS[op, self._known_class])
                //       elif self.descr is not None:
                //           short.append(GUARD_NONNULL[op])
                //           if not optimizer.cpu.remove_gctypeptr:
                //               short.append(GUARD_IS_OBJECT[op])
                //           short.append(GUARD_SUBCLASS[op, ConstInt(descr.get_vtable())])
                //       else:
                //           AbstractStructPtrInfo.make_guards(...)
                //
                // `ctx.remove_gctypeptr` is the
                // `optimizer.cpu.remove_gctypeptr` analogue (llmodel.py:55
                // — translator config `gcremovetypeptr`). Pyre defaults
                // to True because its PyObject layout has static
                // singletons (INSTANCE_TYPE, INT_TYPE, …) with no GC
                // header, and the False-branch GUARD_IS_OBJECT reads
                // `obj - GcHeader::SIZE` (codegen.rs:797-802) which
                // SIGBUSes on those statics. The False branch is still
                // emitted line-by-line so a backend that flips
                // `remove_gctypeptr=false` (e.g. a future heap-only
                // PyObject layout) gets the upstream guard sequence
                // without further changes.
                if let Some(cls) = info.known_class {
                    // info.py:341-345 stores `_known_class` on PtrInfo, but
                    // the emitted guard operand is the same ConstInt vtable
                    // address produced by backend/model.py:199-201
                    // `cls_of_box()`.
                    let class_ref = alloc_const(ctx, Value::Int(cls));
                    if !ctx.remove_gctypeptr {
                        short.push(Op::new(OpCode::GuardNonnull, &[op_b.clone()]));
                        short.push(Op::new(OpCode::GuardIsObject, &[op_b.clone()]));
                        short.push(Op::new(
                            OpCode::GuardClass,
                            &[op_b.clone(), class_ref.clone()],
                        ));
                    } else {
                        short.push(Op::new(
                            OpCode::GuardNonnullClass,
                            &[op_b.clone(), class_ref.clone()],
                        ));
                    }
                } else if let Some(descr) = &info.descr {
                    let vtable = descr
                        .as_size_descr()
                        .map(|sd| sd.vtable() as i64)
                        .unwrap_or(0);
                    let vtable_const = alloc_const(ctx, Value::Int(vtable));
                    short.push(Op::new(OpCode::GuardNonnull, &[op_b.clone()]));
                    if !ctx.remove_gctypeptr {
                        short.push(Op::new(OpCode::GuardIsObject, &[op_b.clone()]));
                    }
                    short.push(Op::new(
                        OpCode::GuardSubclass,
                        &[op_b.clone(), vtable_const.clone()],
                    ));
                } else {
                    // info.py:353 fall-through with neither class nor
                    // descr — base NonNullPtrInfo.make_guards.
                    short.push(Op::new(OpCode::GuardNonnull, &[op_b.clone()]));
                }
            }
            PtrInfo::Struct(info) => {
                // info.py:360-366: StructPtrInfo.make_guards.
                //   if self.descr is not None:
                //       c_typeid = ConstInt(self.descr.get_type_id())
                //       short.extend([GUARD_NONNULL[op],
                //                     GUARD_GC_TYPE[op, c_typeid]])
                short.push(Op::new(OpCode::GuardNonnull, &[op_b.clone()]));
                // GUARD_GC_TYPE only for a real `GcStruct`: it reads a
                // type-id word at `ref - 8` (the GC header).  A header-less
                // raw native struct (registered via `register_struct_layout`
                // — e.g. a `ref(T)` state scalar's target, or a `pools[i]`
                // element loaded by `getarrayitem_gc_r`) has no such word,
                // so the guard would read content-dependent garbage that
                // mutates on every push/pop and fail on every loop
                // re-entry.  RPython emits GUARD_GC_TYPE only for
                // `lltype.GcStruct`; a raw `Struct` gets GUARD_NONNULL only.
                // No descr → preserve the guard (default true).
                let is_gc_managed = info
                    .descr
                    .as_size_descr()
                    .map(|sd| sd.is_gc_managed())
                    .unwrap_or(true);
                if is_gc_managed {
                    let type_id = info
                        .descr
                        .as_size_descr()
                        .map(|sd| sd.type_id() as i64)
                        .unwrap_or(0);
                    let type_id_const = alloc_const(ctx, Value::Int(type_id));
                    short.push(Op::new(
                        OpCode::GuardGcType,
                        &[op_b.clone(), type_id_const.clone()],
                    ));
                }
            }
            PtrInfo::Constant(gcref) => {
                // info.py:715-716: ConstPtrInfo.make_guards
                let c = alloc_const(ctx, Value::Ref(*gcref));
                short.push(Op::new(OpCode::GuardValue, &[op_b.clone(), c.clone()]));
            }
            PtrInfo::Array(info) => {
                // info.py:632-639: ArrayPtrInfo.make_guards.
                //   AbstractVirtualPtrInfo.make_guards → NonNullPtrInfo.make_guards
                //   short.append(GUARD_GC_TYPE[op, ConstInt(descr.get_type_id())])
                //   if self.lenbound is not None:
                //       lenop = ARRAYLEN_GC[op] (descr=self.descr)
                //       short.append(lenop)
                //       self.lenbound.make_guards(lenop, short, optimizer)
                short.push(Op::new(OpCode::GuardNonnull, &[op_b.clone()]));
                // GUARD_GC_TYPE reads a type-id word at `ptr -
                // GcHeader::SIZE`; a header-less raw native pointer-array
                // (a `pool_arrays` base minted by `add_ptr_array_descr`,
                // `is_gc_managed = false`) has no such word, so the guard
                // would read content-dependent memory before the array
                // and fail on every short-preamble re-entry.  Gate it the
                // same way the `PtrInfo::Struct` arm gates a raw struct.
                // No descr → preserve the guard (default true).
                let is_gc_managed = info
                    .descr
                    .as_array_descr()
                    .map(|ad| ad.is_gc_managed())
                    .unwrap_or(true);
                if is_gc_managed {
                    let type_id = info
                        .descr
                        .as_array_descr()
                        .map(|ad| ad.type_id() as i64)
                        .unwrap_or(0);
                    let type_id_const = alloc_const(ctx, Value::Int(type_id));
                    short.push(Op::new(
                        OpCode::GuardGcType,
                        &[op_b.clone(), type_id_const.clone()],
                    ));
                }
                // Always emit ARRAYLEN_GC + bound guards: pyre's
                // ArrayPtrInfo.lenbound is a plain `IntBound`, not an
                // `Option`, so the parity check is on `is_unbounded()`
                // rather than `is None`.
                if !info.lenbound.is_unbounded() {
                    let mut lenop =
                        Op::with_descr(OpCode::ArraylenGc, &[op_b.clone()], info.descr.clone());
                    // info.py:637 `lenop = ResOperation(ARRAYLEN_GC, [op])`
                    // followed by `lenbound.make_guards(lenop, ...)` — the
                    // `lenop` object is the consumer's box arg via Python
                    // identity. Allocate a fresh Int OpRef on `lenop.pos`
                    // so the chained INT_GE/INT_LE/INT_AND check against
                    // the producer result, not the sentinel `OpRef::NONE`.
                    lenop.pos.set(ctx.alloc_op_position_typed(Type::Int));
                    let lenop_pos = lenop.pos.get();
                    short.push(lenop);
                    info.lenbound.make_guards(lenop_pos, short, ctx);
                }
            }
            // info.py:379-384 `AbstractRawPtrInfo.make_guards`:
            //
            // ```python
            // def make_guards(self, op, short, optimizer):
            //     from rpython.jit.metainterp.optimizeopt.optimizer import CONST_0
            //     op = ResOperation(rop.INT_EQ, [op, CONST_0])
            //     short.append(op)
            //     op = ResOperation(rop.GUARD_FALSE, [op])
            //     short.append(op)
            // ```
            //
            // Emits "must not be 0" check (null-pointer equivalent for
            // Int-typed raw pointers) at the short-preamble entry.
            // Both `RawBufferPtrInfo` (info.py:386) and
            // `RawSlicePtrInfo` (info.py:459) inherit this override.
            PtrInfo::VirtualRawBuffer(_) | PtrInfo::VirtualRawSlice(_) => {
                let zero = alloc_const(ctx, Value::Int(0));
                let mut eq_op = Op::new(OpCode::IntEq, &[op_b.clone(), zero.clone()]);
                // info.py:381 `op = ResOperation(INT_EQ, [...])` then
                // `[op]` — INT_EQ result identity for GUARD_FALSE.
                eq_op.pos.set(ctx.alloc_op_position_typed(Type::Int));
                let eq_pos = eq_op.pos.get();
                short.push(eq_op);
                // info.py:381 reuses the INT_EQ ResOperation object as the
                // GUARD_FALSE arg by identity. The producer lives in `short`,
                // not in ctx's registries, so bind the position to its
                // canonical stand-in (resolved at replay) the same way the
                // sibling array/str/IntBound arms do, rather than minting a
                // position-only operand that has no producer to bind.
                let arg_eq = ctx.materialize_operand_at(eq_pos);
                short.push(Op::new(OpCode::GuardFalse, &[arg_eq]));
            }
            PtrInfo::Str(sinfo) => {
                // vstring.py:116-126: StrPtrInfo.make_guards
                short.push(Op::new(OpCode::GuardNonnull, &[op_b.clone()]));
                if let Some(ref bound) = sinfo.lenbound {
                    if bound.lower >= 1 {
                        let lenop_code = if sinfo.mode == 0 {
                            OpCode::Strlen
                        } else {
                            OpCode::Unicodelen
                        };
                        let mut lenop = Op::new(lenop_code, &[op_b.clone()]);
                        // vstring.py:124 `lenop = ResOperation(STRLEN, [op])`
                        // is consumed by `bound.make_guards(lenop, ...)`.
                        // Materialize the producer result before the chain.
                        lenop.pos.set(ctx.alloc_op_position_typed(Type::Int));
                        let lenop_pos = lenop.pos.get();
                        short.push(lenop);
                        // intutils.py:1264-1289 IntBound.make_guards: emits the
                        // chained INT_GE/INT_LE/INT_AND → GUARD_TRUE/GUARD_VALUE
                        // pairs against `lenop_pos`.
                        bound.make_guards(lenop_pos, short, ctx);
                    }
                }
            }
            // Virtuals/Virtualizable: no guards needed in short preamble
            _ => {}
        }
    }

    /// info.py:74-75 / vstring.py:103-105 / 249-258 — common string-length
    /// query across `ConstPtrInfo` and `StrPtrInfo`.
    fn get_known_str_length(&self, ctx: &crate::optimizeopt::OptContext, mode: u8) -> Option<i64> {
        match self {
            PtrInfo::Str(info) => info.getstrlen(ctx, mode),
            // info.py:804-808 ConstPtrInfo.getstrlen — delegate to
            // the runtime resolver for constant string pointers.
            PtrInfo::Constant(gcref) if !gcref.is_null() => ctx
                .string_length_resolver
                .as_deref()
                .and_then(|resolver| resolver(*gcref, mode)),
            _ => None,
        }
    }

    /// info.py:793 ConstPtrInfo.get_constant_string_spec and
    /// vstring.py:178 / 236 / 298 — recursive constant string extraction.
    fn get_constant_string_spec(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<Vec<i64>> {
        match self {
            PtrInfo::Str(info) => info.get_constant_string_spec(ctx, mode),
            // info.py:793: ConstPtrInfo.get_constant_string_spec
            // delegates to _unpack_str(mode) → extracts chars from the
            // constant GcRef.
            PtrInfo::Constant(gcref) if !gcref.is_null() => ctx
                .string_content_resolver
                .as_deref()
                .and_then(|resolver| resolver(*gcref, mode)),
            PtrInfo::Constant(_) => None,
            _ => None,
        }
    }

    /// vstring.py:172 / 230 `strgetitem()` on string ptrinfo — virtual dispatch only.
    /// ConstPtr constant resolution is handled by `OptString::strgetitem`
    /// (vstring.py:393-403 `_strgetitem`), which needs `&mut OptContext`.
    fn strgetitem(&self, index: i64, ctx: &crate::optimizeopt::OptContext) -> Option<OpRef> {
        match self {
            PtrInfo::Str(info) => info.strgetitem(index, ctx),
            _ => None,
        }
    }

    /// info.py:331 / 369 / 376 / 445 / 485 / 598 / 701 +
    /// vstring.py:211 / 263 / 333 `visitor_dispatch_virtual_type`.
    ///
    /// Each virtual `PtrInfo` subclass implements `visitor_dispatch_virtual_type(visitor)`
    /// which calls the corresponding `visitor.visit_*()` method with the
    /// subclass's static metadata (descr, fielddescrs, array clear flag,
    /// raw buffer offsets, etc.). The visitor is free to produce a
    /// `VInfo` per call; the same visitor pattern is shared by
    /// `ResumeDataVirtualAdder` (resume.py:312) and `VirtualStateConstructor`
    /// (virtualstate.py:721).
    ///
    /// Returns `None` for non-virtual `PtrInfo` variants — RPython's
    /// `visitor_dispatch_virtual_type` is only defined on
    /// `AbstractVirtualPtrInfo` subclasses, so callers must check
    /// `is_virtual()` first.
    fn visitor_dispatch_virtual_type<V: crate::walkvirtual::VirtualVisitor>(
        &self,
        visitor: &mut V,
    ) -> Option<V::VInfo> {
        match self {
            // info.py:331-334 InstancePtrInfo.visitor_dispatch_virtual_type.
            // `fields` still stores sparse `(field_index, OpRef)` entries, but
            // the visitor now rebuilds the full descriptor-order slot list so
            // resume.py can pair `fielddescrs` and `fieldnums` 1:1 again.
            PtrInfo::Virtual(info) => {
                let indices: Vec<u32> = info.fields.iter().map(|(fi, _)| *fi).collect();
                let fielddescrs = self.all_fielddescrs_from_descr();
                Some(visitor.visit_virtual(&info.descr, &indices, &fielddescrs))
            }
            // info.py:369-372 StructPtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualStruct(info) => {
                let indices: Vec<u32> = info.fields.iter().map(|(fi, _)| *fi).collect();
                let fielddescrs = self.all_fielddescrs_from_descr();
                Some(visitor.visit_vstruct(&info.descr, &indices, &fielddescrs))
            }
            // info.py:598-599 ArrayPtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualArray(info) => Some(visitor.visit_varray(&info.descr, info.clear)),
            // info.py:701-704 ArrayStructInfo.visitor_dispatch_virtual_type.
            // The visitor consumes the canonical `fielddescrs` ordering; the
            // compatibility indices are the same descriptor-order slot numbers.
            PtrInfo::VirtualArrayStruct(info) => {
                let indices: Vec<u32> = (0..info.fielddescrs.len()).map(|i| i as u32).collect();
                Some(visitor.visit_varraystruct(
                    &info.descr,
                    info.element_fields.len(),
                    &indices,
                    &info.fielddescrs,
                ))
            }
            // info.py:445-450 RawBufferPtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualRawBuffer(info) => Some(visitor.visit_vrawbuffer(
                info.func,
                info.size,
                info.buffer.offsets(),
                info.buffer.descrs(),
            )),
            // info.py:485-486 RawSlicePtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualRawSlice(info) => Some(visitor.visit_vrawslice(info.offset)),
            // vstring.py:211-212 / 263-264 / 333-334 per-variant dispatch
            PtrInfo::Str(info) if info.is_virtual() => {
                let is_unicode = info.mode != 0;
                Some(match &info.variant {
                    VStringVariant::Plain(_) => visitor.visit_vstrplain(is_unicode),
                    VStringVariant::Concat(_) => visitor.visit_vstrconcat(is_unicode),
                    VStringVariant::Slice(_) => visitor.visit_vstrslice(is_unicode),
                    VStringVariant::Ptr => unreachable!("non-virtual Str reached virtual arm"),
                })
            }
            _ => None,
        }
    }

    /// info.py:137-160 / 222-226: force_box() emits the allocation and
    /// field writes via emit_extra(), recursively forcing child virtuals.
    ///
    /// Generated ops are routed via emit_extra() (RPython
    /// emit_extra parity) so downstream passes can observe them.
    fn force_box(&mut self, op: &Operand, ctx: &mut crate::optimizeopt::OptContext) -> OpRef {
        force_box_impl(self, op, ctx)
    }

    /// info.py:273-303: _is_immutable_and_filled_with_constants
    ///
    /// ```text
    /// if not self.descr.is_immutable():
    ///     return False
    /// for op in self._fields:
    ///     if op is None:
    ///         return False     # uninitialized field
    ///     ...
    /// ```
    ///
    /// Check if this virtual is immutable and all fields are constants.
    /// Used by force_box to determine if the virtual can be constant-folded.
    fn is_immutable_and_filled_with_constants(&self, ctx: &crate::optimizeopt::OptContext) -> bool {
        let (fields, descr) = match self {
            PtrInfo::Virtual(v) => (&v.fields, &v.descr),
            PtrInfo::VirtualStruct(v) => (&v.fields, &v.descr),
            _ => return false,
        };
        // info.py:281: `if not self.descr.is_immutable()`.
        let Some(size_descr) = descr.as_size_descr() else {
            return false;
        };
        if !size_descr.is_immutable() {
            return false;
        }
        // info.py:286-288: `for op in self._fields: if op is None: return False`.
        // RPython's _fields is pre-allocated to len(descr.get_all_fielddescrs())
        // with None for unset slots; pyre stores only set entries in `fields`,
        // so parity requires fields.len() to match all_fielddescrs().len()
        // before treating the virtual as filled.
        if fields.len() != size_descr.all_fielddescrs().len() {
            return false;
        }
        for (_, val) in fields {
            let resolved_box = ctx.resolve_operand_operand_opt(val);
            if resolved_box
                .as_ref()
                .and_then(|cb| cb.const_value())
                .is_none()
            {
                // Check if it's a virtual that is also immutable+constant.
                // info.py:282-303 threads a `memo` keyed on the receiver info
                // (`if self in memo: return True`) to break recursion on a
                // cyclic immutable virtual (A.field -> B, B.field -> A). pyre
                // omits the memo: a JIT-created virtual cannot close such a
                // cycle, since immutable fields are written by SETFIELD in
                // construction order and the second box does not exist when the
                // first is sealed. Convergence path if a prebuilt cyclic
                // immutable virtual ever reaches here: thread a
                // `memo: &mut Vec<BoxRef>` keyed on the receiver box (BoxRef
                // Rc-identity) mirroring info.py:282-284.
                if let Some(info) = resolved_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
                    if info.is_virtual() && info.is_immutable_and_filled_with_constants(ctx) {
                        continue;
                    }
                }
                return false;
            }
        }
        true
    }
}

fn force_box_impl(
    self_: &mut PtrInfo,
    op: &Operand,
    ctx: &mut crate::optimizeopt::OptContext,
) -> OpRef {
    use majit_ir::{Op, OpCode};

    // `op` is the bound operand of the virtual being forced (callers resolve to
    // the chain terminal before delegating). The OpRef view drives op identity
    // (pos, logging, alloc-vs-original comparisons); the operand drives every
    // make_equal_to / set_ptr_info receiver, so no `materialize_box_at` round-trip is
    // needed for the forwarding writes.
    let opref = op.to_opref();

    // info.py:307: subbox = optimizer.force_box(fld) — `fld` is the field's
    // own box. Force it (materialising a nested virtual) and return the
    // resulting box, which the parent uses as a SETFIELD/SETARRAYITEM value
    // operand. Box-native: the field box is passed and returned directly,
    // matching `force_box(fld)` rather than an OpRef round-trip.
    fn force_child(orig: &Operand, ctx: &mut crate::optimizeopt::OptContext) -> Operand {
        let value_box = ctx.resolve_operand_operand_opt(orig);
        // optimizer.py:356-357: `info = op.get_forwarded()` then `if op.type ==
        // 'i' and info.is_constant(): return ConstInt(info.get_constant_int())`.
        // The const-int synthesis is gated on an info that is ALREADY forwarded
        // — force_box only reads the slot. `peek_intbound_box` reads the
        // forwarded IntBound without the lazy `set_forwarded(IntBound())`
        // install that `getintbound_handle` performs (optimizer.py:111), so a
        // plain int box with no info is returned as-is below rather than given a
        // fresh unbounded slot.
        if let Some(b) = &value_box {
            if b.const_value().is_none() && b.type_() == majit_ir::Type::Int {
                if let Some(bound) = ctx.peek_intbound_box(b) {
                    if bound.is_constant() {
                        let c = ctx.make_constant_int(bound.get_constant_int());
                        return ctx.materialize_operand_at(c);
                    }
                }
            }
        }
        let value_is_virtual = match &value_box {
            Some(b) => ctx.is_virtual(b),
            None => false,
        };
        if value_is_virtual {
            let vb_op = value_box.expect("recorder-populated");
            let mut info = ctx.take_ptr_info(&vb_op).unwrap();
            let forced = force_box_impl(&mut info, &vb_op, ctx);
            return ctx
                .get_box_replacement_operand_opt(forced)
                .unwrap_or_else(|| ctx.materialize_operand_at(forced));
        }
        value_box.unwrap_or_else(|| orig.clone())
    }

    // RPython info.py:148,226: optforce.emit_extra(op)
    // `optforce` determines where emitted ops enter the pass chain:
    //   optforce=Optimizer (in_final_emission) → emit directly
    //   optforce=OptEarlyForce → route from earlyforce.next (= heap)
    // When called from EarlyForce pass, current_pass_idx == earlyforce_idx
    // so emit_extra automatically routes from earlyforce.next.
    // When called from _emit_operation, in_final_emission=true → direct.
    let emit_op = |ctx: &mut crate::optimizeopt::OptContext, op: Op| -> OpRef {
        if ctx.in_final_emission {
            ctx.emit(op)
        } else {
            ctx.emit_extra(ctx.current_pass_idx, op)
        }
    };

    // Descr-derived view of the full fielddescr slot list, used by both
    // the constant-fold path and the per-field SETFIELD_GC emission in the
    // Virtual/VirtualStruct match arms below. Computed once so the call
    // sites don't need to re-borrow `self` while `vinfo` is borrowed.
    let cached_fielddescrs = self_.all_fielddescrs_from_descr();

    // RPython info.py:140-145: immutable virtual filled with constants
    // → constant fold to a compile-time constant pointer.
    if self_.is_immutable_and_filled_with_constants(ctx) {
        // info.py:305-311 _force_elements_immutable: `subbox =
        // optimizer.force_box(fld)` materializes EACH field — including a
        // nested immutable virtual — into a Const before the parent's slots
        // are written. `is_immutable_and_filled_with_constants` guarantees a
        // non-Const field is itself an immutable virtual filled with
        // constants, so `force_child` recurses through this same fold and
        // const-folds it (no SETFIELD ops emitted). Run before borrowing
        // `constant_fold_alloc` (force_child needs `&mut ctx`); a field that
        // already resolves to a Const is left untouched.
        // Should a nested field fail to const-fold (its own fold fell back to
        // materialization), the parent's constant slot would be left
        // uninitialized, so fold the parent only when every field is constant.
        let mut can_fold_parent = true;
        if ctx.constant_fold_alloc.is_some() {
            let field_boxes: Vec<Operand> = match self_ {
                PtrInfo::Virtual(v) => v.fields.iter().map(|(_, r)| r.clone()).collect(),
                PtrInfo::VirtualStruct(v) => v.fields.iter().map(|(_, r)| r.clone()).collect(),
                _ => Vec::new(),
            };
            for fb in field_boxes {
                if ctx
                    .resolve_operand_operand_opt(&fb)
                    .and_then(|b| b.const_value())
                    .is_none()
                {
                    force_child(&fb, ctx);
                    if ctx
                        .resolve_operand_operand_opt(&fb)
                        .and_then(|b| b.const_value())
                        .is_none()
                    {
                        can_fold_parent = false;
                    }
                }
            }
        }
        if let Some(alloc_fn) = ctx.constant_fold_alloc.as_ref().filter(|_| can_fold_parent) {
            let field_descrs = &cached_fielddescrs;
            let (descr, fields) = match self_ {
                PtrInfo::Virtual(v) => (&v.descr, &v.fields),
                PtrInfo::VirtualStruct(v) => (&v.descr, &v.fields),
                _ => unreachable!(),
            };
            let obj_size = descr.as_size_descr().map(|sd| sd.size()).unwrap_or(0);
            if obj_size > 0 {
                let ptr = alloc_fn(obj_size);
                if !ptr.is_null() {
                    // info.py:144: _force_elements_immutable
                    // Write constant field values directly to the allocated memory.
                    for (field_idx, val_ref) in fields.iter() {
                        let field_idx = *field_idx;
                        if let Some(value) = ctx
                            .resolve_operand_operand_opt(val_ref)
                            .and_then(|cb| cb.const_value())
                        {
                            if let Some(fd) = lookup_field_descr(field_descrs, field_idx) {
                                if let Some(field_d) = fd.as_field_descr() {
                                    let offset = field_d.offset();
                                    match value {
                                        Value::Int(v) => unsafe {
                                            let dest = (ptr.0 as *mut u8).add(offset) as *mut i64;
                                            *dest = v;
                                        },
                                        Value::Ref(r) => unsafe {
                                            let dest = (ptr.0 as *mut u8).add(offset) as *mut usize;
                                            *dest = r.0;
                                        },
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    // info.py:142: op.set_forwarded(constptr) — write
                    // unconditional. `set_ptr_info` on `op` walks the
                    // chain to the just-installed Const target (where it
                    // is a no-op per Const-box invariant).
                    let const_ref = GcRef(ptr.0);
                    ctx.make_constant_arg(op, Value::Ref(const_ref));
                    ctx.set_ptr_info(op, PtrInfo::Constant(const_ref));
                    return opref;
                }
            }
        }
        // No allocator or size unknown: fall through to normal force.
    }

    match self_ {
        PtrInfo::VirtualStruct(vinfo) => {
            // RPython info.py:216-226 _force_elements clears each
            // `self._fields[i] = None` BEFORE `optforce.emit_extra(setfieldop)`.
            // After force, the non-virtual structinfo carries no field cache,
            // so heap.py do_setfield records the SETFIELD_GC as a lazy_set
            // instead of MUST_ALIAS-eliding it against the preserved value.
            let preserved = PtrInfo::Struct(StructPtrInfo {
                descr: vinfo.descr.clone(),
                fields: Vec::new(),
                last_guard_pos: -1,
            });
            let mut new_op = Op::new(OpCode::New, &[]);
            // RPython info.py:146-151 force_box emits the ORIGINAL box op.
            // Preserve that identity here instead of inventing a fresh
            // OpRef, so later passes (earlyforce → heap → call) all talk
            // about the same concrete allocation.
            new_op.pos.set(opref);
            new_op.setdescr(vinfo.descr.clone());
            let alloc_ref = emit_op(ctx, new_op);
            // info.py:152 `newop.set_forwarded(self)` — unconditional.
            // The just-emitted alloc op is bound, so its resolved box
            // carries the PtrInfo install.
            if let Some(b) = ctx.get_box_replacement_operand_opt(alloc_ref) {
                ctx.set_ptr_info(&b, preserved);
            }
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!(
                    "[jit][force-box] virtual-struct {:?} -> {:?} in_final_emission={} pass_idx={}",
                    opref, alloc_ref, ctx.in_final_emission, ctx.current_pass_idx
                );
            }
            if opref != alloc_ref {
                let b_alloc = ctx.get_box_replacement_operand(alloc_ref);
                ctx.make_equal_to(op, &b_alloc);
            }
            for (field_idx, value_ref) in std::mem::take(&mut vinfo.fields) {
                let value_ref = force_child(&value_ref, ctx);
                let descr = lookup_field_descr(&cached_fielddescrs, field_idx);
                debug_assert!(
                    descr.is_some(),
                    "force_box: field_idx={} has value but no descriptor \
                     — field_descrs out of sync with fields",
                    field_idx,
                );
                let descr = descr.expect(
                    "force_box: field_idx must resolve through descr.get_all_fielddescrs()[i]",
                );
                let arg_alloc = ctx.materialize_operand_at(alloc_ref);
                let arg_value = ctx.resolve_operand_operand(&value_ref);
                let mut set_op =
                    Op::new(OpCode::SetfieldGc, &[arg_alloc.clone(), arg_value.clone()]);
                set_op.setdescr(descr);
                emit_op(ctx, set_op);
            }
            alloc_ref
        }
        PtrInfo::Virtual(vinfo) => {
            // info.py:216-226 — see VirtualStruct branch above. Build the
            // non-virtual replacement with no field cache so heap.py
            // do_setfield does not MUST_ALIAS-elide the materialization
            // SETFIELD_GC against the preserved value.
            let preserved = PtrInfo::Instance(InstancePtrInfo {
                descr: Some(vinfo.descr.clone()),
                known_class: vinfo.known_class,
                fields: Vec::new(),
                last_guard_pos: -1,
            });
            let mut new_op = Op::new(OpCode::NewWithVtable, &[]);
            // RPython info.py:146-151 force_box emits the ORIGINAL box op.
            // Preserve that identity here instead of inventing a fresh
            // OpRef, so later passes (earlyforce → heap → call) all talk
            // about the same concrete allocation.
            new_op.pos.set(opref);
            new_op.setdescr(vinfo.descr.clone());
            let alloc_ref = emit_op(ctx, new_op);
            // info.py:152 `newop.set_forwarded(self)` — unconditional.
            if let Some(b) = ctx.get_box_replacement_operand_opt(alloc_ref) {
                ctx.set_ptr_info(&b, preserved);
            }
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!(
                    "[jit][force-box] virtual {:?} -> {:?} in_final_emission={} pass_idx={}",
                    opref, alloc_ref, ctx.in_final_emission, ctx.current_pass_idx
                );
            }
            if opref != alloc_ref {
                let b_alloc = ctx.get_box_replacement_operand(alloc_ref);
                ctx.make_equal_to(op, &b_alloc);
            }
            for (field_idx, value_ref) in std::mem::take(&mut vinfo.fields) {
                let value_ref = force_child(&value_ref, ctx);
                let descr = lookup_field_descr(&cached_fielddescrs, field_idx);
                let descr = descr.expect(
                    "force_box: field_idx must resolve through descr.get_all_fielddescrs()[i]",
                );
                let arg_alloc = ctx.materialize_operand_at(alloc_ref);
                let arg_value = ctx.resolve_operand_operand(&value_ref);
                let mut set_op =
                    Op::new(OpCode::SetfieldGc, &[arg_alloc.clone(), arg_value.clone()]);
                set_op.setdescr(descr);
                emit_op(ctx, set_op);
            }
            alloc_ref
        }
        PtrInfo::VirtualArray(vinfo) => {
            // info.py:152: AbstractVirtualPtrInfo.force_box does
            // `newop.set_forwarded(self); self._is_virtual = False` — it keeps
            // the SAME ArrayPtrInfo, now non-virtual, so getarrayinfo still
            // recovers the array identity after forcing.
            // PRE-EXISTING DIVERGENCE: pyre installs `nonnull()` instead,
            // dropping the array identity (a later GETARRAYITEM residualizes
            // rather than reading a tracked item). Convergence needs an
            // `is_virtual` flag on `VirtualArrayInfo` plus an `is_virtual()`
            // gate at the `matches!(PtrInfo::VirtualArray(_))` sites in
            // virtualize.rs (same shape as the RawBuffer size=-1 sentinel).
            let len = vinfo.items.len();
            ctx.set_ptr_info(op, PtrInfo::nonnull());

            let len_ref = ctx.emit_constant_int(len as i64);
            let alloc_opcode = if vinfo.clear {
                OpCode::NewArrayClear
            } else {
                OpCode::NewArray
            };
            let arg_len = ctx.materialize_operand_at(len_ref);
            let mut alloc_op = Op::new(alloc_opcode, &[arg_len.clone()]);
            alloc_op.pos.set(opref);
            alloc_op.setdescr(vinfo.descr.clone());
            let alloc_ref = emit_op(ctx, alloc_op);
            if opref != alloc_ref {
                let b_alloc = ctx.get_box_replacement_operand(alloc_ref);
                ctx.make_equal_to(op, &b_alloc);
            }

            // info.py:542: const = optforce.optimizer.new_const_item(self.descr)
            // info.py:546-548: skip items equal to the default when _clear=True
            let items = std::mem::take(&mut vinfo.items);
            let clear = vinfo.clear;
            let descr = vinfo.descr.clone();
            for (i, item_ref) in items.into_iter().enumerate() {
                if item_ref.is_none() {
                    continue;
                }
                // info.py:543: const = optforce.optimizer.new_const_item(self.descr)
                // info.py:546-548: if self._clear and const.same_constant(item)
                // new_const_item returns CONST_0/CONST_NULL/CONST_ZERO_FLOAT
                // (all raw=0).
                if clear {
                    let is_default = ctx
                        .resolve_operand_operand_opt(&item_ref)
                        .as_ref()
                        .and_then(|b| ctx.getconst(b))
                        .map_or(false, |(raw, _)| raw == 0);
                    if is_default {
                        continue;
                    }
                }
                let subbox = force_child(&item_ref, ctx);
                let idx_ref = ctx.emit_constant_int(i as i64);
                let arg_alloc = ctx.materialize_operand_at(alloc_ref);
                let arg_idx = ctx.materialize_operand_at(idx_ref);
                let arg_sub = ctx.resolve_operand_operand(&subbox);
                let mut set_op = Op::new(
                    OpCode::SetarrayitemGc,
                    &[arg_alloc.clone(), arg_idx.clone(), arg_sub.clone()],
                );
                set_op.setdescr(descr.clone());
                emit_op(ctx, set_op);
            }
            // info.py:557: optforce.pure_from_args(ARRAYLEN_GC, [op], ConstInt(len))
            ctx.pure_from_args_arraylen(alloc_ref, len as i64);
            alloc_ref
        }
        PtrInfo::VirtualArrayStruct(vinfo) => {
            // info.py:670-684 ArrayStructInfo._force_elements
            // virtualize.py:31: assert clear — ArrayStruct is always
            // created with clear=True, so the original op is always
            // NEW_ARRAY_CLEAR.
            // info.py:152 force_box keeps the SAME info (`set_forwarded(self);
            // _is_virtual = False`). PRE-EXISTING DIVERGENCE: pyre installs
            // `nonnull()`, dropping the array-struct identity — same convergence
            // path as VirtualArray (an `is_virtual` flag + gated match sites).
            let num_elements = vinfo.element_fields.len();
            ctx.set_ptr_info(op, PtrInfo::nonnull());

            let len_ref = ctx.emit_constant_int(num_elements as i64);
            let arg_len = ctx.materialize_operand_at(len_ref);
            let mut alloc_op = Op::new(OpCode::NewArrayClear, &[arg_len.clone()]);
            alloc_op.pos.set(opref);
            alloc_op.setdescr(vinfo.descr.clone());
            let alloc_ref = emit_op(ctx, alloc_op);
            if opref != alloc_ref {
                let b_alloc = ctx.get_box_replacement_operand(alloc_ref);
                ctx.make_equal_to(op, &b_alloc);
            }

            // info.py:672: fielddescrs = op.getdescr().get_all_fielddescrs()
            let fielddescrs: Vec<majit_ir::DescrRef> = vinfo
                .descr
                .as_array_descr()
                .and_then(|ad| ad.get_all_interiorfielddescrs())
                .map(|fds| fds.to_vec())
                .unwrap_or_else(|| vinfo.fielddescrs.clone());
            let element_fields = std::mem::take(&mut vinfo.element_fields);
            // info.py:673-684:
            //   for index in range(self.length):
            //       for fielddescr in fielddescrs:
            //           fld = self._items[i]
            //           if fld is not None:
            //               subbox = optforce.optimizer.force_box(fld)
            //               setfieldop = ResOperation(SETINTERIORFIELD_GC,
            //                   [op, ConstInt(index), subbox], descr=fielddescr)
            //               optforce.emit_extra(setfieldop)
            //           i += 1
            for (elem_idx, fields) in element_fields.into_iter().enumerate() {
                let idx_ref = ctx.emit_constant_int(elem_idx as i64);
                for (field_idx, value_ref) in fields {
                    if value_ref.is_none() {
                        continue;
                    }
                    let subbox = force_child(&value_ref, ctx);
                    let arg_alloc = ctx.materialize_operand_at(alloc_ref);
                    let arg_idx = ctx.materialize_operand_at(idx_ref);
                    let arg_sub = ctx.resolve_operand_operand(&subbox);
                    let mut set_op = Op::new(
                        OpCode::SetinteriorfieldGc,
                        &[arg_alloc.clone(), arg_idx.clone(), arg_sub.clone()],
                    );
                    if let Some(d) = fielddescrs.get(field_idx as usize).cloned() {
                        set_op.setdescr(d);
                    }
                    emit_op(ctx, set_op);
                }
            }
            alloc_ref
        }
        PtrInfo::VirtualRawBuffer(vinfo) => {
            // info.py:420-436: RawBufferPtrInfo._force_elements()
            // info.py:421: self.size = -1 (mark as no longer virtual)
            let entries = vinfo.buffer.drain_entries();
            let func = vinfo.func;
            let size = vinfo.size;
            let calldescr = vinfo.calldescr.take();

            // info.py:148: emit CALL_I(func, ConstInt(size), descr=calldescr)
            let func_ref = ctx.emit_constant_int(func);
            let size_ref = ctx.emit_constant_int(size as i64);
            let arg_func = ctx.materialize_operand_at(func_ref);
            let arg_size = ctx.materialize_operand_at(size_ref);
            let mut call_op = Op::new(OpCode::CallI, &[arg_func.clone(), arg_size.clone()]);
            call_op.pos.set(opref);
            if let Some(d) = calldescr {
                call_op.setdescr(d);
            }
            let alloc_ref = emit_op(ctx, call_op);

            // info.py:152/421 set_forwarded(self). RPython keeps the SAME
            // RawBufferPtrInfo on the forced op and flips it non-virtual via
            // `self.size = -1` (is_virtual = `size != -1`), so a later
            // `getrawptrinfo` still recovers the raw-buffer identity. pyre
            // installs `nonnull()` instead — a PRE-EXISTING-ADAPTATION: the
            // RawSlice force keeps identity via a `parent = none` sentinel
            // (see below), but a non-virtual RawBuffer would need a size=-1
            // sentinel on `RawBufferPtrInfo` AND an `is_virtual()` gate at
            // every `matches!(PtrInfo::VirtualRawBuffer(_))` site in
            // virtualize.rs (1232/1261/1269/…), which currently assume virtual
            // and would re-virtualize a forced buffer. Raw buffers are absent
            // from the check.py corpus, so that cascade is ungateable here.
            // nonnull() preserves nonnull-ness; a later RAW_LOAD/RAW_STORE just
            // residualizes (getrawptrinfo→None), which is the correct result.
            if let Some(b) = ctx.get_box_replacement_operand_opt(alloc_ref) {
                ctx.set_ptr_info(&b, PtrInfo::nonnull());
            }
            if opref != alloc_ref {
                let b_alloc = ctx.get_box_replacement_operand(alloc_ref);
                ctx.make_equal_to(op, &b_alloc);
            }

            // info.py:425: CHECK_MEMORY_ERROR
            let arg_alloc = ctx.materialize_operand_at(alloc_ref);
            let check_op = Op::new(OpCode::CheckMemoryError, &[arg_alloc.clone()]);
            emit_op(ctx, check_op);

            // info.py:429-436: emit RAW_STORE for each buffered write.
            // info.py:433 uses `itembox = buffer.values[i]` DIRECTLY in the
            // RAW_STORE — unlike the struct/array paths (info.py:222/548),
            // RawBufferPtrInfo._force_elements does NOT optimizer.force_box the
            // value (info.py:240: "there can be no virtuals stored in raw
            // buffer"). Use the stored value box as-is; routing it through
            // force_child would force-box it and synthesize a ConstInt for a
            // constant-bound int, neither of which upstream does here.
            for (offset, _length, descr, value) in entries {
                let value_box = ctx.materialize_operand_at(value);
                let offset_ref = ctx.emit_constant_int(offset);
                let arg_alloc = ctx.materialize_operand_at(alloc_ref);
                let arg_offset = ctx.materialize_operand_at(offset_ref);
                let arg_value = ctx.resolve_operand_operand(&value_box);
                let mut store_op = Op::new(
                    OpCode::RawStore,
                    &[arg_alloc.clone(), arg_offset.clone(), arg_value.clone()],
                );
                store_op.setdescr(descr);
                emit_op(ctx, store_op);
            }

            alloc_ref
        }
        PtrInfo::VirtualRawSlice(slice) => {
            // `info.py:473-476` `RawSlicePtrInfo._force_elements`:
            //
            // ```python
            // def _force_elements(self, op, optforce, descr):
            //     if self.parent.is_virtual():
            //         self.parent._force_elements(op, optforce, descr)
            //     self.parent = None
            // ```
            //
            // RPython keeps the `RawSlicePtrInfo` attached to the op and
            // flips it to non-virtual by setting `self.parent = None`
            // (`is_virtual` at info.py:464-465 is `self.parent is not None`).
            // The info class stays RawSlicePtrInfo so subsequent
            // `getrawptrinfo` lookups still identify it as a raw slice.
            //
            // pyre's `RawSlicePtrInfo` stores `parent: OpRef`; the
            // `OpRef::NONE` sentinel plays the role of `None`, and
            // `PtrInfo::is_virtual` gates on `slice.parent.is_none()`.
            // Overwriting with `PtrInfo::nonnull()` would lose the
            // raw-slice identity and mis-route any later
            // `get_virtual_fields` / raw-guard path.
            let parent_forced = force_child(&slice.parent, ctx);
            let offset_ref = ctx.emit_constant_int(slice.offset as i64);
            let arg_parent = ctx.resolve_operand_operand(&parent_forced);
            let arg_offset = ctx.materialize_operand_at(offset_ref);
            let mut add_op = Op::new(OpCode::IntAdd, &[arg_parent.clone(), arg_offset.clone()]);
            add_op.pos.set(opref);
            let new_ref = emit_op(ctx, add_op);
            // Preserve raw-slice identity; mark non-virtual via
            // `parent = OpRef::NONE` (RPython `self.parent = None`).
            // info.py:152 unconditional set_forwarded — the emitted
            // IntAdd op is bound, so its resolved box carries PtrInfo.
            if let Some(b) = ctx.get_box_replacement_operand_opt(new_ref) {
                ctx.set_ptr_info(
                    &b,
                    PtrInfo::VirtualRawSlice(RawSlicePtrInfo {
                        offset: slice.offset,
                        parent: Operand::None,
                        last_guard_pos: slice.last_guard_pos,
                        avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                    }),
                );
            }
            if opref != new_ref {
                let b_new = ctx.get_box_replacement_operand(new_ref);
                ctx.make_equal_to(op, &b_new);
            }
            new_ref
        }
        PtrInfo::Str(sinfo) if sinfo.is_virtual() => {
            // vstring.py:76-103 StrPtrInfo.force_box
            let mode = sinfo.mode;
            let is_unicode = mode != 0;

            // vstring.py:79-90: if self.mode is mode_string / else
            let c_s = if mode == crate::optimizeopt::vstring::mode_string {
                // vstring.py:80-84
                sinfo
                    .get_constant_string_spec(&*ctx, mode)
                    .and_then(|chars| crate::history::get_const_ptr_for_string(&chars, ctx))
            } else {
                // vstring.py:86-90
                sinfo
                    .get_constant_string_spec(&*ctx, mode)
                    .and_then(|chars| crate::history::get_const_ptr_for_unicode(&chars, ctx))
            };
            if let Some(gcref) = c_s {
                // vstring.py:83: get_box_replacement(op).set_forwarded(c_s)
                ctx.make_constant_arg(op, Value::Ref(gcref));
                return opref;
            }

            // vstring.py:91: self._is_virtual = False
            let sinfo_full = match std::mem::replace(self_, PtrInfo::nonnull()) {
                PtrInfo::Str(s) => s,
                _ => unreachable!(),
            };
            let variant = sinfo_full.variant;

            // vstring.py:92: lengthbox = self.getstrlen(op, optstring, mode)
            let lengthbox = match &variant {
                VStringVariant::Plain(info) => ctx.emit_constant_int(info._chars.len() as i64),
                VStringVariant::Slice(info) => ctx
                    .resolve_operand_operand_opt(&info.lgtop)
                    .map(|b| b.to_opref())
                    .unwrap_or(info.lgtop.to_opref()),
                VStringVariant::Concat(info) => {
                    let left_len = ctx.getstrlen_opref(info.vleft.to_opref(), mode);
                    let right_len = ctx.getstrlen_opref(info.vright.to_opref(), mode);
                    let left_len = ctx.materialize_operand_at(left_len);
                    let right_len = ctx.materialize_operand_at(right_len);
                    crate::optimizeopt::vstring::_int_add(&left_len, &right_len, ctx).to_opref()
                }
                VStringVariant::Ptr => unreachable!(),
            };

            // vstring.py:93-96: newop = ResOperation(mode.NEWSTR, [lengthbox])
            let new_opcode = if is_unicode {
                OpCode::Newunicode
            } else {
                OpCode::Newstr
            };
            let arg_length = ctx.materialize_operand_at(lengthbox);
            let mut newstr_op = Op::new(new_opcode, &[arg_length.clone()]);
            newstr_op.pos.set(opref);
            let newop = emit_op(ctx, newstr_op);

            // vstring.py:98: newop.set_forwarded(self) — unconditional.
            if let Some(b) = ctx.get_box_replacement_operand_opt(newop) {
                ctx.set_ptr_info(
                    &b,
                    PtrInfo::Str(StrPtrInfo {
                        lenbound: sinfo_full.lenbound,
                        lgtop: Some(arg_length.clone()), // vstring.py:98 preserve computed length
                        mode: sinfo_full.mode,
                        length: sinfo_full.length,
                        variant: VStringVariant::Ptr, // non-virtual
                        last_guard_pos: sinfo_full.last_guard_pos,
                        avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                    }),
                );
            }

            // vstring.py:99-100: op.set_forwarded(newop)
            if opref != newop {
                let b_newop = ctx.get_box_replacement_operand(newop);
                ctx.make_equal_to(op, &b_newop);
            }

            // vstring.py:101-102: initialize_forced_string(op, optstring, op, CONST_0, mode)
            let zero = ctx.emit_constant_int(0);
            let set_opcode = if is_unicode {
                OpCode::Unicodesetitem
            } else {
                OpCode::Strsetitem
            };

            match variant {
                VStringVariant::Plain(info) => {
                    // vstring.py:194-205 VStringPlainInfo.initialize_forced_string
                    let mut offset = ctx.materialize_operand_at(zero);
                    let one = ctx.emit_constant_int(1);
                    let one = ctx.materialize_operand_at(one);
                    for ch in &info._chars {
                        if let Some(ch_ref) = ch {
                            // vstring.py:194 get_box_replacement(charbox) — walk the
                            // char operand's own forwarding (object-native).
                            let ch_resolved = ctx.resolve_operand_operand(ch_ref).to_opref();
                            let arg_newop = ctx.materialize_operand_at(newop);
                            let arg_offset = ctx.resolve_operand_operand(&offset);
                            let arg_ch = ctx.materialize_operand_at(ch_resolved);
                            let setitem_op = Op::new(
                                set_opcode,
                                &[arg_newop.clone(), arg_offset.clone(), arg_ch.clone()],
                            );
                            emit_op(ctx, setitem_op);
                        }
                        offset = crate::optimizeopt::vstring::_int_add(&offset, &one, ctx);
                    }
                }
                VStringVariant::Concat(info) => {
                    // vstring.py:309-317 VStringConcatInfo.string_copy_parts
                    let newop_box = ctx.materialize_operand_at(newop);
                    let zero_box = ctx.materialize_operand_at(zero);
                    let offset = crate::optimizeopt::vstring::string_copy_parts(
                        &info.vleft.clone(),
                        &newop_box,
                        &zero_box,
                        mode,
                        ctx,
                    );
                    crate::optimizeopt::vstring::string_copy_parts(
                        &info.vright.clone(),
                        &newop_box,
                        &offset,
                        mode,
                        ctx,
                    );
                }
                VStringVariant::Slice(info) => {
                    // vstring.py:230-233 VStringSliceInfo.string_copy_parts
                    let newop_box = ctx.materialize_operand_at(newop);
                    let zero_box = ctx.materialize_operand_at(zero);
                    crate::optimizeopt::vstring::copy_str_content(
                        ctx,
                        &info.s.clone(),
                        &newop_box,
                        &info.start.clone(),
                        &zero_box,
                        &info.lgtop.clone(),
                        mode,
                        true,
                    );
                }
                VStringVariant::Ptr => unreachable!(),
            }

            newop
        }
        _ => opref,
    }
}

/// info.py:124-128 `AbstractVirtualPtrInfo` line-by-line shared state.
///
/// ```python
/// class AbstractVirtualPtrInfo(NonNullPtrInfo):
///     _attrs_ = ('_cached_vinfo', 'descr', '_is_virtual')
///     _cached_vinfo = None
/// ```
///
/// Every concrete virtual-flavoured PtrInfo (Virtual, VirtualStruct,
/// VirtualArray, VirtualArrayStruct, VirtualRawBuffer, VirtualRawSlice,
/// Str) inherits `_cached_vinfo` from `AbstractVirtualPtrInfo`. Pyre
/// embeds this shared struct as `pub avpi: AbstractVirtualPtrInfo` in
/// each concrete variant so the inheritance contract is structural,
/// not per-variant copy-paste.
///
/// `descr` and `_is_virtual` are NOT lifted here:
///   - `descr` is variant-specific (SizeDescr for Virtual, ArrayDescr
///     for VirtualArray, etc.) — RPython's `_attrs_` is a hint to the
///     translator's slot allocator, not a parity constraint on the
///     storage *type*. Each pyre variant keeps its own typed `descr`.
///   - `_is_virtual` collapses into the pyre enum tag itself
///     (`PtrInfo::Virtual(_)` IS the truthy carrier of `_is_virtual`);
///     no separate slot is needed.
///
/// `make_virtual_info` (resume.py:307-315) reads `cached_vinfo` to
/// dedup RdVirtualInfo allocations across multiple finish() calls
/// referencing the same virtual. `RefCell` provides interior
/// mutability so the immutable-receiver accessor can populate the
/// cache on first miss.
pub use majit_ir::ptr_info::{
    AbstractVirtualPtrInfo, ArrayPtrInfo, ArrayStructInfo, InstancePtrInfo, RawBufferPtrInfo,
    RawSlicePtrInfo, StructPtrInfo, VirtualArrayInfo, VirtualInfo, VirtualStructInfo,
    VirtualizableFieldState,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::OptContext;
    use majit_ir::box_ref::BoxRef;
    use majit_ir::{Descr, OpCode, Value};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestDescr;
    impl Descr for TestDescr {}

    /// Bound-producer `Operand` at position `int_op(pos)` / `ref_op(pos)`,
    /// the field-value analog of the old `BoxRef::from_opref` test stand-ins.
    fn field_op(tp: Type, pos: u32) -> Operand {
        crate::history::test_support::rooted_resop_operand(tp, pos)
    }

    #[test]
    fn test_ptr_info_factories() {
        let nonnull = PtrInfo::nonnull();
        assert!(nonnull.is_nonnull());
        assert!(!nonnull.is_virtual());

        let constant = PtrInfo::constant(GcRef(0x1000));
        assert!(constant.is_nonnull());
        assert!(constant.is_constant());

        let kc = PtrInfo::known_class(0x2000, true);
        assert!(kc.is_nonnull());
        // Instance arm — no cpu read; DefaultCpu is a placeholder.
        assert!(kc.get_known_class(&crate::cpu::DefaultCpu).is_some());
    }

    #[test]
    fn test_ptr_info_virtual_factories() {
        let descr: DescrRef = Arc::new(TestDescr);

        let virtual_obj = PtrInfo::virtual_obj(descr.clone(), Some(0x3000));
        assert!(virtual_obj.is_virtual());
        assert!(virtual_obj.is_nonnull());
        assert!(virtual_obj.get_descr().is_some());

        let virtual_arr = PtrInfo::virtual_array(descr.clone(), 5, false);
        assert!(virtual_arr.is_virtual());
        assert_eq!(virtual_arr.num_fields(), 5);

        let virtual_struct = PtrInfo::virtual_struct(descr);
        assert!(virtual_struct.is_virtual());
    }

    #[test]
    fn test_const_ptr_info_getlenbound_returns_none_at_base() {
        // The base `PtrInfo::getlenbound` returns None for `PtrInfo::Constant`
        // — the constant string-length lookup runs through
        // `EnsuredPtrInfo::Constant::getlenbound`, which threads in the
        // runtime `string_length_resolver`. Callers that bypass
        // EnsuredPtrInfo (and thus skip the resolver) must not get a
        // misleading nonnegative answer here.
        let mut info = PtrInfo::constant(GcRef(0x1000));

        assert_eq!(info.getlenbound(Some(0)), None);
        assert_eq!(info.getlenbound(Some(1)), None);
        assert_eq!(info.getlenbound(None), None);
    }

    #[test]
    fn test_str_ptr_info_virtual_variants() {
        let mut ctx = OptContext::new(32);
        let plain = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: 2,
            variant: VStringVariant::Plain(VStringPlainInfo {
                _chars: vec![None, None],
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        assert!(plain.is_virtual());

        let slice = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: Some(ctx.materialize_operand_at(OpRef::int_op(3))), // vstring.py:223: self.lgtop = length
            mode: 0,
            length: -1,
            variant: VStringVariant::Slice(VStringSliceInfo {
                s: ctx.materialize_operand_at(OpRef::int_op(1)),
                start: ctx.materialize_operand_at(OpRef::int_op(2)),
                lgtop: ctx.materialize_operand_at(OpRef::int_op(3)),
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        assert!(slice.is_virtual());

        let concat = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: -1,
            variant: VStringVariant::Concat(VStringConcatInfo {
                vleft: ctx.materialize_operand_at(OpRef::int_op(4)),
                vright: ctx.materialize_operand_at(OpRef::int_op(5)),
                _is_virtual: true,
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        assert!(concat.is_virtual());

        let ptr = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: -1,
            variant: VStringVariant::Ptr,
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        assert!(!ptr.is_virtual());
    }

    #[test]
    fn test_str_ptr_info_constant_string_spec_and_strgetitem() {
        let mut ctx = OptContext::new(16);
        let b = ctx.materialize_operand_at(OpRef::int_op(10));
        ctx.make_constant_box(&b, Value::Int(97));
        let b = ctx.materialize_operand_at(OpRef::int_op(11));
        ctx.make_constant_box(&b, Value::Int(98));
        let b = ctx.materialize_operand_at(OpRef::int_op(12));
        ctx.make_constant_box(&b, Value::Int(99));

        let info = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: 3,
            variant: VStringVariant::Plain(VStringPlainInfo {
                _chars: vec![
                    Some(ctx.materialize_operand_at(OpRef::int_op(10))),
                    Some(ctx.materialize_operand_at(OpRef::int_op(11))),
                    Some(ctx.materialize_operand_at(OpRef::int_op(12))),
                ],
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });

        assert_eq!(
            info.get_constant_string_spec(&ctx, 0),
            Some(vec![97, 98, 99])
        );
        assert_eq!(info.get_known_str_length(&ctx, 0), Some(3));
        assert_eq!(info.strgetitem(1, &ctx), Some(OpRef::int_op(11)));
    }

    #[test]
    fn test_str_ptr_info_plain_constant_string_spec_rejects_intbound_constant_chars() {
        let mut ctx = OptContext::new(16);
        let ch = OpRef::int_op(10);
        let ch_box = ctx.materialize_operand_at(ch);
        ctx.setintbound(&ch_box, &IntBound::from_constant(97));

        assert_eq!(
            ctx.get_box_replacement_operand_opt(ch)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            Some(97),
            "test setup should expose a get_constant_box-style IntBound constant",
        );

        let info = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: 1,
            variant: VStringVariant::Plain(VStringPlainInfo {
                _chars: vec![Some(ctx.materialize_operand_at(ch))],
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });

        // vstring.py:178-183 checks `c.is_constant()` directly for Plain
        // strings; a constant IntBound is not enough here.
        assert_eq!(info.get_constant_string_spec(&ctx, 0), None);
    }

    #[test]
    fn test_str_ptr_info_slice_and_concat_dispatch() {
        let mut ctx = OptContext::new(32);
        let b = ctx.materialize_operand_at(OpRef::int_op(10));
        ctx.make_constant_box(&b, Value::Int(97));
        let b = ctx.materialize_operand_at(OpRef::int_op(11));
        ctx.make_constant_box(&b, Value::Int(98));
        let b = ctx.materialize_operand_at(OpRef::int_op(12));
        ctx.make_constant_box(&b, Value::Int(99));
        let b = ctx.materialize_operand_at(OpRef::int_op(20));
        ctx.make_constant_box(&b, Value::Int(1));
        let b = ctx.materialize_operand_at(OpRef::int_op(21));
        ctx.make_constant_box(&b, Value::Int(2));

        let source = OpRef::int_op(1);
        let source_box = ctx.materialize_operand_at(source);
        let source_chars = vec![
            Some(ctx.materialize_operand_at(OpRef::int_op(10))),
            Some(ctx.materialize_operand_at(OpRef::int_op(11))),
            Some(ctx.materialize_operand_at(OpRef::int_op(12))),
        ];
        ctx.set_ptr_info(
            &source_box,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: 3,
                variant: VStringVariant::Plain(VStringPlainInfo {
                    _chars: source_chars,
                }),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let slice = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: Some(ctx.materialize_operand_at(OpRef::int_op(21))), // vstring.py:223: self.lgtop = length
            mode: 0,
            length: -1,
            variant: VStringVariant::Slice(VStringSliceInfo {
                s: ctx.materialize_operand_at(source),
                start: ctx.materialize_operand_at(OpRef::int_op(20)),
                lgtop: ctx.materialize_operand_at(OpRef::int_op(21)),
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        assert_eq!(slice.get_known_str_length(&ctx, 0), Some(2));
        assert_eq!(slice.get_constant_string_spec(&ctx, 0), Some(vec![98, 99]));
        assert_eq!(slice.strgetitem(0, &ctx), Some(OpRef::int_op(11)));

        let concat = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: -1,
            variant: VStringVariant::Concat(VStringConcatInfo {
                vleft: ctx.materialize_operand_at(source),
                vright: ctx.materialize_operand_at(OpRef::int_op(2)),
                _is_virtual: true,
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        let pos2 = ctx.materialize_operand_at(OpRef::int_op(2));
        let pos2_chars = vec![
            Some(ctx.materialize_operand_at(OpRef::int_op(11))),
            Some(ctx.materialize_operand_at(OpRef::int_op(12))),
        ];
        ctx.set_ptr_info(
            &pos2,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: 2,
                variant: VStringVariant::Plain(VStringPlainInfo { _chars: pos2_chars }),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        assert_eq!(concat.get_known_str_length(&ctx, 0), Some(5));
        assert_eq!(
            concat.get_constant_string_spec(&ctx, 0),
            Some(vec![97, 98, 99, 98, 99])
        );
        assert_eq!(concat.strgetitem(3, &ctx), Some(OpRef::int_op(11)));
    }

    #[test]
    fn test_ptr_info_set_getfield() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_obj(descr, None);

        assert!(info.getfield(0).is_none());
        info.setfield(0, field_op(Type::Int, 10));
        assert_eq!(
            info.getfield(0).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(10))
        );
        info.setfield(0, field_op(Type::Int, 20)); // overwrite
        assert_eq!(
            info.getfield(0).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(20))
        );
        info.setfield(1, field_op(Type::Int, 30));
        assert_eq!(
            info.getfield(1).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(30))
        );
    }

    #[test]
    fn test_ptr_info_set_getitem() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_array(descr, 3, false);

        assert_eq!(
            info.getitem(0).and_then(|e| e.as_opref()),
            Some(OpRef::NONE)
        ); // initialized to NONE
        info.setitem(0, field_op(Type::Int, 10));
        assert_eq!(
            info.getitem(0).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(10))
        );
        info.setitem(2, field_op(Type::Int, 30));
        assert_eq!(
            info.getitem(2).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(30))
        );
        assert!(info.getitem(5).is_none()); // out of bounds
    }

    #[test]
    fn test_preamble_item_keeps_regular_array_item_visible() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::array(descr, crate::optimizeopt::intutils::IntBound::nonnegative());
        info.setitem(1, field_op(Type::Int, 77));
        assert_eq!(
            info.getitem(1).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(77))
        );

        let mut replay = Op::new(
            OpCode::GetarrayitemGcI,
            &[
                crate::history::test_support::rooted_resop_operand(Type::Int, 10),
                majit_ir::operand::Operand::from_opref(OpRef::const_int(0)),
            ],
        );
        replay.pos.set(OpRef::int_op(88));
        let pop = PreambleOp {
            op: BoxRef::from_opref(OpRef::int_op(88)),
            invented_name: false,
            preamble_op: std::rc::Rc::new(replay),
            same_as_source: None,
        };
        info.set_preamble_item(1, pop.clone());

        assert!(info.has_preamble_item(1));
        // After set_preamble_item, getitem returns Preamble (not the old Value)
        assert!(info.getitem(1).map_or(false, |e| e.is_preamble()));
        let recovered = info
            .take_preamble_item(1)
            .expect("preamble item should be recoverable");
        assert_eq!(recovered.op.to_opref(), OpRef::int_op(88));
        // After take_preamble_item, slot is Value(NONE)
        assert_eq!(
            info.getitem(1).and_then(|e| e.as_opref()),
            Some(OpRef::NONE)
        );
    }

    #[test]
    fn test_all_items_exposes_preamble_source_box() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::instance(Some(descr), None);
        let replay = Op::new(
            OpCode::GetfieldGcI,
            &[crate::history::test_support::rooted_resop_operand(
                Type::Int,
                10,
            )],
        );
        let pop = PreambleOp {
            op: BoxRef::from_opref(OpRef::int_op(88)),
            invented_name: false,
            preamble_op: std::rc::Rc::new(replay),
            same_as_source: None,
        };
        info.set_preamble_field(3, pop);

        // all_items includes Preamble entries (RPython parity: _fields returns raw)
        let items = info.all_items();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, 3);
        assert!(items[0].1.is_preamble());
    }

    #[test]
    fn test_ptr_info_guard_opcodes() {
        let nonnull = PtrInfo::nonnull();
        let guards = nonnull.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardNonnull));

        let constant = PtrInfo::constant(GcRef(0x1000));
        let guards = constant.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardValue));

        let kc = PtrInfo::known_class(0x2000, true);
        let guards = kc.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardNonnullClass));
    }

    #[test]
    fn test_ptr_info_visitor_walk() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_obj(descr, None);
        info.setfield(0, field_op(Type::Int, 10));
        info.setfield(1, field_op(Type::Int, 20));
        let refs = info.visitor_walk_recursive();
        assert_eq!(refs, vec![OpRef::int_op(10), OpRef::int_op(20)]);
    }

    #[test]
    fn test_opinfo_is_nonnull() {
        assert!(!OpInfo::Unknown.is_nonnull());
        assert!(OpInfo::ptr(PtrInfo::nonnull()).is_nonnull());
    }

    #[test]
    fn test_opinfo_float_const() {
        let info = OpInfo::FloatConstInfo(FloatConstInfo::new(3.14));
        assert!(info.is_constant());
        assert_eq!(info.get_constant_float(), Some(3.14));
    }
}
