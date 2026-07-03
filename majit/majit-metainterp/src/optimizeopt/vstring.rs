#![allow(non_upper_case_globals)]

use majit_ir::operand::Operand;
use majit_ir::{EffectInfo, OopSpecIndex, Op, OpCode, OpRef, Value};

use crate::optimizeopt::info::{PtrInfo, PtrInfoExt, VStringVariant};
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

pub use crate::optimizeopt::info::{
    StrPtrInfo, VStringConcatInfo, VStringPlainInfo, VStringSliceInfo,
};

/// vstring.py:18 MAX_CONST_LEN
const MAX_CONST_LEN: usize = 100;

/// vstring.py mode_string / mode_unicode discriminators.
pub const mode_string: u8 = 0;
pub const mode_unicode: u8 = 1;

/// vstring.py:21 `class StrOrUnicode`.
///
/// The optimizer currently stores the active mode as the compact
/// `mode_string` / `mode_unicode` discriminator above, but the upstream
/// type name belongs to this module and callers should not need to invent
/// another name for mode metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrOrUnicode {
    pub mode: u8,
    pub newstr: OpCode,
    pub strlen: OpCode,
    pub strgetitem: OpCode,
    pub strsetitem: OpCode,
    pub copystrcontent: OpCode,
    pub os_offset: i32,
}

/// vstring.py:371-381 _int_add(optstring, box1, box2)
///
/// Constant-folding INT_ADD: folds add-0 and const+const at the optimizer
/// level. Non-constant adds emit an INT_ADD operation that is re-dispatched
/// from `first_optimization` via `send_extra_operation` (vstring.py:380), so
/// OptIntBounds — a pass BEFORE OptString — computes the result bound.
pub fn _int_add(box1: &Operand, box2: &Operand, ctx: &mut OptContext) -> Operand {
    if let Some(v1) = ctx
        .resolve_operand_operand_opt(box1)
        .and_then(|cb| cb.const_int())
    {
        if v1 == 0 {
            return box2.clone();
        }
        if let Some(v2) = ctx
            .resolve_operand_operand_opt(box2)
            .and_then(|cb| cb.const_int())
        {
            let __c = ctx.emit_constant_int(v1 + v2);
            return ctx.materialize_operand_at(__c);
        }
    } else if ctx
        .resolve_operand_operand_opt(box2)
        .and_then(|cb| cb.const_int())
        == Some(0)
    {
        return box1.clone();
    }
    let arg1 = ctx.resolve_operand_operand(box1);
    let arg2 = ctx.resolve_operand_operand(box2);
    let op = Op::new(OpCode::IntAdd, &[arg1, arg2]);
    // vstring.py:380 optstring.optimizer.send_extra_operation(op).
    let __r = ctx.send_extra_operation(op);
    ctx.materialize_operand_at(__r)
}

/// vstring.py:337-369 copy_str_content(optstring, srcbox, targetbox,
///     srcoffsetbox, offsetbox, lengthbox, mode, need_next_offset=True)
///
/// Emits either inline STRGETITEM/STRSETITEM (for small constant lengths)
/// or a single COPYSTRCONTENT/COPYUNICODECONTENT operation.
pub fn copy_str_content(
    ctx: &mut OptContext,
    srcbox: &Operand,
    targetbox: &Operand,
    srcoffsetbox: &Operand,
    offsetbox: &Operand,
    lengthbox: &Operand,
    mode: u8,
    need_next_offset: bool,
) -> Option<Operand> {
    let srcbox = ctx.resolve_operand_operand(&srcbox);
    let (set_opcode, copy_opcode) = if mode != 0 {
        (OpCode::Unicodesetitem, OpCode::Copyunicodecontent)
    } else {
        (OpCode::Strsetitem, OpCode::Copystrcontent)
    };

    // vstring.py:341-347: determine inline threshold M using intbound
    // A producer-less operand has no forwarded bound; getintbound returns
    // unbounded for it, so resolve-or-unbounded matches the prior
    // materialize_operand_at (mint synthetic → unbounded) behavior without minting.
    let srcoffset_bound = ctx
        .resolve_operand_operand_opt(srcoffsetbox)
        .map(|b| ctx.getintbound_handle(&b).borrow().clone())
        .unwrap_or_else(crate::optimizeopt::intutils::IntBound::unbounded);
    let lgt_bound = ctx
        .resolve_operand_operand_opt(lengthbox)
        .map(|b| ctx.getintbound_handle(&b).borrow().clone())
        .unwrap_or_else(crate::optimizeopt::intutils::IntBound::unbounded);
    // vstring.py:343: isinstance(srcbox, ConstPtr)
    let src_is_const = ctx
        .resolve_operand_operand_opt(&srcbox)
        .as_ref()
        .and_then(|b| ctx.getconst(b))
        .is_some_and(|(_, tp)| tp == majit_ir::Type::Ref);
    let m = if src_is_const && srcoffset_bound.is_constant() {
        5
    } else {
        2
    };

    // vstring.py:347: if lgt.is_constant() and lgt.get_constant_int() <= M
    // Signed comparison: a negative constant length is `<= M`, so it takes the
    // inline path where `range(length)` runs zero times (emitting nothing),
    // matching RPython rather than falling through to a bulk COPYSTRCONTENT.
    if lgt_bound.is_constant() {
        let length = lgt_bound.get_constant_int();
        if length <= m as i64 {
            // vstring.py:350-357: inline STRGETITEM/STRSETITEM
            // RPython calls optstring.strgetitem(None, srcbox, srcoffsetbox, mode)
            // which tries PtrInfo lookup first (virtual chars), falling back to
            // emitting STRGETITEM.
            let mut src_offset = srcoffsetbox.clone();
            let mut dst_offset = offsetbox.clone();
            let one = {
                let __one = ctx.emit_constant_int(1);
                ctx.materialize_operand_at(__one)
            };
            for _i in 0..length {
                // vstring.py:350-351: charbox = optstring.strgetitem(None,
                // srcbox, srcoffsetbox, mode). OptString is a ZST, so a local
                // instance routes the read through the shared strgetitem
                // (make_nonnull_str, slice rebase, concat recursion, virtual /
                // ConstPtr fold, residual) instead of reimplementing it here.
                let charbox = {
                    let ch = OptString.strgetitem_emit_box(&srcbox, &src_offset, mode, ctx);
                    ctx.materialize_operand_at(ch)
                };
                src_offset = _int_add(&src_offset, &one, ctx);
                let arg_target = ctx.resolve_operand_operand(&targetbox);
                let arg_dst_off = ctx.resolve_operand_operand(&dst_offset);
                let arg_char = ctx.resolve_operand_operand(&charbox);
                let setitem_op = Op::new(
                    set_opcode,
                    &[arg_target.clone(), arg_dst_off.clone(), arg_char.clone()],
                );
                ctx.emit_for_force(setitem_op);
                dst_offset = _int_add(&dst_offset, &one, ctx);
            }
            // vstring.py:369: the inline path returns the incremented
            // offsetbox unconditionally (need_next_offset only gates the
            // bulk path below).
            return Some(dst_offset);
        }
    }

    // vstring.py:359-368: bulk COPYSTRCONTENT
    // vstring.py:360-363: nextoffsetbox = _int_add(...) if need_next_offset
    // else None — the caller that passes need_next_offset=False discards it.
    let next_offset = if need_next_offset {
        Some(_int_add(offsetbox, lengthbox, ctx))
    } else {
        None
    };
    let arg_src = ctx.resolve_operand_operand(&srcbox);
    let arg_target = ctx.resolve_operand_operand(&targetbox);
    let arg_srcoff = ctx.resolve_operand_operand(&srcoffsetbox);
    let arg_off = ctx.resolve_operand_operand(&offsetbox);
    let arg_len = ctx.resolve_operand_operand(&lengthbox);
    let copy_op = Op::new(
        copy_opcode,
        &[
            arg_src.clone(),
            arg_target.clone(),
            arg_srcoff.clone(),
            arg_off.clone(),
            arg_len.clone(),
        ],
    );
    ctx.emit_for_force(copy_op);
    next_offset
}

/// vstring.py:132-140 / 185-205 / 230-233 / 309-317
/// string_copy_parts — recursive dispatch to copy string content
/// into an already-allocated target string at `offsetbox`.
/// Returns the updated offset after the copy.
///
/// This is the Rust equivalent of RPython's per-subclass
/// `string_copy_parts` / `initialize_forced_string` polymorphic dispatch.
pub fn string_copy_parts(
    opref: &Operand,
    targetbox: &Operand,
    offsetbox: &Operand,
    mode: u8,
    ctx: &mut OptContext,
) -> Operand {
    // Extract variant data without keeping PtrInfo borrow alive.
    // RPython dispatches via subclass; we dispatch via enum variant.
    enum Action {
        /// vstring.py:194-205 VStringPlainInfo.initialize_forced_string
        Plain(Vec<Option<Operand>>),
        /// vstring.py:230-233 VStringSliceInfo.string_copy_parts
        Slice {
            s: Operand,
            start: Operand,
            lgtop: Operand,
        },
        /// vstring.py:309-317 VStringConcatInfo.string_copy_parts
        Concat { vleft: Operand, vright: Operand },
        /// vstring.py:132-140 StrPtrInfo.string_copy_parts (base class, non-virtual)
        NonVirtual,
    }

    let resolved_box = ctx.resolve_operand_operand_opt(opref);
    let action = match resolved_box.as_ref().and_then(|b| ctx.getptrinfo(b)) {
        Some(info) => match info {
            PtrInfo::Str(sinfo) if sinfo.is_virtual() => match &sinfo.variant {
                VStringVariant::Plain(p) => {
                    Action::Plain(p._chars.iter().map(|slot| slot.clone()).collect())
                }
                VStringVariant::Slice(s) => Action::Slice {
                    s: s.s.clone(),
                    start: s.start.clone(),
                    lgtop: s.lgtop.clone(),
                },
                VStringVariant::Concat(c) => Action::Concat {
                    vleft: c.vleft.clone(),
                    vright: c.vright.clone(),
                },
                VStringVariant::Ptr => Action::NonVirtual,
            },
            _ => Action::NonVirtual,
        },
        None => Action::NonVirtual,
    };

    let set_opcode = if mode != 0 {
        OpCode::Unicodesetitem
    } else {
        OpCode::Strsetitem
    };

    match action {
        Action::Plain(chars) => {
            // vstring.py:194-205 VStringPlainInfo.initialize_forced_string
            let mut offset = offsetbox.clone();
            let one = {
                let __one = ctx.emit_constant_int(1);
                ctx.materialize_operand_at(__one)
            };
            for ch in &chars {
                if let Some(ch_ref) = ch {
                    let arg_char = ctx.resolve_operand_operand(ch_ref);
                    let arg_target = ctx.resolve_operand_operand(&targetbox);
                    let arg_offset = ctx.resolve_operand_operand(&offset);
                    let setitem_op = Op::new(
                        set_opcode,
                        &[arg_target.clone(), arg_offset.clone(), arg_char.clone()],
                    );
                    ctx.emit_for_force(setitem_op);
                }
                offset = _int_add(&offset, &one, ctx);
            }
            offset
        }
        Action::Slice { s, start, lgtop } => {
            // vstring.py:230-233 VStringSliceInfo.string_copy_parts
            copy_str_content(ctx, &s, targetbox, &start, offsetbox, &lgtop, mode, true)
                .expect("need_next_offset=true always returns the offset")
        }
        Action::Concat { vleft, vright } => {
            // vstring.py:309-317 VStringConcatInfo.string_copy_parts
            let offset = string_copy_parts(&vleft, targetbox, offsetbox, mode, ctx);
            string_copy_parts(&vright, targetbox, &offset, mode, ctx)
        }
        Action::NonVirtual => {
            // vstring.py:132-140 StrPtrInfo.string_copy_parts (base class)
            // lengthbox = self.getstrlen(op, optstring, mode)
            // srcbox = self.force_box(op, optstring)  -- no-op for non-virtual
            let lengthbox = ctx.getstrlen_opref(opref.to_opref(), mode);
            let lengthbox = ctx.materialize_operand_at(lengthbox);
            let srcbox = force_child_for_string(opref, ctx);
            let zero = {
                let __zero = ctx.emit_constant_int(0);
                ctx.materialize_operand_at(__zero)
            };
            copy_str_content(
                ctx, &srcbox, targetbox, &zero, offsetbox, &lengthbox, mode, true,
            )
            .expect("need_next_offset=true always returns the offset")
        }
    }
}

/// Force a string-typed OpRef if it's virtual. Used by string_copy_parts
/// base class path (vstring.py:138: srcbox = self.force_box(op, optstring)).
fn force_child_for_string(opref: &Operand, ctx: &mut OptContext) -> Operand {
    // One chain walk; the position view falls back to the source.
    let resolved_box = ctx.resolve_operand_operand_opt(opref);
    let resolved = resolved_box
        .as_ref()
        .map_or_else(|| opref.clone(), |b| b.clone());
    if resolved_box.as_ref().map_or(false, |b| ctx.is_virtual(b)) {
        let resolved_box = resolved_box.expect("recorder-populated");
        let mut info = ctx.take_ptr_info(&resolved_box).unwrap();
        let forced = info.force_box(&resolved_box, ctx);
        let forced_box = ctx.materialize_operand_at(forced);
        return ctx.resolve_operand_operand(&forced_box);
    }
    resolved
}

/// vstring.py:413 `class OptString(Optimization)` — stateless string/unicode
/// optimization pass. All per-string state lives on `PtrInfo::Str`
/// (length, mode, virtual variant, lgtop). STRLEN/UNICODELEN caching is
/// wired through `OptPure` via `pure_from_args1`.
pub struct OptString;

impl OptString {
    pub fn new() -> Self {
        OptString
    }

    fn get_plain_info(&self, op: &Operand, ctx: &OptContext) -> Option<VStringPlainInfo> {
        match ctx.peek_ptr_info(op) {
            Some(PtrInfo::Str(sinfo)) => match sinfo.variant {
                VStringVariant::Plain(info) => Some(info),
                _ => None,
            },
            _ => None,
        }
    }

    /// Run `f` against the VStringPlainInfo of `op`, auto-mirroring the
    /// operand snapshot after mutation via OptContext::with_ptr_info_mut.
    /// Returns `None` if the box has no PtrInfo, is not Str, or its variant
    /// is not Plain.
    fn with_plain_info_mut<R>(
        &self,
        op: &Operand,
        ctx: &mut OptContext,
        f: impl FnOnce(&mut VStringPlainInfo) -> R,
    ) -> Option<R> {
        ctx.with_ptr_info_mut(op, |info| {
            if let PtrInfo::Str(sinfo) = info {
                if let VStringVariant::Plain(plain) = &mut sinfo.variant {
                    return Some(f(plain));
                }
            }
            None
        })
        .flatten()
    }

    fn is_virtual_plain(&self, op: &Operand, ctx: &OptContext) -> bool {
        self.get_plain_info(op, ctx).is_some()
    }

    fn get_concat_info(&self, op: &Operand, ctx: &OptContext) -> Option<VStringConcatInfo> {
        let resolved_box = ctx.resolve_operand_operand_opt(op);
        match resolved_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
            Some(PtrInfo::Str(sinfo)) => match sinfo.variant {
                VStringVariant::Concat(info) => Some(info),
                _ => None,
            },
            _ => None,
        }
    }

    fn is_virtual_concat(&self, op: &Operand, ctx: &OptContext) -> bool {
        self.get_concat_info(op, ctx).is_some()
    }

    fn get_slice_info(&self, op: &Operand, ctx: &OptContext) -> Option<VStringSliceInfo> {
        match ctx.peek_ptr_info(op) {
            Some(PtrInfo::Str(sinfo)) => match sinfo.variant {
                VStringVariant::Slice(info) => Some(info),
                _ => None,
            },
            _ => None,
        }
    }

    fn is_virtual_slice(&self, op: &Operand, ctx: &OptContext) -> bool {
        self.get_slice_info(op, ctx).is_some()
    }

    /// vstring.py: read the string mode (0 = byte string, 1 = unicode) from
    /// the installed `StrPtrInfo`. Returns 0 when no PtrInfo is set — callers
    /// inside the pass only hit this path for constant/forwarded refs where
    /// the mode is not observable and defaulting to string is harmless.
    fn get_mode(&self, op: &Operand, ctx: &OptContext) -> u8 {
        let resolved_box = ctx.resolve_operand_operand_opt(op);
        match resolved_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
            Some(PtrInfo::Str(sinfo)) => sinfo.mode,
            _ => 0,
        }
    }

    /// vstring.py:76-103 StrPtrInfo.force_box — delegate to PtrInfo::force_box.
    fn force_box(&mut self, op: &Operand, ctx: &mut OptContext) -> OpRef {
        // One chain walk; the position view falls back to the source.
        let resolved_box = ctx.resolve_operand_operand_opt(op);
        let resolved = resolved_box
            .as_ref()
            .map_or_else(|| op.to_opref(), |b| b.to_opref());
        if resolved_box.as_ref().map_or(false, |b| ctx.is_virtual(b)) {
            let resolved_box = resolved_box.expect("recorder-populated");
            let mut info = ctx.take_ptr_info(&resolved_box).unwrap();
            let forced = info.force_box(&resolved_box, ctx);
            return ctx.get_replacement_opref(forced);
        }
        resolved
    }

    /// Emit a SameAsI op that produces a constant integer value.
    ///
    /// We need a way to reference constant values as OpRefs. We emit a
    /// SameAsI(dummy) and record the constant in the context.
    fn emit_constant_int(&self, value: i64, ctx: &mut OptContext) -> OpRef {
        // Emit a dummy SameAsI to get an OpRef, then record the constant.
        let op = Op::new(OpCode::SameAsI, &[Operand::none()]);
        let opref = ctx.emit(op);
        let b = ctx.materialize_operand_at(opref);
        ctx.make_constant_box(&b, Value::Int(value));
        opref
    }

    /// vstring.py:110-119 StrPtrInfo.getstrlen — delegates to
    /// OptContext::getstrlen_opref which handles per-variant dispatch
    /// and lgtop caching (box identity reuse).
    fn getstrlen(&self, op: &Operand, ctx: &mut OptContext) -> OpRef {
        let mode = self.get_mode(op, ctx);
        ctx.getstrlen_opref(op.to_opref(), mode)
    }

    /// vstring.py:112-114 — get the strlen OpRef if already known,
    /// without emitting a new op. Checks lgtop first (RPython parity),
    /// then structurally-known constant length on the virtual variant.
    fn getstrlen_if_known(&self, op: &Operand, ctx: &mut OptContext) -> Option<OpRef> {
        let resolved_box = ctx.resolve_operand_operand_opt(op);
        // vstring.py:112: if self.lgtop is not None: return self.lgtop
        if let Some(info) = resolved_box.as_ref().and_then(|b| ctx.getptrinfo(b)) {
            if let Some(lgtop) = info.get_cached_lgtop() {
                return Some(lgtop);
            }
        }
        // vstring.py:174: self.lgtop = ConstInt(len(self._chars))
        // RPython creates a pure ConstInt — no op emission.
        let known_len = resolved_box
            .as_ref()
            .and_then(|b| ctx.getptrinfo(b))
            .and_then(|info| {
                let mode = self.get_mode(op, ctx);
                info.get_known_str_length(ctx, mode)
            });
        if let Some(len) = known_len {
            let len_opref = ctx.make_constant_int(len);
            // Cache in lgtop for identity reuse. `resolved_box` is Some here
            // (known_len required its ptr_info), so reuse it instead of
            // re-resolving (vstring.py:117/174/293).
            if let Some(b) = &resolved_box {
                ctx.set_str_lgtop(b, len_opref);
            }
            return Some(len_opref);
        }
        None
    }

    /// vstring.py:486-517 OptString.strgetitem
    ///
    /// `mode` is threaded from the caller (mode_string / mode_unicode) rather
    /// than recovered from the receiver's PtrInfo, so make_nonnull_str installs
    /// the correct string flavour even on a not-yet-typed receiver.
    ///
    /// Tries virtual dispatch (Plain/Slice/Concat), then the ConstPtr fold of
    /// `_strgetitem`. Returns None when the char is not statically known: the
    /// `_optimize_STRGETITEM` dispatcher keeps the op in that case (resbox=op),
    /// while the string-compare callers route through `strgetitem_emit`.
    fn strgetitem(&self, s: &Operand, index: i64, mode: u8, ctx: &mut OptContext) -> Option<OpRef> {
        let resolved_box = ctx.resolve_operand_operand_opt(s);
        // vstring.py:487: self.make_nonnull_str(s, mode) — ensure the receiver
        // carries a (nonnull) string PtrInfo before it is read. A no-op when the
        // box is constant or already a virtual string; otherwise it installs the
        // non-virtual StrPtrInfo{Ptr} so nonnull-ness reaches later passes.
        if let Some(b) = &resolved_box {
            ctx.make_nonnull_str(b, mode);
        }
        // vstring.py:488-503: sinfo = getptrinfo(s). Virtual dispatch:
        // PtrInfo::Str → VStringInfo.strgetitem (Plain/Slice/Concat); Ptr → None.
        let from_virtual = resolved_box
            .as_ref()
            .and_then(|b| ctx.getptrinfo(b))
            .and_then(|info| info.strgetitem(index, &*ctx));
        if from_virtual.is_some() {
            return from_virtual;
        }
        // vstring.py:398-407 _strgetitem: isinstance(strbox, ConstPtr)
        match resolved_box.as_ref().and_then(|b| ctx.getconst(b)) {
            Some((raw, majit_ir::Type::Ref)) if raw != 0 => {
                let r = majit_ir::GcRef(raw as usize);
                let ch_val = ctx
                    .string_content_resolver
                    .as_deref()
                    .and_then(|resolver| resolver(r, mode))
                    .and_then(|chars| chars.get(index as usize).copied())?;
                Some(ctx.emit_constant_int(ch_val))
            }
            _ => None,
        }
    }

    /// vstring.py:486-517 strgetitem invoked with resbox=None — the
    /// string-compare and inline-copy callers. Resolves a virtual/constant char
    /// via `strgetitem`, otherwise emits a residual STRGETITEM. Always returns a
    /// box, matching `strgetitem(None, ...)`.
    fn strgetitem_emit(&self, s: &Operand, index: i64, mode: u8, ctx: &mut OptContext) -> OpRef {
        if let Some(r) = self.strgetitem(s, index, mode, ctx) {
            return r;
        }
        // vstring.py:490-493: a virtual slice's residual STRGETITEM reads the
        // source string at `_int_add(slice.start, index)`, not the slice box —
        // emitting STRGETITEM(slice, index) would force the slice instead. The
        // index becomes a box: a non-constant INT_ADD when the slice start is
        // not constant, and `start + 0` collapses back to `start`.
        let resolved_s = ctx.resolve_operand_operand(s);
        let index_const = ctx.make_constant_int(index);
        let index_const_box = ctx.materialize_operand_at(index_const);
        let (strbox, index_box) = if let Some(slice) = self.get_slice_info(&resolved_s, ctx) {
            let index_box = _int_add(&slice.start, &index_const_box, ctx);
            (ctx.resolve_operand_operand(&slice.s), index_box)
        } else {
            (resolved_s, index_const_box)
        };
        // vstring.py:505-512: a virtual concat with a constant index recurses
        // into the child holding that position, so the residual STRGETITEM
        // reads that child (forcing only it) rather than the whole concat. The
        // top `strgetitem` already folded the statically-known case; this only
        // fires when the child's char is a variable but the left length is a
        // known constant.
        if let Some(idx) = ctx
            .resolve_operand_operand_opt(&index_box)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(concat) = self.get_concat_info(&strbox, ctx) {
                // vstring.py:506-507: len1box = leftinfo.getstrlen(...); recurse
                // only when it is an actual ConstInt. A non-constant length —
                // including a variable whose IntBound merely happens to be
                // constant — is not `isinstance(len1box, ConstInt)`, so it falls
                // through to the residual STRGETITEM on the whole concat.
                let len1box = ctx.getstrlen_opref(concat.vleft.to_opref(), mode);
                if let Some(len1) = ctx.isinstance_const_int(len1box) {
                    return if idx < len1 {
                        self.strgetitem_emit(&concat.vleft, idx, mode, ctx)
                    } else {
                        self.strgetitem_emit(&concat.vright, idx - len1, mode, ctx)
                    };
                }
            }
        }
        self._strgetitem(&strbox, &index_box, mode, ctx)
    }

    /// vstring.py:411-415 _strgetitem residual emission — `strgetitem` already
    /// folded a ConstPtr read, so this only emits the STRGETITEM op (the
    /// resbox=None branch) with the resolved string and index boxes, and routes
    /// it through `emit_extra` to the downstream passes.
    fn _strgetitem(
        &self,
        strbox: &Operand,
        index_box: &Operand,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OpRef {
        let get_opcode = if mode == mode_unicode {
            OpCode::Unicodegetitem
        } else {
            OpCode::Strgetitem
        };
        let arg_str = ctx.resolve_operand_operand(&strbox);
        let arg_index = ctx.resolve_operand_operand(&index_box);
        // vstring.py:411-415 emit_extra(resbox). emit_for_force routes to
        // emit() during forcing (in_final_emission, the copy_str_content path)
        // and to emit_extra(current_pass_idx) during the pass — identical to
        // the previous emit_extra(current_pass_idx) for the pass-time
        // string-compare / dispatcher callers.
        ctx.emit_for_force(Op::new(get_opcode, &[arg_str.clone(), arg_index.clone()]))
    }

    /// vstring.py:486-517 strgetitem(None, s, index, mode) with a box-valued
    /// index — the copy_str_content reader. A constant index folds through the
    /// `strgetitem_emit` path (static virtual/ConstPtr fold, slice rebase,
    /// concat recursion); a non-constant index rebases a virtual slice to its
    /// source at `start + index` (vstring.py:490-493) and residualizes
    /// (vstring.py:495: vindex non-constant, so the Plain/Concat folds are
    /// skipped).
    fn strgetitem_emit_box(
        &self,
        s: &Operand,
        index: &Operand,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OpRef {
        if let Some(idx) = ctx
            .resolve_operand_operand_opt(index)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            return self.strgetitem_emit(s, idx, mode, ctx);
        }
        // vstring.py:487: make_nonnull_str(s, mode)
        let resolved_s = ctx.resolve_operand_operand(s);
        ctx.make_nonnull_str(&resolved_s, mode);
        let (strbox, index_box) = if let Some(slice) = self.get_slice_info(&resolved_s, ctx) {
            let new_index = _int_add(&slice.start, index, ctx);
            (ctx.resolve_operand_operand(&slice.s), new_index)
        } else {
            (resolved_s, index.clone())
        };
        self._strgetitem(&strbox, &index_box, mode, ctx)
    }

    /// Get the known length of a virtual string as a constant, if available.
    /// Delegates to `PtrInfo::Str::getstrlen` which walks Plain/Slice/Concat
    /// variants. Matches vstring.py:171/251/281 `getstrlen()` per-variant.
    fn get_known_length(&self, op: &Operand, ctx: &OptContext) -> Option<i64> {
        let resolved_box = ctx.resolve_operand_operand_opt(op);
        let info = resolved_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
        let mode = self.get_mode(op, ctx);
        info.get_known_str_length(ctx, mode)
    }

    /// vstring.py:440-453 `_optimize_NEWSTR(self, op, mode)`. Virtualize if
    /// length is a small constant; otherwise install a non-virtual StrPtrInfo
    /// and emit the op. `postprocess_NEWSTR` (vstring.py:455-459) registers
    /// `pure(STRLEN, op) = length_arg` for CSE via the pure cache.
    fn _optimize_newstr(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let len_ref = op.arg(0).to_opref();
        if let Some(len) = ctx
            .resolve_operand_operand_opt(&op.arg(0))
            .and_then(|b_| ctx.get_constant_int_box(&b_))
        {
            if len >= 0 && (len as usize) <= MAX_CONST_LEN {
                // vstring.py:450: self.make_vstring_plain(op, mode, length)
                let b = Operand::from_bound_op(op_rc);
                {
                    ctx.set_ptr_info(
                        &b,
                        PtrInfo::Str(StrPtrInfo {
                            lenbound: None,
                            lgtop: None,
                            mode,
                            length: len as i32,
                            variant: VStringVariant::Plain(VStringPlainInfo {
                                _chars: vec![None; len as usize],
                            }),
                            last_guard_pos: -1,
                            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                        }),
                    );
                }
                return OptimizationResult::Remove;
            }
        }
        // vstring.py:452: self.make_nonnull_str(op, mode); return self.emit(op)
        // NEWSTR/NEWUNICODE produce the string object directly; use the
        // bound result box as the PtrInfo host.
        let op_box = Operand::from_bound_op(op_rc);
        ctx.make_nonnull_str(&op_box, mode);
        // vstring.py:455-459 postprocess_NEWSTR / postprocess_NEWUNICODE:
        //   self.pure_from_args1(mode.STRLEN, op, op.getarg(0))
        let strlen_opcode = if mode == 1 {
            OpCode::Unicodelen
        } else {
            OpCode::Strlen
        };
        ctx.register_pure_from_args1(strlen_opcode, op.pos.get(), len_ref);
        OptimizationResult::PassOn
    }

    /// Handle STRSETITEM: if target is virtual Plain and index is constant, track.
    fn optimize_strsetitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let str_ref = ctx.resolve_operand_operand(&op.arg(0));
        let char_ref = op.arg(2).to_opref();
        let char_resolved = ctx.get_replacement_opref(char_ref);

        if let Some(idx) = ctx
            .resolve_operand_operand_opt(&op.arg(1))
            .and_then(|b_| ctx.get_constant_int_box(&b_))
        {
            let i = idx as usize;
            // Materialize the char position before borrowing the plain info so a
            // bound producer is stored, not a position-only box. `with_plain_info_mut`
            // holds `ctx`, so the materialize must precede it.
            let char_operand = ctx.materialize_operand_at(char_resolved);
            let did_write = self
                .with_plain_info_mut(&str_ref, ctx, |info| {
                    if i < info._chars.len() {
                        info._chars[i] = Some(char_operand.clone());
                        return true;
                    }
                    false
                })
                .unwrap_or(false);
            if did_write {
                return OptimizationResult::Remove;
            }
        }
        // Not virtual or index not constant -> force and emit.
        self.force_if_virtual(&str_ref, ctx);
        OptimizationResult::PassOn
    }

    /// Handle STRGETITEM: if source is virtual, resolve the character.
    fn optimize_strgetitem(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let str_ref = ctx.resolve_operand_operand(&op.arg(0));

        if let Some(idx) = ctx
            .resolve_operand_operand_opt(&op.arg(1))
            .and_then(|b_| ctx.get_constant_int_box(&b_))
        {
            if let Some(ch_ref) = self.strgetitem(&str_ref, idx, mode, ctx) {
                let b_old = Operand::from_bound_op(op_rc);
                let b_new = ctx.get_box_replacement_operand(ch_ref);
                ctx.make_equal_to(&b_old, &b_new);
                return OptimizationResult::Remove;
            }
        }
        // vstring.py:490-512: the char could not be folded. Continue the
        // strgetitem dispatch on the residual — unwrap a virtual slice to its
        // source (index → start+index) and recurse a virtual concat into the
        // child that holds the position — then emit STRGETITEM on that target,
        // forcing only the target. The slice/concat is left unreferenced rather
        // than forced wholesale.
        if let Some((target, index_box)) =
            self.strgetitem_rebase_residual(&str_ref, &op.arg(1), mode, ctx)
        {
            // PRE-EXISTING DIVERGENCE: vstring.py:404 `_strgetitem` only builds
            // the STRGETITEM and hands it to emit_extra; the operand is forced
            // later at final emission (optimizer.py:650 force_box on args). pyre
            // has no emit-time operand forcing for general ops, so the target is
            // materialized here, ahead of upstream's timing. Convergence needs
            // force_box to run over an emitted op's args at emission time.
            self.force_if_virtual(&target, ctx);
            let arg_s = ctx.resolve_operand_operand(&target);
            let arg_i = ctx.resolve_operand_operand(&index_box);
            let get_opcode = if mode == mode_unicode {
                OpCode::Unicodegetitem
            } else {
                OpCode::Strgetitem
            };
            let mut getitem = Op::new(get_opcode, &[arg_s.clone(), arg_i.clone()]);
            getitem.pos.set(op.pos.get());
            // vstring.py:407-409 `_strgetitem`: resbox = replace_op_with(resbox,
            // STRGETITEM, ...); emit_extra(resbox). emit_extra =
            // send_extra_operation(op, self.next_optimization) — the rewritten
            // op flows through the passes AFTER OptString (OptPure/OptHeap), not
            // a final Emit. pyre's Replace re-runs current_op through the rest
            // of the chain and emits at the end.
            return OptimizationResult::Replace(getitem);
        }
        // Plain / non-virtual: keep the op and force the string.
        self.force_if_virtual(&str_ref, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:490-512: resolve the residual STRGETITEM target after a
    /// static fold miss, mirroring the rest of `strgetitem` past the Plain
    /// fold. A virtual slice rebases `index → start+index`, `s → source`, then
    /// the dispatch CONTINUES on the rebased string (vstring.py:494 onward); a
    /// virtual concat with a constant index recurses into the child holding
    /// that position (vstring.py:505-512). Returns `Some((string, index))` when
    /// rebasing moved the target off `s`, and `None` when `s` is a plain /
    /// non-virtual string the residual should read directly (the caller then
    /// keeps the op and forces it).
    fn strgetitem_rebase_residual(
        &self,
        s: &Operand,
        index: &Operand,
        mode: u8,
        ctx: &mut OptContext,
    ) -> Option<(Operand, Operand)> {
        let resolved_s = ctx.resolve_operand_operand(s);
        // vstring.py:490-493: slice → rebase to source, then continue dispatch.
        if let Some(slice) = self.get_slice_info(&resolved_s, ctx) {
            let new_index = _int_add(&slice.start, index, ctx);
            let source = ctx.resolve_operand_operand(&slice.s);
            return Some(
                self.strgetitem_rebase_residual(&source, &new_index, mode, ctx)
                    .unwrap_or((source, new_index)),
            );
        }
        // vstring.py:505-512: virtual concat + constant index → recurse into
        // the child holding the position (vleft if index < len1, else vright at
        // index - len1).
        if let Some(idx) = ctx
            .resolve_operand_operand_opt(index)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(concat) = self.get_concat_info(&resolved_s, ctx) {
                // vstring.py:506-507: recurse only when getstrlen is an actual
                // ConstInt, matching strgetitem_emit's concat gate; a constant
                // IntBound on a non-ConstInt length is not enough.
                let len1box = ctx.getstrlen_opref(concat.vleft.to_opref(), mode);
                if let Some(len1) = ctx.isinstance_const_int(len1box) {
                    let (child, child_idx) = if idx < len1 {
                        (concat.vleft.clone(), idx)
                    } else {
                        (concat.vright.clone(), idx - len1)
                    };
                    let child_idx_box = {
                        let c = ctx.make_constant_int(child_idx);
                        ctx.materialize_operand_at(c)
                    };
                    let child_resolved = ctx.resolve_operand_operand(&child);
                    return Some(
                        self.strgetitem_rebase_residual(&child_resolved, &child_idx_box, mode, ctx)
                            .unwrap_or((child_resolved, child_idx_box)),
                    );
                }
            }
        }
        None
    }

    /// vstring.py:525-533 _optimize_STRLEN
    fn optimize_strlen(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let mode = if op.opcode == OpCode::Unicodelen {
            1u8
        } else {
            0u8
        };
        // vstring.py:526-527
        let has_info = ctx
            .getptrinfo(&op.arg(0).get_box_replacement(false))
            .is_some();
        if has_info {
            // vstring.py:529: lgtop = opinfo.getstrlen(arg1, self, mode)
            let lgtop = ctx.getstrlen_opref(op.arg(0).to_opref(), mode);
            // vstring.py:531: self.make_equal_to(op, lgtop)
            let b_old = Operand::from_bound_op(op_rc);
            let b_lgtop = ctx.get_box_replacement_operand(lgtop);
            ctx.make_equal_to(&b_old, &b_lgtop);
            return OptimizationResult::Remove;
        }
        // vstring.py:533: return self.emit(op)
        OptimizationResult::PassOn
    }

    fn get_constant_int_bound(&self, op: &Operand, ctx: &OptContext) -> Option<i64> {
        // optimizer.py:99-113 getintbound resolves `op = get_box_replacement(op)`
        // before reading the bound. Route through `resolve_box_box` (not the raw
        // box-native walk) so a non-canonical InputArg operand reaches its
        // canonical slot (mod.rs:4337) instead of missing the recorded bound.
        let op = ctx.resolve_operand_operand(op);
        ctx.peek_intbound_box(&op)
            .filter(|bound| bound.is_constant())
            .map(|bound| bound.get_constant_int())
            .or_else(|| ctx.get_constant_int_box(&op))
    }

    /// vstring.py:556-589 _optimize_COPYSTRCONTENT
    fn optimize_copystrcontent(
        &mut self,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // copystrcontent(src, dst, src_start, dst_start, length)
        let src_ref_box = ctx.resolve_operand_operand(&op.arg(0));
        let dst_ref = ctx.resolve_operand_operand(&op.arg(1));
        let src_info = ctx.getptrinfo(&src_ref_box);
        let src_is_virtual_or_constant = src_info
            .as_ref()
            .is_some_and(|info| info.is_virtual() || info.is_constant());
        // vstring.py:564/568: dst = getptrinfo(op.getarg(1)) resolves once;
        // reuse the resolved box instead of re-walking the operand chain.
        let dst_virtual = self.is_virtual_plain(&dst_ref, ctx);
        let src_start = self.get_constant_int_bound(&op.arg(2), ctx);
        let dst_start = self.get_constant_int_bound(&op.arg(3), ctx);
        let length = self.get_constant_int_bound(&op.arg(4), ctx);

        if length == Some(0) {
            return OptimizationResult::Remove;
        }

        if let (Some(src_start), Some(dst_start), Some(length)) = (src_start, dst_start, length) {
            if src_is_virtual_or_constant
                && (length < 20 || (src_is_virtual_or_constant && dst_virtual))
            {
                let setitem_opcode = if mode == mode_unicode {
                    OpCode::Unicodesetitem
                } else {
                    OpCode::Strsetitem
                };
                // vstring.py:578: `for index in range(actual_length)` — a
                // negative constant length runs zero iterations (the copy is a
                // no-op), so clamp the capacity hint to avoid `-n as usize`.
                let mut dst_chars = Vec::with_capacity(length.max(0) as usize);
                for index in 0..length {
                    // vstring.py:580-581: vresult = self.strgetitem(None, ...) —
                    // const-folds a virtual/constant char or emits a STRGETITEM.
                    let ch_ref = self.strgetitem_emit(&src_ref_box, src_start + index, mode, ctx);
                    let char_ref = ctx.get_replacement_opref(ch_ref);
                    if dst_virtual {
                        dst_chars.push(Some(char_ref));
                    } else {
                        // vstring.py:585-589: self.emit_extra(new_op)
                        let dst_index_ref = ctx.make_constant_int(dst_start + index);
                        let pass_idx = ctx.current_pass_idx;
                        let arg_dst = ctx.materialize_operand_at(dst_ref.to_opref());
                        let arg_dst_index = ctx.materialize_operand_at(dst_index_ref);
                        let arg_char = ctx.materialize_operand_at(char_ref);
                        ctx.emit_extra(
                            pass_idx,
                            Op::new(
                                setitem_opcode,
                                &[arg_dst.clone(), arg_dst_index.clone(), arg_char.clone()],
                            ),
                        );
                    }
                }
                if dst_virtual {
                    // Materialize each char position to a bound producer before
                    // borrowing the plain info (the closure holds `ctx`).
                    let dst_operands: Vec<Option<Operand>> = dst_chars
                        .into_iter()
                        .map(|o| o.map(|r| ctx.materialize_operand_at(r)))
                        .collect();
                    self.with_plain_info_mut(&dst_ref, ctx, |info| {
                        for (index, ch_op) in dst_operands.into_iter().enumerate() {
                            let dst_index = (dst_start as usize) + index;
                            if dst_index < info._chars.len() {
                                info._chars[dst_index] = ch_op;
                            }
                        }
                    });
                }
                return OptimizationResult::Remove;
            }
        }

        // vstring.py:590-593: fallback — emit via copy_str_content
        // which may still inline small constant-length copies.
        copy_str_content(
            ctx,
            &op.arg(0),
            &op.arg(1),
            &op.arg(2),
            &op.arg(3),
            &op.arg(4),
            mode,
            false, // need_next_offset=False
        );
        OptimizationResult::Remove
    }

    /// Force a string if it is virtual.
    fn force_if_virtual(&mut self, op: &Operand, ctx: &mut OptContext) {
        if self.is_virtual(op, ctx) {
            self.force_box(op, ctx);
        }
    }

    /// Check if a box references a virtual string (after forwarding).
    #[allow(dead_code)]
    fn is_virtual(&self, op: &Operand, ctx: &OptContext) -> bool {
        ctx.is_virtual(op)
    }

    /// vstring.py:383-391 _int_sub — constant-fold if both args are constant,
    /// otherwise emit INT_SUB so downstream passes (int bounds, CSE) see it.
    ///
    /// PRE-EXISTING DIVERGENCE: vstring.py:389 uses
    /// `optstring.optimizer.send_extra_operation(op)` (re-dispatch from
    /// first_optimization); `emit_for_force` only routes from the next pass.
    /// Same convergence note as the sibling `_int_add`.
    fn int_sub(&self, a: &Operand, b: &Operand, ctx: &mut OptContext) -> Operand {
        if let Some(vb) = ctx
            .resolve_operand_operand_opt(b)
            .and_then(|cb| cb.const_int())
        {
            if vb == 0 {
                return a.clone();
            }
            if let Some(va) = ctx
                .resolve_operand_operand_opt(a)
                .and_then(|cb| cb.const_int())
            {
                let __c = self.emit_constant_int(va - vb, ctx);
                return ctx.materialize_operand_at(__c);
            }
        }
        let arg_a = ctx.resolve_operand_operand(&a);
        let arg_b = ctx.resolve_operand_operand(&b);
        let op = Op::new(OpCode::IntSub, &[arg_a.clone(), arg_b.clone()]);
        let __r = ctx.emit_for_force(op);
        ctx.materialize_operand_at(__r)
    }

    /// vstring.py: postprocess — after STRLEN on a known-length string,
    /// record as pure (for CSE with OptPure).
    fn postprocess_strlen(&self, op: &Op, ctx: &mut OptContext) {
        // vstring.py: postprocess_STRLEN → make_nonnull_str
        let mode = if op.opcode == OpCode::Strlen {
            0u8
        } else {
            1u8
        };
        // STRLEN postprocess updates PtrInfo on the resolved receiver box.
        if let Some(arg0_box) = ctx.resolve_operand_operand_opt(&op.arg(0)) {
            ctx.make_nonnull_str(&arg0_box, mode);
        }
        if let Some(len) = self.get_known_length(&op.arg(0), ctx) {
            let _ = len;
        }
    }

    fn force_args_if_virtual(&mut self, op: &Op, ctx: &mut OptContext) {
        // earlyforce.py exempt set: SETFIELD_GC, SETARRAYITEM_GC, SAME_AS_*,
        // QUASIIMMUT_FIELD, raw_free do NOT force their args (the value of a
        // store can stay virtual). OptString is not RPython's forcing pass —
        // earlyforce is — so it must honor the same exemptions, else a virtual
        // stored into a non-virtual object (e.g. the exc published to the EC)
        // gets materialized here at pass 3 instead of routed to pendingfields.
        if !crate::optimizeopt::earlyforce::OptEarlyForce::should_force_args(op) {
            return;
        }
        // Collect operands first to avoid borrow issues. An unbound op-arg
        // position (no producer / inputarg yet) has no resolvable Operand, so
        // fall back to its canonical materialized stand-in rather than the
        // total resolver's position-only panic.
        let args: Vec<Operand> = op
            .getarglist()
            .iter()
            .map(|a| match ctx.resolve_operand_operand_opt(a) {
                Some(resolved) => resolved,
                None => ctx.materialize_operand_at(a.to_opref()),
            })
            .collect();
        for arg_op in args {
            if self.is_virtual(&arg_op, ctx) {
                self.force_box(&arg_op, ctx);
            }
        }
    }

    /// vstring.py:594-621 optimize_CALL_I — dispatch oopspec calls to
    /// specialized handlers. Str* variants get `mode_string`, Uni* variants
    /// get `mode_unicode`, paralleling the RPython table walk via
    /// `_OS_offset_uni`.
    fn optimize_oopspec_call(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ei: &EffectInfo,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        match ei.oopspecindex {
            OopSpecIndex::StrConcat => {
                self.opt_call_stroruni_str_concat(op, op_rc, mode_string, ctx)
            }
            OopSpecIndex::StrSlice => self.opt_call_stroruni_str_slice(op, op_rc, mode_string, ctx),
            OopSpecIndex::StrEqual => self.opt_call_stroruni_str_equal(op, mode_string, ctx),
            OopSpecIndex::StrCmp => self.opt_call_stroruni_str_cmp(op, op_rc, mode_string, ctx),
            OopSpecIndex::UniConcat => {
                self.opt_call_stroruni_str_concat(op, op_rc, mode_unicode, ctx)
            }
            OopSpecIndex::UniSlice => {
                self.opt_call_stroruni_str_slice(op, op_rc, mode_unicode, ctx)
            }
            OopSpecIndex::UniEqual => self.opt_call_stroruni_str_equal(op, mode_unicode, ctx),
            OopSpecIndex::UniCmp => self.opt_call_stroruni_str_cmp(op, op_rc, mode_unicode, ctx),
            OopSpecIndex::ShrinkArray => self.opt_call_shrink_array(op, op_rc, ctx),
            _ => {
                self.force_args_if_virtual(op, ctx);
                OptimizationResult::PassOn
            }
        }
    }

    /// vstring.py:653-661 opt_call_stroruni_STR_CONCAT
    fn opt_call_stroruni_str_concat(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() >= 3 {
            // vstring.py:654-655: make_nonnull_str on each concat operand,
            // unconditionally. The resolved string boxes are the PtrInfo hosts.
            let vleft_box = ctx.resolve_operand_operand(&op.arg(1));
            let vright_box = ctx.resolve_operand_operand(&op.arg(2));
            ctx.make_nonnull_str(&vleft_box, mode);
            ctx.make_nonnull_str(&vright_box, mode);
            let b = Operand::from_bound_op(op_rc);
            ctx.set_ptr_info(
                &b,
                PtrInfo::Str(StrPtrInfo {
                    lenbound: None,
                    lgtop: None,
                    mode,
                    length: -1,
                    variant: VStringVariant::Concat(VStringConcatInfo {
                        vleft: vleft_box.clone(),
                        vright: vright_box.clone(),
                        _is_virtual: true,
                    }),
                    last_guard_pos: -1,
                    avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                }),
            );
            return OptimizationResult::Remove;
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:662-690 opt_call_stroruni_STR_SLICE
    fn opt_call_stroruni_str_slice(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() >= 4 {
            // vstring.py:663: self.make_nonnull_str(op.getarg(1), mode)
            let mut s = ctx.resolve_operand_operand(&op.arg(1));
            ctx.make_nonnull_str(&s, mode);
            let mut start = ctx.resolve_operand_operand(&op.arg(2));
            let stop = ctx.resolve_operand_operand(&op.arg(3));
            let lgtop = self.int_sub(&stop, &start, ctx);
            // vstring.py:682-685: double slicing s[i:j][k:l]
            if let Some(info) = self.get_slice_info(&s, ctx) {
                let source = info.s;
                let source_start = info.start;
                s = source;
                start = _int_add(&source_start, &start, ctx);
            }
            // vstring.py:220-225: VStringSliceInfo.__init__ sets
            // self.lgtop = length on the inherited StrPtrInfo field.
            let b = Operand::from_bound_op(op_rc);
            ctx.set_ptr_info(
                &b,
                PtrInfo::Str(StrPtrInfo {
                    lenbound: None,
                    lgtop: Some(lgtop.clone()),
                    mode,
                    length: -1,
                    variant: VStringVariant::Slice(VStringSliceInfo {
                        s: s.clone(),
                        start,
                        lgtop,
                    }),
                    last_guard_pos: -1,
                    avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                }),
            );
            return OptimizationResult::Remove;
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:692-733 opt_call_stroruni_STR_EQUAL
    fn opt_call_stroruni_str_equal(
        &mut self,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() < 3 {
            self.force_args_if_virtual(op, ctx);
            return OptimizationResult::PassOn;
        }
        // vstring.py:693-696
        let arg1 = ctx.resolve_operand_operand(&op.arg(1));
        let arg2 = ctx.resolve_operand_operand(&op.arg(2));
        let i1 = ctx.getptrinfo(&arg1).is_some();
        let i2 = ctx.getptrinfo(&arg2).is_some();
        // vstring.py:698-705: l1box = i1.getstrlen(arg1, self, mode)
        let l1box = if i1 {
            Some(ctx.getstrlen_opref(arg1.to_opref(), mode))
        } else {
            None
        };
        let l2box = if i2 {
            Some(ctx.getstrlen_opref(arg2.to_opref(), mode))
        } else {
            None
        };
        // vstring.py:706-712: isinstance(ConstInt) + different values
        if let (Some(l1), Some(l2)) = (l1box, l2box) {
            let l1c = ctx.isinstance_const_int(l1);
            let l2c = ctx.isinstance_const_int(l2);
            if let (Some(v1), Some(v2)) = (l1c, l2c) {
                if v1 != v2 {
                    let b = ctx.materialize_operand_at(op.pos.get());
                    ctx.make_constant_box(&b, Value::Int(0));
                    return OptimizationResult::Remove;
                }
            }
        }
        // vstring.py:714-718: handle_str_equal_level1 both directions
        if let Some(result) = self.handle_str_equal_level1(&arg1, &arg2, op, mode, ctx) {
            return result;
        }
        if let Some(result) = self.handle_str_equal_level1(&arg2, &arg1, op, mode, ctx) {
            return result;
        }
        // vstring.py:720-724: handle_str_equal_level2 both directions, each
        // passing the strlen box of its second argument computed above.
        if let Some(result) = self.handle_str_equal_level2(&arg1, &arg2, l2box, op, mode, ctx) {
            return result;
        }
        if let Some(result) = self.handle_str_equal_level2(&arg2, &arg1, l1box, op, mode, ctx) {
            return result;
        }
        // vstring.py:727-732: nonnull fallback with same_box check
        let a_nonnull = i1 && self.is_known_nonnull(&arg1, ctx);
        let b_nonnull = i2 && self.is_known_nonnull(&arg2, ctx);
        if a_nonnull && b_nonnull {
            // vstring.py:728: l1box.same_box(l2box) routes through
            // `OptContext::same_box` (history.py:204-205) which combines
            // `get_box_replacement` + identity + Const value equality.
            let same_len = matches!((l1box, l2box), (Some(a), Some(b)) if ctx.same_box(a, b));
            let oopspec = if same_len {
                OopSpecIndex::StreqLengthok
            } else {
                OopSpecIndex::StreqNonnull
            };
            if let Some(result) =
                self.generate_modified_call(oopspec, &[arg1.to_opref(), arg2.to_opref()], op, ctx)
            {
                return result;
            }
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:735-787 handle_str_equal_level1
    fn handle_str_equal_level1(
        &self,
        arg1: &Operand,
        arg2: &Operand,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        // vstring.py:740-741: l2box = i2.getstrlen(arg2, self, mode)
        let i2 = ctx.getptrinfo(arg2).is_some();
        let l2box = if i2 {
            Some(ctx.getstrlen_opref(arg2.to_opref(), mode))
        } else {
            None
        };
        let l2_const = l2box.and_then(|r| ctx.isinstance_const_int(r));
        // vstring.py:742-756: isinstance(l2box, ConstInt) checks
        if let Some(l2val) = l2_const {
            if l2val == 0 {
                // vstring.py:744-755: len-0 check
                if self.is_known_nonnull(arg1, ctx) {
                    // vstring.py:745: self.make_nonnull_str(arg1, mode)
                    ctx.make_nonnull_str(arg1, mode);
                    // vstring.py:747: lengthbox = i1.getstrlen(arg1, self, mode)
                    let lengthbox = ctx.getstrlen_opref(arg1.to_opref(), mode);
                    let zero = ctx.emit_constant_int(0);
                    let arg_len = ctx.materialize_operand_at(lengthbox);
                    let arg_zero = ctx.materialize_operand_at(zero);
                    let mut eq_op = Op::new(OpCode::IntEq, &[arg_len.clone(), arg_zero.clone()]);
                    eq_op.pos.set(op.pos.get());
                    // vstring.py:751-754: replace_op_with(INT_EQ, [len, 0]) then
                    // seo(op) = send_extra_operation(op, opt=None) restarts from
                    // first_optimization. Restart, not a final Emit.
                    return Some(OptimizationResult::Restart(eq_op));
                }
            }
            if l2val == 1 {
                // vstring.py:758-759: l1box = i1.getstrlen(arg1, self, mode)
                let i1 = ctx.getptrinfo(arg1).is_some();
                let l1box = if i1 {
                    Some(ctx.getstrlen_opref(arg1.to_opref(), mode))
                } else {
                    None
                };
                let l1_const = l1box.and_then(|r| ctx.isinstance_const_int(r));
                if l1_const == Some(1) {
                    // vstring.py:761-768: both length 1 → compare chars. Each
                    // strgetitem either folds a virtual/const char or emits a
                    // STRGETITEM, so both operands are always available.
                    let c1 = self.strgetitem_emit(arg1, 0, mode, ctx);
                    let c2 = self.strgetitem_emit(arg2, 0, mode, ctx);
                    let arg_ch1 = ctx.materialize_operand_at(c1);
                    let arg_ch2 = ctx.materialize_operand_at(c2);
                    let mut eq_op = Op::new(OpCode::IntEq, &[arg_ch1.clone(), arg_ch2.clone()]);
                    eq_op.pos.set(op.pos.get());
                    // vstring.py:765-767: replace_op_with(INT_EQ, [c1, c2]) then
                    // seo(op) = send_extra_operation(op, opt=None) restarts from
                    // first_optimization. Restart, not a final Emit.
                    return Some(OptimizationResult::Restart(eq_op));
                }
                // vstring.py:769-774: arg1 is a virtual slice, arg2 is length 1
                if let Some(info) = self.get_slice_info(arg1, ctx) {
                    let source = info.s.to_opref();
                    let start = info.start.to_opref();
                    let length = info.lgtop.to_opref();
                    let vchar = self.strgetitem_emit(arg2, 0, mode, ctx);
                    return self.generate_modified_call(
                        OopSpecIndex::StreqSliceChar,
                        &[source, start, length, vchar],
                        op,
                        ctx,
                    );
                }
            }
        }
        // vstring.py:776-787: arg2 is null
        if self.is_known_null(arg2, ctx) {
            if self.is_known_nonnull(arg1, ctx) {
                let b = ctx.materialize_operand_at(op.pos.get());
                ctx.make_constant_box(&b, Value::Int(0));
                return Some(OptimizationResult::Remove);
            }
            if self.is_known_null(arg1, ctx) {
                let b = ctx.materialize_operand_at(op.pos.get());
                ctx.make_constant_box(&b, Value::Int(1));
                return Some(OptimizationResult::Remove);
            }
            // vstring.py:784: PTR_EQ against CONST_NULL (ref-null, not int-zero)
            let null_const = ctx.emit_constant_ref(majit_ir::GcRef::NULL);
            let arg_a = ctx.materialize_operand_at(arg1.to_opref());
            let arg_null = ctx.materialize_operand_at(null_const);
            let mut eq_op = Op::new(OpCode::PtrEq, &[arg_a.clone(), arg_null.clone()]);
            eq_op.pos.set(op.pos.get());
            // vstring.py:785-786: replace_op_with(PTR_EQ, ...) then self.emit(op)
            // (Optimization.emit) — the op flows on to the passes after OptString.
            // pyre's Replace continues current_op through the rest of the chain.
            return Some(OptimizationResult::Replace(eq_op));
        }
        None
    }

    /// vstring.py:789-811 handle_str_equal_level2 — `l2box` is the strlen
    /// box of `arg2`, computed once by the caller (vstring.py:700-704) and
    /// threaded through, never recomputed here.
    fn handle_str_equal_level2(
        &self,
        arg1: &Operand,
        arg2: &Operand,
        l2box: Option<OpRef>,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        // vstring.py:792-805: if l2box: l2info = self.getintbound(l2box)
        if let Some(l2ref) = l2box {
            let l2info = {
                let b = ctx.get_box_replacement_operand(l2ref);
                ctx.getintbound_handle(&b).borrow().clone()
            };
            if l2info.is_constant() && l2info.get_constant_int() == 1 {
                // vstring.py:799: vchar = self.strgetitem(None, arg2, CONST_0, mode)
                let vchar = self.strgetitem_emit(arg2, 0, mode, ctx);
                // vstring.py:800-804
                let oopspec = if self.is_known_nonnull(arg1, ctx) {
                    OopSpecIndex::StreqNonnullChar
                } else {
                    OopSpecIndex::StreqChecknullChar
                };
                return self.generate_modified_call(oopspec, &[arg1.to_opref(), vchar], op, ctx);
            }
        }
        // vstring.py:807-813: if arg1 is a virtual slice
        if let Some(info) = self.get_slice_info(arg1, ctx) {
            let source = info.s.to_opref();
            let start = info.start.to_opref();
            let length = info.lgtop.to_opref();
            let oopspec = if self.is_known_nonnull(arg2, ctx) {
                OopSpecIndex::StreqSliceNonnull
            } else {
                OopSpecIndex::StreqSliceChecknull
            };
            return self.generate_modified_call(
                oopspec,
                &[source, start, length, arg2.to_opref()],
                op,
                ctx,
            );
        }
        None
    }

    /// vstring.py:776 `i2 and i2.is_null()` — uses getptrinfo which
    /// synthesizes ConstPtrInfo for constant refs.
    fn is_known_null(&self, op: &Operand, ctx: &OptContext) -> bool {
        if let Some(info) = ctx.getptrinfo(op) {
            return info.is_null();
        }
        false
    }

    /// vstring.py:777,800,808 `i1 and i1.is_nonnull()` — uses getptrinfo
    /// which synthesizes ConstPtrInfo for constant refs.
    fn is_known_nonnull(&self, op: &Operand, ctx: &OptContext) -> bool {
        if let Some(info) = ctx.getptrinfo(op) {
            return info.is_nonnull() || info.is_virtual();
        }
        false
    }

    /// vstring.py:853-860 generate_modified_call
    ///
    /// Look up the calldescr and func_ptr for the given oopspec in the
    /// CallInfoCollection, and emit a CALL_I with those args.
    fn generate_modified_call(
        &self,
        oopspec: OopSpecIndex,
        args: &[OpRef],
        result_op: &Op,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        // Clone Arc to avoid borrow conflict with ctx
        let cic = ctx.callinfocollection.clone()?;
        // vstring.py:852: calldescr, func = cic.callinfo_for_oopspec(oopspecindex)
        // — a missing oopspec yields (None, 0). PyPy then builds CALL_I with
        // descr=calldescr (possibly None). Op.descr is `Option<DescrRef>`, so
        // encode the None-descr CALL directly instead of bailing.
        let (calldescr, func_addr) = cic.callinfo_for_oopspec(oopspec);
        let func_const = ctx.alloc_op_position_typed(majit_ir::Type::Int);
        let b = ctx.materialize_operand_at(func_const);
        ctx.make_constant_box(&b, Value::Int(func_addr as i64));
        let mut call_args = vec![func_const];
        call_args.extend_from_slice(args);
        let mut call_args_operand: Vec<Operand> = Vec::with_capacity(call_args.len());
        for a in &call_args {
            call_args_operand.push(ctx.materialize_operand_at(*a));
        }
        // vstring.py:854: replace_op_with(result, rop.CALL_I, [...], descr=calldescr)
        let mut call_op = match calldescr {
            Some(d) => Op::with_descr(OpCode::CallI, &call_args_operand, d.clone()),
            None => Op::new(OpCode::CallI, &call_args_operand),
        };
        call_op.pos.set(result_op.pos.get());
        // vstring.py:857: return self.emit(op) (Optimization.emit) — the CALL_I
        // flows on to the passes after OptString (OptPure/OptHeap), not a final
        // emit. pyre's Replace re-runs current_op through the rest of the chain.
        Some(OptimizationResult::Replace(call_op))
    }

    /// vstring.py:816-838 opt_call_stroruni_STR_CMP
    fn opt_call_stroruni_str_cmp(
        &mut self,
        op: &Op,
        _op_rc: &majit_ir::OpRc,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() < 3 {
            self.force_args_if_virtual(op, ctx);
            return OptimizationResult::PassOn;
        }
        // vstring.py:819-822: bail out if either info is missing
        let i1 = ctx
            .getptrinfo(&op.arg(1).get_box_replacement(false))
            .is_some();
        let i2 = ctx
            .getptrinfo(&op.arg(2).get_box_replacement(false))
            .is_some();
        if !i1 || !i2 {
            self.force_args_if_virtual(op, ctx);
            return OptimizationResult::PassOn;
        }
        // vstring.py:823-824: l1box = i1.getstrlen(arg1, self, mode)
        let l1box = ctx.getstrlen_opref(op.arg(1).to_opref(), mode);
        let l2box = ctx.getstrlen_opref(op.arg(2).to_opref(), mode);
        // vstring.py:825-828: isinstance(ConstInt) and both == 1
        let l1c = ctx.isinstance_const_int(l1box);
        let l2c = ctx.isinstance_const_int(l2box);
        if l1c == Some(1) && l2c == Some(1) {
            // vstring.py:830-836: comparing two single chars. `replace_op_with`
            // rewrites the original op into INT_SUB(char1, char2) preserving its
            // result box, then `seo = send_extra_operation; seo(op)` re-dispatches
            // it. `seo`'s default `opt=None` (optimizer.py:594) restarts from
            // first_optimization, so the INT_SUB runs the whole pass chain — not
            // a final Emit, which would skip every subsequent pass. Mirror that
            // with a new op whose pos is the original result position
            // (replace_op_with) and a Restart result (send_extra_operation).
            let char1 = self.strgetitem_emit(&op.arg(1), 0, mode, ctx);
            let char2 = self.strgetitem_emit(&op.arg(2), 0, mode, ctx);
            let arg_char1 = ctx.materialize_operand_at(char1);
            let arg_char2 = ctx.materialize_operand_at(char2);
            let mut sub_op = Op::new(OpCode::IntSub, &[arg_char1.clone(), arg_char2.clone()]);
            sub_op.pos.set(op.pos.get());
            return OptimizationResult::Restart(sub_op);
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:155-158 VStringPlainInfo.shrink
    ///
    /// ```text
    /// def shrink(self, length):
    ///     assert length >= 0
    ///     self.length = length
    ///     del self._chars[length:]
    /// ```
    ///
    fn vstring_plain_shrink(sinfo: &mut StrPtrInfo, length: usize) {
        sinfo.length = length as i32;
        if let VStringVariant::Plain(info) = &mut sinfo.variant {
            info._chars.truncate(length);
        }
    }

    /// vstring.py:839-851 opt_call_SHRINK_ARRAY
    ///
    /// ```text
    /// def opt_call_SHRINK_ARRAY(self, op):
    ///     i1 = getptrinfo(op.getarg(1))
    ///     i2 = self.getintbound(op.getarg(2))
    ///     # If the index is constant, if the argument is virtual (we only
    ///     # support VStringPlainValue for now) we can optimize away the call.
    ///     if (i2 and i2.is_constant() and i1 and i1.is_virtual() and
    ///         isinstance(i1, VStringPlainInfo)):
    ///         length = i2.get_constant_int()
    ///         i1.shrink(length)
    ///         self.last_emitted_operation = REMOVED
    ///         self.make_equal_to(op, op.getarg(1))
    ///         return True
    ///     return False
    /// ```
    fn opt_call_shrink_array(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() >= 3 {
            let arg1_box = ctx.resolve_operand_operand_opt(&op.arg(1));
            let length = ctx
                .resolve_operand_operand_opt(&op.arg(2))
                .and_then(|b| ctx.get_constant_int_box(&b));
            // vstring.py:844-845: i2.is_constant() && i1.is_virtual() &&
            // isinstance(i1, VStringPlainInfo)
            if let Some(length) = length {
                let did_shrink = arg1_box
                    .as_ref()
                    .and_then(|b| {
                        ctx.with_ptr_info_mut(b, |info| {
                            if let PtrInfo::Str(sinfo) = info {
                                if matches!(sinfo.variant, VStringVariant::Plain(_)) {
                                    // vstring.py:847: i1.shrink(length)
                                    Self::vstring_plain_shrink(sinfo, length as usize);
                                    return true;
                                }
                            }
                            false
                        })
                    })
                    .unwrap_or(false);
                if did_shrink {
                    // vstring.py:849: self.make_equal_to(op, op.getarg(1))
                    let b_old = Operand::from_bound_op(op_rc);
                    let b_arg1 = arg1_box.expect("body-namespace OpRef must have an operand slot");
                    ctx.make_equal_to(&b_old, &b_arg1);
                    return OptimizationResult::Remove;
                }
            }
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }
}

impl Default for OptString {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptString {
    fn propagate_forward(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        match op.opcode {
            // vstring.py:440-444 optimize_NEWSTR / optimize_NEWUNICODE:
            // both dispatch to _optimize_NEWSTR(op, mode).
            OpCode::Newstr => self._optimize_newstr(op, op_rc, mode_string, ctx),
            OpCode::Newunicode => self._optimize_newstr(op, op_rc, mode_unicode, ctx),
            OpCode::Strsetitem => self.optimize_strsetitem(op, ctx),
            OpCode::Strgetitem => self.optimize_strgetitem(op, op_rc, mode_string, ctx),
            OpCode::Strlen => self.optimize_strlen(op, op_rc, ctx),
            OpCode::Copystrcontent => self.optimize_copystrcontent(op, mode_string, ctx),

            // vstring.py: Unicode operations — same logic as string ops
            // but with unicode-specific opcodes.
            OpCode::Unicodesetitem => self.optimize_strsetitem(op, ctx),
            OpCode::Unicodegetitem => self.optimize_strgetitem(op, op_rc, mode_unicode, ctx),
            OpCode::Unicodelen => self.optimize_strlen(op, op_rc, ctx),
            OpCode::Copyunicodecontent => self.optimize_copystrcontent(op, mode_unicode, ctx),

            // vstring.py: STRHASH/UNICODEHASH — force virtual string and emit.
            OpCode::Strhash | OpCode::Unicodehash => {
                let src = ctx.resolve_operand_operand(&op.arg(0));
                self.force_if_virtual(&src, ctx);
                OptimizationResult::PassOn
            }

            // vstring.py: optimize_GUARD_NO_EXCEPTION — if the last
            // emitted operation was removed (e.g. a string oopspec call
            // was virtualized), skip the guard.
            OpCode::GuardNoException => {
                // Delegate to default — the pure.rs pass handles this
                // via last_emitted_was_removed tracking.
                OptimizationResult::PassOn
            }

            // vstring.py: oopspec call handlers for string operations.
            // STR_CONCAT, STR_SLICE, STR_EQUAL are dispatched by OopSpecIndex
            // on CALL_* ops. For now, check if the call is a string oopspec.
            // vstring.py:621-627: optimize_CALL_R/F/N + optimize_CALL_PURE_*
            // are all aliased to optimize_CALL_I.
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN => {
                let __descr_arc_descr = op.getdescr();
                if let Some(ref descr) = __descr_arc_descr.as_ref() {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.get_extra_info();
                        if ei.has_oopspec() {
                            return self.optimize_oopspec_call(op, op_rc, &ei, ctx);
                        }
                    }
                }
                self.force_args_if_virtual(op, ctx);
                OptimizationResult::PassOn
            }

            _ => {
                // For any other op, force virtual strings that appear as arguments.
                self.force_args_if_virtual(op, ctx);
                OptimizationResult::PassOn
            }
        }
    }

    fn name(&self) -> &'static str {
        "string"
    }
}

#[cfg(test)]
mod tests {
    //! Upstream parity anchor: `rpython/jit/metainterp/test/test_string.py`
    //! for string-builder and copy-content behavior, plus
    //! `rpython/jit/metainterp/optimizeopt/vstring.py`.
    //!
    //! Tests that focus on `IntBound`-only constants, `lgtop` caching identity,
    //! or partial-pass behavior are original Rust regressions for helper paths
    //! that upstream usually exercises only through larger optimizer tests.

    use super::*;
    use crate::history::test_support::rooted_resop_operand;
    use crate::optimizeopt::info::{
        PtrInfo, StrPtrInfo, VStringConcatInfo, VStringPlainInfo, VStringSliceInfo, VStringVariant,
    };
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::Type;

    /// Bound drop-in for `from_opref(OpRef::int_op(n))` at an op-argument site:
    /// a rooted synthetic Int ResOp producer (sheds to `Operand::Op`, not the
    /// position-only `Operand::Box`) whose `to_opref()` stays `int_op(n)`, so
    /// the constant-map / PtrInfo keys this trace seeds by absolute position
    /// still resolve. The OptString driver runs that single pass and resolves
    /// args by position, so the detached synthetic never diverges.
    fn iop(n: u32) -> Operand {
        rooted_resop_operand(Type::Int, n)
    }

    /// Bound drop-in for `from_opref(OpRef::ref_op(n))` at an op-argument site.
    fn rop(n: u32) -> Operand {
        rooted_resop_operand(Type::Ref, n)
    }

    /// Assign sequential positions to ops and pre-seed constants in OptContext.
    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos.set(OpRef::op_typed(i as u32, op.type_));
        }
    }

    /// Run the OptString pass on a list of ops, with given pre-seeded constants.
    fn run_with_constants(ops: &[Op], constants: &[(u32, i64)]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptString::new()));

        // Seed constants into the context. Since Optimizer::optimize
        // creates its own context, we use a custom approach: run the pass
        // manually. Seed reserve_pos above any trace op.pos so that
        // force_virtual's synthesized ops don't collide with the original
        // trace positions — matches the invariant
        // `optimize_with_constants_and_inputs` maintains
        // (start_next_pos = max(num_inputs, max_pos + 1)).
        let max_pos = ops
            .iter()
            .map(|op| op.pos.get())
            .filter(|op| !op.is_none() && !op.is_constant())
            .map(|op| op.raw())
            .max()
            .unwrap_or(0);
        let start_next_pos = (max_pos + 1).max(ops.len() as u32);
        let mut ctx = OptContext::with_num_inputs_and_start_pos(ops.len(), 0, 0, start_next_pos);
        for &(idx, val) in constants {
            let b = ctx.materialize_operand_at(OpRef::int_op(idx));
            ctx.make_constant_box(&b, Value::Int(val));
        }

        // Register every non-constant LEAF arg position as a bound synthetic
        // producer in the context (`resop_refs`). The trace's char / source
        // operands are leaf values with no producing op in this fixture slice;
        // without a registered producer, a later force/emit resolves such an
        // arg to a position-only `from_opref` box that mints `Operand::Box`.
        // `materialize_operand_at` binds a `SameAs*` synthetic at the same position
        // (oparser's leaf-var wiring), so resolution sheds to `Operand::Op`.
        // Positions produced by a trace op are skipped — materializing a
        // synthetic there would shadow the real producer and defeat
        // virtualization. Constant positions already carry a Const box.
        let produced: std::collections::HashSet<OpRef> =
            ops.iter().map(|op| op.pos.get()).collect();
        for op in ops {
            for i in 0..op.num_args() {
                let r = op.arg(i).to_opref();
                if !r.is_none() && !r.is_constant() && !produced.contains(&r) {
                    ctx.materialize_operand_at(r);
                }
            }
        }

        let mut pass = OptString::new();
        pass.setup();

        for op in ops {
            // Resolve forwarded arguments.
            let mut resolved_op = op.clone();
            // optimizer.py:651-652 setarg loop parity. Store the canonical
            // terminal box (carrying the live _forwarded chain) like
            // propagate_from_pass_range, so the pass reads PtrInfo/IntBound
            // directly off resolved_op.arg(i) instead of a fresh unbound box.
            for i in 0..resolved_op.num_args() {
                resolved_op.setarg(i, ctx.resolve_operand_operand(&resolved_op.arg(i)));
            }
            let resolved_rc = std::rc::Rc::new(resolved_op.clone());
            ctx.bind_input_resops(std::slice::from_ref(&resolved_rc));
            match pass.propagate_forward(&resolved_op, &resolved_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {
                    // Op removed, nothing emitted.
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        pass.flush(&mut ctx);
        // Drain extra_operations_after (from emit_extra during force_box)
        // into new_operations so the test can see all emitted ops.
        while let Some((_pass_idx, extra_op)) = ctx.extra_operations_after.pop_front() {
            ctx.new_operations.push(extra_op);
        }
        ctx.new_operations
            .into_iter()
            .map(|rc| (*rc).clone())
            .collect()
    }

    fn set_vstring_plain(ctx: &mut OptContext, opref: OpRef, chars: Vec<Option<OpRef>>) {
        let length = chars.len() as i32;
        let b = ctx.materialize_operand_at(opref);
        // Materialize each char position so it carries a bound synthetic
        // producer in `resop_refs`. A later force/emit then resolves the char
        // arg to that bound box (sheds to `Operand::Op`) instead of a
        // position-only `from_opref` box (which would mint `Operand::Box`).
        // `materialize_operand_at` keeps the box's `to_opref()` at the same
        // position, so the char-identity assertions still hold.
        let char_boxes: Vec<Option<Operand>> = chars
            .into_iter()
            .map(|o| o.map(|r| ctx.materialize_operand_at(r)))
            .collect();
        ctx.set_ptr_info(
            &b,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length,
                variant: VStringVariant::Plain(VStringPlainInfo { _chars: char_boxes }),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );
    }

    fn set_vstring_concat(ctx: &mut OptContext, opref: OpRef, vleft: OpRef, vright: OpRef) {
        let b = ctx.materialize_operand_at(opref);
        // Materialize the child refs so they carry a bound synthetic producer; a
        // residual emit then sheds them to `Operand::Op` instead of panicking on
        // a position-only `from_opref` box. `materialize_operand_at` keeps each
        // box's `to_opref()` at the same position, so identity assertions hold.
        let vleft_box = ctx.materialize_operand_at(vleft);
        let vright_box = ctx.materialize_operand_at(vright);
        ctx.set_ptr_info(
            &b,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: VStringVariant::Concat(VStringConcatInfo {
                    vleft: vleft_box,
                    vright: vright_box,
                    _is_virtual: true,
                }),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );
    }

    fn set_vstring_slice(ctx: &mut OptContext, opref: OpRef, s: OpRef, start: OpRef, lgtop: OpRef) {
        let b = ctx.materialize_operand_at(opref);
        // Materialize the source/start/length refs so they carry a bound
        // synthetic producer; a residual emit then sheds them to `Operand::Op`
        // instead of panicking on a position-only `from_opref` box.
        // `materialize_operand_at` keeps each box's `to_opref()` at the same
        // position, so identity assertions hold.
        let s_box = ctx.materialize_operand_at(s);
        let start_box = ctx.materialize_operand_at(start);
        let lgtop_box = ctx.materialize_operand_at(lgtop);
        ctx.set_ptr_info(
            &b,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: Some(lgtop_box.clone()), // vstring.py:223: self.lgtop = length
                mode: 0,
                length: -1,
                variant: VStringVariant::Slice(VStringSliceInfo {
                    s: s_box,
                    start: start_box,
                    lgtop: lgtop_box,
                }),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );
    }

    // ── Test 1: STRGETITEM on virtual string returns tracked character ──

    #[test]
    fn test_strgetitem_virtual_plain() {
        // Setup:
        //   i100 = const 3   (length)
        //   i101 = const 0   (index 0)
        //   i102 = const 1   (index 1)
        //   i200 = <some char value for index 0>
        //   i201 = <some char value for index 1>
        //
        // Trace:
        //   p0 = newstr(i100)         -> virtual, removed
        //   _  = strsetitem(p0, i101, i200)  -> stored in virtual, removed
        //   _  = strsetitem(p0, i102, i201)  -> stored in virtual, removed
        //   i3 = strgetitem(p0, i101) -> should resolve to i200, removed

        let mut ops = vec![
            Op::new(OpCode::Newstr, &[iop(100)]), // op 0: p0 = newstr(3)
            Op::new(OpCode::Strsetitem, &[rop(0), iop(101), iop(200)]), // op 1
            Op::new(OpCode::Strsetitem, &[rop(0), iop(102), iop(201)]), // op 2
            Op::new(OpCode::Strgetitem, &[rop(0), iop(101)]), // op 3: get char at 0
        ];
        assign_positions(&mut ops);

        let constants = vec![
            (100, 3), // length = 3
            (101, 0), // index 0
            (102, 1), // index 1
        ];

        let result = run_with_constants(&ops, &constants);

        // All ops should be removed (string is fully virtual).
        assert!(
            result.is_empty(),
            "Expected all ops removed, got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Test 2: STRLEN on virtual string returns constant ──

    #[test]
    fn test_strlen_virtual() {
        // p0 = newstr(5)
        // i1 = strlen(p0) -> should be constant 5
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[iop(100)]), // op 0
            Op::new(OpCode::Strlen, &[rop(0)]),   // op 1
        ];
        assign_positions(&mut ops);

        let constants = vec![(100, 5)];

        let result = run_with_constants(&ops, &constants);

        // Both ops removed: newstr virtualized, strlen resolved to constant.
        assert!(
            result.is_empty(),
            "Expected empty result, got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Test 3: Virtual string that escapes -> forced ──

    #[test]
    fn test_force_virtual_on_escape() {
        // p0 = newstr(2)
        // strsetitem(p0, 0, c0)
        // strsetitem(p0, 1, c1)
        // call_n(p0)     -> forces the string
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[iop(100)]), // op 0
            Op::new(OpCode::Strsetitem, &[rop(0), iop(101), iop(200)]), // op 1
            Op::new(OpCode::Strsetitem, &[rop(0), iop(102), iop(201)]), // op 2
            Op::new(OpCode::CallN, &[rop(0)]),    // op 3: forces
        ];
        assign_positions(&mut ops);

        let constants = vec![(100, 2), (101, 0), (102, 1)];

        let result = run_with_constants(&ops, &constants);

        // After forcing, we expect:
        // - SameAsI (constant 2 for length)
        // - Newstr
        // - SameAsI (constant 0), Strsetitem (char at 0)
        // - SameAsI (constant 1), Strsetitem (char at 1)
        // - CallN (with forwarded ref to the new Newstr)
        //
        // The exact count depends on how many constant-int SameAsI ops are emitted.
        // Key check: there should be a Newstr, Strsetitem ops, and the call.

        let newstr_count = result.iter().filter(|o| o.opcode == OpCode::Newstr).count();
        let setitem_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::Strsetitem)
            .count();
        let call_count = result.iter().filter(|o| o.opcode == OpCode::CallN).count();

        assert_eq!(newstr_count, 1, "Should have 1 Newstr after forcing");
        assert_eq!(setitem_count, 2, "Should have 2 Strsetitem after forcing");
        assert_eq!(call_count, 1, "Should have 1 CallN");
    }

    // ── Test 4: Concat virtual string length ──

    #[test]
    fn test_concat_length() {
        // Build two virtual strings, create a concat in PtrInfo::Str, then
        // query length.

        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        // Constant length refs
        let b = ctx.materialize_operand_at(OpRef::int_op(100));
        ctx.make_constant_box(&b, Value::Int(3));
        let b = ctx.materialize_operand_at(OpRef::int_op(101));
        ctx.make_constant_box(&b, Value::Int(4));

        // Virtual plain strings
        let left_ref = OpRef::ref_op(10);
        let right_ref = OpRef::ref_op(11);
        set_vstring_plain(&mut ctx, left_ref, vec![None; 3]);
        set_vstring_plain(&mut ctx, right_ref, vec![None; 4]);

        // Virtual concat
        let concat_ref = OpRef::ref_op(12);
        set_vstring_concat(&mut ctx, concat_ref, left_ref, right_ref);

        // Check total length = 3 + 4 = 7
        let concat_op = ctx.materialize_operand_at(concat_ref);
        let total_len = pass.get_known_length(&concat_op, &ctx);
        assert_eq!(total_len, Some(7));
    }

    // ── Test 5: Slice virtual string ──

    #[test]
    fn test_slice_get_char() {
        // Build a virtual plain string, create a slice, get a character.
        let pass = OptString::new();
        let mut ctx = OptContext::new(10);

        // source = "abc" (chars at indices 0, 1, 2)
        let src_ref = OpRef::ref_op(10);
        set_vstring_plain(
            &mut ctx,
            src_ref,
            vec![
                Some(OpRef::int_op(200)),
                Some(OpRef::int_op(201)),
                Some(OpRef::int_op(202)),
            ],
        );

        // slice = source[1:3] (start=1, length=2)
        let b = ctx.materialize_operand_at(OpRef::int_op(300));
        ctx.make_constant_box(&b, Value::Int(1)); // start
        let b = ctx.materialize_operand_at(OpRef::int_op(301));
        ctx.make_constant_box(&b, Value::Int(2)); // length
        let slice_ref = OpRef::ref_op(11);
        set_vstring_slice(
            &mut ctx,
            slice_ref,
            src_ref,
            OpRef::int_op(300),
            OpRef::int_op(301),
        );

        let slice_op = ctx.materialize_operand_at(slice_ref);
        // Get char at index 0 of the slice -> should be source[1] = int_op(201)
        let ch = pass.strgetitem(&slice_op, 0, mode_string, &mut ctx);
        assert_eq!(ch, Some(OpRef::int_op(201)));

        // Get char at index 1 of the slice -> should be source[2] = int_op(202)
        let ch = pass.strgetitem(&slice_op, 1, mode_string, &mut ctx);
        assert_eq!(ch, Some(OpRef::int_op(202)));
    }

    #[test]
    fn test_slice_get_char_with_intbound_constant_start() {
        use crate::optimizeopt::intutils::IntBound;

        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        let src_ref = OpRef::ref_op(10);
        set_vstring_plain(
            &mut ctx,
            src_ref,
            vec![
                Some(OpRef::int_op(200)),
                Some(OpRef::int_op(201)),
                Some(OpRef::int_op(202)),
            ],
        );

        // start is not a literal ConstInt box; it is only known via IntBound.
        let start_ref = OpRef::int_op(300);
        let start_box = ctx.materialize_operand_at(start_ref);
        ctx.with_intbound_mut(&start_box, |b| {
            *b = IntBound::from_constant(1);
        });
        let b = ctx.materialize_operand_at(OpRef::int_op(301));
        ctx.make_constant_box(&b, Value::Int(2)); // length

        let slice_ref = OpRef::ref_op(11);
        set_vstring_slice(&mut ctx, slice_ref, src_ref, start_ref, OpRef::int_op(301));

        let slice_op = ctx.materialize_operand_at(slice_ref);
        assert_eq!(
            pass.strgetitem(&slice_op, 0, mode_string, &mut ctx),
            Some(OpRef::int_op(201))
        );
        assert_eq!(
            pass.strgetitem(&slice_op, 1, mode_string, &mut ctx),
            Some(OpRef::int_op(202))
        );
    }

    #[test]
    fn test_strgetitem_emit_virtual_slice_targets_source() {
        // vstring.py:490-493: when a virtual slice's char cannot be folded
        // statically (here the slice start is non-constant), the residual
        // STRGETITEM must read the SOURCE string at `start + index`, not the
        // slice box — emitting STRGETITEM(slice, index) would force the slice.
        let pass = OptString::new();
        let mut ctx = OptContext::new(12);

        let src_ref = OpRef::ref_op(10);
        set_vstring_plain(&mut ctx, src_ref, vec![None; 3]);

        // start is a non-constant int box (no constant/bound installed), so the
        // slice can't resolve the char statically and falls to the residual.
        // Materialize so the residual re-emits start as a bound box.
        let start_ref = OpRef::int_op(300);
        ctx.materialize_operand_at(start_ref);
        let b = ctx.materialize_operand_at(OpRef::int_op(301));
        ctx.make_constant_box(&b, Value::Int(2)); // length

        let slice_ref = OpRef::ref_op(11);
        set_vstring_slice(&mut ctx, slice_ref, src_ref, start_ref, OpRef::int_op(301));

        let slice_op = ctx.materialize_operand_at(slice_ref);
        let res = pass.strgetitem_emit(&slice_op, 0, mode_string, &mut ctx);

        let (_pass_idx, op) = ctx
            .extra_operations_after
            .back()
            .expect("strgetitem_emit must emit a residual STRGETITEM");
        assert_eq!(res, op.pos.get());
        assert_eq!(op.opcode, OpCode::Strgetitem);
        // arg0 is the SOURCE (ref_op(10)), not the slice (ref_op(11)); arg1 is
        // `start + 0`, which collapses back to the start box (int_op(300)).
        assert_eq!(
            op.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![src_ref, start_ref]
        );
    }

    #[test]
    fn test_strgetitem_emit_virtual_concat_targets_child() {
        // vstring.py:505-512: when a virtual concat's char cannot be folded
        // statically (here the children's chars are unset), the residual
        // STRGETITEM must read the CHILD holding that index, not the concat
        // box — STRGETITEM(concat, index) would force the whole concat.
        let pass = OptString::new();
        let mut ctx = OptContext::new(14);

        // vleft: plain length 2, vright: plain length 3; chars unset.
        let vleft_ref = OpRef::ref_op(10);
        let vright_ref = OpRef::ref_op(11);
        set_vstring_plain(&mut ctx, vleft_ref, vec![None; 2]);
        set_vstring_plain(&mut ctx, vright_ref, vec![None; 3]);

        let concat_ref = OpRef::ref_op(12);
        set_vstring_concat(&mut ctx, concat_ref, vleft_ref, vright_ref);

        // index 3 lands in vright at offset 3 - len(vleft) = 1.
        let concat_op = ctx.materialize_operand_at(concat_ref);
        let res = pass.strgetitem_emit(&concat_op, 3, mode_string, &mut ctx);

        let (res_pos, opcode, arg0, arg1) = {
            let (_pass_idx, op) = ctx
                .extra_operations_after
                .back()
                .expect("strgetitem_emit must emit a residual STRGETITEM");
            (
                op.pos.get(),
                op.opcode,
                op.arg(0).to_opref(),
                op.arg(1).get_box_replacement(false),
            )
        };
        assert_eq!(res, res_pos);
        assert_eq!(opcode, OpCode::Strgetitem);
        // arg0 is the RIGHT child (ref_op(11)), not the concat (ref_op(12)).
        assert_eq!(arg0, vright_ref);
        // arg1 is the rebased index 3 - 2 = 1.
        assert_eq!(ctx.get_constant_int_box(&arg1), Some(1));
    }

    #[test]
    fn test_optimize_strgetitem_virtual_slice_rebases_to_source() {
        // vstring.py:490-493: STRGETITEM over a virtual slice with a
        // non-constant index rewrites to read the SOURCE at start+index
        // (replace_op_with) rather than forcing the slice and passing the op on.
        let mut pass = OptString::new();
        pass.setup();
        let mut ctx = OptContext::new(20);

        // Non-virtual source ref so force_if_virtual leaves it as the source box.
        // Materialize the source / start / index leaf positions so the residual
        // STRGETITEM (and its INT_ADD index) re-emit them as bound boxes
        // (`Operand::Op`) rather than position-only `from_opref` boxes.
        let src_ref = OpRef::ref_op(10);
        ctx.materialize_operand_at(src_ref);
        // Non-constant start so the static fold misses → residual path.
        let start_ref = OpRef::int_op(300);
        ctx.materialize_operand_at(start_ref);
        let b = ctx.materialize_operand_at(OpRef::int_op(301));
        ctx.make_constant_box(&b, Value::Int(3)); // slice length

        let slice_ref = OpRef::ref_op(11);
        set_vstring_slice(&mut ctx, slice_ref, src_ref, start_ref, OpRef::int_op(301));

        // STRGETITEM(slice, index) with a non-constant index.
        let index_ref = OpRef::int_op(302);
        ctx.materialize_operand_at(index_ref);
        let pos = ctx.alloc_op_position_typed(majit_ir::Type::Int);
        let mut getitem = Op::new(OpCode::Strgetitem, &[rop(11), iop(302)]);
        getitem.pos.set(pos);
        let op_rc = std::rc::Rc::new(getitem.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op_rc));

        let result = pass.optimize_strgetitem(&getitem, &op_rc, mode_string, &mut ctx);

        match result {
            OptimizationResult::Replace(emitted) => {
                assert_eq!(emitted.opcode, OpCode::Strgetitem);
                // arg0 is the SOURCE (ref_op(10)), not the slice (ref_op(11)).
                assert_eq!(emitted.arg(0).to_opref(), src_ref);
                // arg1 is INT_ADD(start, index), not the original index alone.
                assert_ne!(emitted.arg(1).to_opref(), index_ref);
                // The rewritten op keeps the original result position.
                assert_eq!(emitted.pos.get(), pos);
            }
            other => panic!("expected Replace(STRGETITEM on source), got {other:?}"),
        }
    }

    #[test]
    fn test_optimize_strgetitem_slice_of_concat_rebases_to_child() {
        // vstring.py:490-512: STRGETITEM over a virtual slice whose source is a
        // virtual concat continues the dispatch past the slice rebase into the
        // concat branch, so the residual reads the CHILD that holds the index,
        // not the whole concat (which the slice-only rebase would have forced).
        let mut pass = OptString::new();
        pass.setup();
        let mut ctx = OptContext::new(20);

        // concat = vleft(plain len 2) ++ vright(non-virtual source ref).
        // Materialize the non-virtual child so the residual STRGETITEM re-emits
        // it as a bound box rather than a position-only `from_opref` box.
        let vleft_ref = OpRef::ref_op(10);
        let vright_ref = OpRef::ref_op(11);
        ctx.materialize_operand_at(vright_ref);
        set_vstring_plain(&mut ctx, vleft_ref, vec![None; 2]);
        let concat_ref = OpRef::ref_op(12);
        set_vstring_concat(&mut ctx, concat_ref, vleft_ref, vright_ref);

        // slice of the concat: start = 1, length = 3.
        let b = ctx.materialize_operand_at(OpRef::int_op(300));
        ctx.make_constant_box(&b, Value::Int(1));
        let b = ctx.materialize_operand_at(OpRef::int_op(301));
        ctx.make_constant_box(&b, Value::Int(3));
        let slice_ref = OpRef::ref_op(13);
        set_vstring_slice(
            &mut ctx,
            slice_ref,
            concat_ref,
            OpRef::int_op(300),
            OpRef::int_op(301),
        );

        // STRGETITEM(slice, 1) → concat[start 1 + 1 = 2] → vright[2 - len 2 = 0].
        let b = ctx.materialize_operand_at(OpRef::int_op(302));
        ctx.make_constant_box(&b, Value::Int(1));
        let pos = ctx.alloc_op_position_typed(majit_ir::Type::Int);
        let mut getitem = Op::new(OpCode::Strgetitem, &[rop(13), iop(302)]);
        getitem.pos.set(pos);
        let op_rc = std::rc::Rc::new(getitem.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op_rc));

        let result = pass.optimize_strgetitem(&getitem, &op_rc, mode_string, &mut ctx);

        match result {
            OptimizationResult::Replace(emitted) => {
                assert_eq!(emitted.opcode, OpCode::Strgetitem);
                // arg0 is the RIGHT child (ref_op(11)), not the slice or concat.
                assert_eq!(emitted.arg(0).to_opref(), vright_ref);
                // arg1 is the fully rebased index (1 + 1) - 2 = 0.
                assert_eq!(
                    ctx.get_constant_int_box(&emitted.arg(1).get_box_replacement(false)),
                    Some(0)
                );
                assert_eq!(emitted.pos.get(), pos);
            }
            other => panic!("expected Replace(STRGETITEM on child), got {other:?}"),
        }
    }

    // ── Test 6: Slice length via STRLEN ──

    #[test]
    fn test_slice_strlen() {
        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        let src_ref = OpRef::ref_op(10);
        set_vstring_plain(&mut ctx, src_ref, vec![None; 5]);

        let b = ctx.materialize_operand_at(OpRef::int_op(300));
        ctx.make_constant_box(&b, Value::Int(1)); // start
        let b = ctx.materialize_operand_at(OpRef::int_op(301));
        ctx.make_constant_box(&b, Value::Int(3)); // length

        let slice_ref = OpRef::ref_op(11);
        set_vstring_slice(
            &mut ctx,
            slice_ref,
            src_ref,
            OpRef::int_op(300),
            OpRef::int_op(301),
        );

        let slice_op = ctx.materialize_operand_at(slice_ref);
        let len = pass.get_known_length(&slice_op, &ctx);
        assert_eq!(len, Some(3));
    }

    #[test]
    fn test_getstrlen_uses_unicodelen_for_unicode() {
        let pass = OptString::new();
        let mut ctx = OptContext::new(10);
        let unicode_ref = OpRef::ref_op(7);
        // vstring.py:452 make_nonnull_str(op, mode_unicode) installs a
        // non-virtual StrPtrInfo with `mode = 1` so that later getstrlen
        // selects UNICODELEN instead of STRLEN.
        // Synthetic-OpRef test fixture: lazy-allocate the operand for the unicode_ref slot.
        let unicode_box = ctx.materialize_operand_at(unicode_ref);
        ctx.make_nonnull_str(&unicode_box, 1);

        let unicode_op = ctx.materialize_operand_at(unicode_ref);
        let len_ref = pass.getstrlen(&unicode_op, &mut ctx);
        // getstrlen delegates to ctx.getstrlen_opref which emits via
        // emit_extra (downstream pipeline), so check extra_operations_after.
        let (_pass_idx, last_op) = ctx
            .extra_operations_after
            .back()
            .expect("getstrlen must emit a len op");

        assert_eq!(len_ref, last_op.pos.get());
        assert_eq!(last_op.opcode, OpCode::Unicodelen);
        assert_eq!(
            last_op
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![unicode_ref]
        );
    }

    // ── Test 7: Non-constant length NEWSTR passes through ──

    #[test]
    fn test_newstr_non_constant_passes_through() {
        // newstr(i0) where i0 is not a known constant -> should emit.
        let mut ops = vec![Op::new(OpCode::Newstr, &[iop(50)])];
        assign_positions(&mut ops);

        let result = run_with_constants(&ops, &[]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Newstr);
    }

    // ── Test 8: Too-large NEWSTR passes through ──

    #[test]
    fn test_newstr_too_large_passes_through() {
        let mut ops = vec![Op::new(OpCode::Newstr, &[iop(50)])];
        assign_positions(&mut ops);

        let constants = vec![(50, (MAX_CONST_LEN + 1) as i64)];
        let result = run_with_constants(&ops, &constants);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Newstr);
    }

    // ── Test 9: STRGETITEM on non-virtual string passes through ──

    #[test]
    fn test_strgetitem_non_virtual() {
        let mut ops = vec![Op::new(OpCode::Strgetitem, &[rop(50), iop(51)])];
        assign_positions(&mut ops);

        let constants = vec![(51, 0)];
        let result = run_with_constants(&ops, &constants);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Strgetitem);
    }

    // ── Test 10: Force empty virtual string ──

    #[test]
    fn test_force_empty_virtual() {
        // p0 = newstr(0) -> virtual (0 chars)
        // call_n(p0)      -> force: emits newstr(0) only, no strsetitem
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[iop(100)]),
            Op::new(OpCode::CallN, &[rop(0)]),
        ];
        assign_positions(&mut ops);

        let constants = vec![(100, 0)];
        let result = run_with_constants(&ops, &constants);

        let newstr_count = result.iter().filter(|o| o.opcode == OpCode::Newstr).count();
        let setitem_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::Strsetitem)
            .count();
        assert_eq!(newstr_count, 1);
        assert_eq!(setitem_count, 0);
    }

    // ── Test 11: COPYSTRCONTENT into virtual string ──

    #[test]
    fn test_copystrcontent_virtual_to_virtual() {
        // src = newstr(2), strsetitem(src, 0, c0), strsetitem(src, 1, c1)
        // dst = newstr(2)
        // copystrcontent(src, dst, 0, 0, 2)
        // strgetitem(dst, 0) -> c0
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[iop(100)]), // op 0: src
            Op::new(OpCode::Strsetitem, &[rop(0), iop(101), iop(200)]), // op 1
            Op::new(OpCode::Strsetitem, &[rop(0), iop(102), iop(201)]), // op 2
            Op::new(OpCode::Newstr, &[iop(100)]), // op 3: dst
            Op::new(
                OpCode::Copystrcontent,
                &[rop(0), rop(3), iop(101), iop(101), iop(100)],
            ), // op 4: copy src->dst
            Op::new(OpCode::Strgetitem, &[rop(3), iop(101)]), // op 5: get dst[0]
        ];
        assign_positions(&mut ops);

        let constants = vec![(100, 2), (101, 0), (102, 1)];

        let result = run_with_constants(&ops, &constants);

        // All ops should be removed since everything is virtual.
        assert!(
            result.is_empty(),
            "Expected all ops removed, got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_copyunicodecontent_inline_uses_unicodegetitem() {
        let mut ops = vec![
            Op::new(OpCode::Newunicode, &[iop(100)]),
            Op::new(OpCode::Newunicode, &[iop(100)]),
            Op::new(
                OpCode::Copyunicodecontent,
                &[rop(0), rop(1), iop(101), iop(101), iop(100)],
            ),
        ];
        assign_positions(&mut ops);

        let constants = vec![(100, 2), (101, 0)];

        let result = run_with_constants(&ops, &constants);

        let unicode_getitem_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::Unicodegetitem)
            .count();
        let str_getitem_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::Strgetitem)
            .count();

        assert_eq!(unicode_getitem_count, 2);
        assert_eq!(str_getitem_count, 0);
    }

    // ── Test 12: Multiple STRLEN calls on same virtual ──

    #[test]
    fn test_strlen_multiple_calls() {
        // p0 = newstr(3)
        // i1 = strlen(p0) -> const 3
        // i2 = strlen(p0) -> const 3
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[iop(100)]),
            Op::new(OpCode::Strlen, &[rop(0)]),
            Op::new(OpCode::Strlen, &[rop(0)]),
        ];
        assign_positions(&mut ops);

        let constants = vec![(100, 3)];
        let result = run_with_constants(&ops, &constants);

        // All should be removed.
        assert!(result.is_empty());
    }

    // ── Test 13: STRGETITEM with uninitialized char falls through ──

    #[test]
    fn test_strgetitem_uninitialized_char() {
        // p0 = newstr(3), no strsetitem for index 0
        // strgetitem(p0, 0) -> char not set, must force and emit
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[iop(100)]),
            Op::new(OpCode::Strgetitem, &[rop(0), iop(101)]),
        ];
        assign_positions(&mut ops);

        let constants = vec![(100, 3), (101, 0)];

        let result = run_with_constants(&ops, &constants);

        // The string is forced, so we should see at least a Newstr + Strgetitem.
        let has_newstr = result.iter().any(|o| o.opcode == OpCode::Newstr);
        let has_getitem = result.iter().any(|o| o.opcode == OpCode::Strgetitem);
        assert!(has_newstr, "Should have forced Newstr");
        assert!(has_getitem, "Should have Strgetitem in output");
    }

    // ── Test 14: Concat virtual get_known_length with nested concat ──

    #[test]
    fn test_nested_concat_length() {
        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        let b = ctx.materialize_operand_at(OpRef::int_op(100));
        ctx.make_constant_box(&b, Value::Int(2));
        let b = ctx.materialize_operand_at(OpRef::int_op(101));
        ctx.make_constant_box(&b, Value::Int(3));
        let b = ctx.materialize_operand_at(OpRef::int_op(102));
        ctx.make_constant_box(&b, Value::Int(4));

        let a = OpRef::ref_op(10);
        let b = OpRef::ref_op(11);
        let c = OpRef::ref_op(12);
        set_vstring_plain(&mut ctx, a, vec![None; 2]);
        set_vstring_plain(&mut ctx, b, vec![None; 3]);
        set_vstring_plain(&mut ctx, c, vec![None; 4]);

        // ab = concat(a, b)
        let ab = OpRef::ref_op(20);
        set_vstring_concat(&mut ctx, ab, a, b);

        // abc = concat(ab, c)
        let abc = OpRef::ref_op(21);
        set_vstring_concat(&mut ctx, abc, ab, c);

        let abc_op = ctx.materialize_operand_at(abc);
        assert_eq!(pass.get_known_length(&abc_op, &ctx), Some(9));
    }

    // ── Test 15: Concat get char across boundary ──

    #[test]
    fn test_concat_get_char() {
        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        let b = ctx.materialize_operand_at(OpRef::int_op(100));
        ctx.make_constant_box(&b, Value::Int(2));
        let b = ctx.materialize_operand_at(OpRef::int_op(101));
        ctx.make_constant_box(&b, Value::Int(2));

        let left = OpRef::ref_op(10);
        let right = OpRef::ref_op(11);
        set_vstring_plain(
            &mut ctx,
            left,
            vec![Some(OpRef::int_op(200)), Some(OpRef::int_op(201))],
        );
        set_vstring_plain(
            &mut ctx,
            right,
            vec![Some(OpRef::int_op(202)), Some(OpRef::int_op(203))],
        );

        let concat = OpRef::ref_op(12);
        set_vstring_concat(&mut ctx, concat, left, right);

        let concat_op = ctx.materialize_operand_at(concat);
        // Index 0 -> left[0] = 200
        assert_eq!(
            pass.strgetitem(&concat_op, 0, mode_string, &mut ctx),
            Some(OpRef::int_op(200))
        );
        // Index 1 -> left[1] = 201
        assert_eq!(
            pass.strgetitem(&concat_op, 1, mode_string, &mut ctx),
            Some(OpRef::int_op(201))
        );
        // Index 2 -> right[0] = 202
        assert_eq!(
            pass.strgetitem(&concat_op, 2, mode_string, &mut ctx),
            Some(OpRef::int_op(202))
        );
        // Index 3 -> right[1] = 203
        assert_eq!(
            pass.strgetitem(&concat_op, 3, mode_string, &mut ctx),
            Some(OpRef::int_op(203))
        );
    }

    #[test]
    fn test_strlen_caching_non_virtual() {
        // Original Rust smoke test: `OptString` alone does not eliminate the
        // second non-virtual `STRLEN`, but this still guards the local
        // `known_lengths` cache wiring from panicking or regressing.
        // STRLEN on a non-virtual string should be cached for the second call.
        let mut ops = vec![
            Op::new(OpCode::Strlen, &[rop(100)]),
            Op::new(OpCode::Strlen, &[rop(100)]),
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);
        let result = run_with_constants(&ops, &[]);
        // Second STRLEN should be eliminated by heap.rs STRLEN caching
        // (if running through full pipeline) or by vstring.rs known_lengths.
        // With just OptString pass, the first STRLEN passes through and
        // records in known_lengths, but the second one checks known_lengths
        // which maps ref_op(100) → int_op(0) (result of first STRLEN).
        // Since int_op(0) is not a constant, it won't be removed by OptString alone.
        // This test just verifies no crash occurs.
        assert!(result.len() >= 1);
    }

    #[test]
    fn test_concat_oopspec_creates_virtual() {
        // Verify that STR_CONCAT creates a virtual Concat.
        let mut pass = OptString::new();
        pass.setup();

        let left = OpRef::ref_op(100);

        // Simulate: NEWSTR(2) for left
        let mut left_op = Op::new(OpCode::Newstr, &[iop(200)]);
        left_op.pos.set(left);
        let mut ctx = OptContext::new(10);
        let b = ctx.materialize_operand_at(OpRef::int_op(200));
        ctx.make_constant_box(&b, Value::Int(2));

        // Process NEWSTR → creates virtual Plain
        let left_op_rc = std::rc::Rc::new(left_op.clone());
        ctx.bind_input_resops(std::slice::from_ref(&left_op_rc));
        let _ = pass.propagate_forward(&left_op, &left_op_rc, &mut ctx);
        assert!(pass.is_virtual(&ctx.get_box_replacement_operand(left), &ctx));
    }

    // ── Box/state parity tests ──

    /// vstring.py:174: VStringPlainInfo.getstrlen caches lgtop.
    /// Second call must return the SAME OpRef (identity reuse).
    #[test]
    fn test_lgtop_reuse_plain() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(4, 0, 0, 50);
        let p0 = OpRef::ref_op(0);
        set_vstring_plain(
            &mut ctx,
            p0,
            vec![
                Some(OpRef::int_op(10)),
                Some(OpRef::int_op(11)),
                Some(OpRef::int_op(12)),
            ],
        );

        let first = ctx.getstrlen_opref(p0, 0);
        let second = ctx.getstrlen_opref(p0, 0);
        // vstring.py:174: self.lgtop = ConstInt(len(self._chars))
        // Both calls must return the SAME cached OpRef.
        assert_eq!(
            first, second,
            "lgtop must be reused: first={:?}, second={:?}",
            first, second
        );
        // The cached value must equal the Plain length (3).
        assert_eq!(
            ctx.get_box_replacement_operand_opt(first)
                .and_then(|cb| cb.const_int()),
            Some(3)
        );
    }

    /// vstring.py:117: StrPtrInfo.getstrlen caches STRLEN result in lgtop.
    /// After emitting STRLEN, the second call must return the cached OpRef.
    #[test]
    fn test_lgtop_reuse_nonvirtual() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(4, 0, 0, 50);
        let p0 = OpRef::ref_op(0);
        // Non-virtual Str with unknown length
        let p0_box = ctx.materialize_operand_at(p0);
        ctx.set_ptr_info(
            &p0_box,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let first = ctx.getstrlen_opref(p0, 0);
        let second = ctx.getstrlen_opref(p0, 0);
        // vstring.py:117: self.lgtop = lengthop — cached STRLEN result
        assert_eq!(
            first, second,
            "STRLEN result must be cached in lgtop: first={:?}, second={:?}",
            first, second
        );
    }

    /// vstring.py:728: l1box.same_box(l2box) succeeds when both strings
    /// have the same cached lgtop. getstrlen_if_known must return the
    /// cached OpRef, not a freshly-created constant.
    #[test]
    fn test_same_box_identity() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(4, 0, 0, 50);
        let pass = OptString::new();

        let p0 = OpRef::ref_op(0);
        let p1 = OpRef::ref_op(1);
        // Two virtual strings of the same length (3 chars).
        set_vstring_plain(
            &mut ctx,
            p0,
            vec![
                Some(OpRef::int_op(10)),
                Some(OpRef::int_op(11)),
                Some(OpRef::int_op(12)),
            ],
        );
        set_vstring_plain(
            &mut ctx,
            p1,
            vec![
                Some(OpRef::int_op(20)),
                Some(OpRef::int_op(21)),
                Some(OpRef::int_op(22)),
            ],
        );

        let p0_op = ctx.materialize_operand_at(p0);
        let p1_op = ctx.materialize_operand_at(p1);
        // First call caches lgtop on each string.
        let l1 = pass.getstrlen_if_known(&p0_op, &mut ctx);
        let l2 = pass.getstrlen_if_known(&p1_op, &mut ctx);
        assert!(l1.is_some() && l2.is_some());

        // Second call must return the same cached OpRef.
        let l1_again = pass.getstrlen_if_known(&p0_op, &mut ctx);
        let l2_again = pass.getstrlen_if_known(&p1_op, &mut ctx);
        assert_eq!(l1, l1_again, "lgtop identity: p0 must return same OpRef");
        assert_eq!(l2, l2_again, "lgtop identity: p1 must return same OpRef");

        // Both have value 3, and RPython's same_box checks constant equality.
        assert_eq!(
            ctx.get_box_replacement_operand_opt(l1.unwrap())
                .and_then(|cb| cb.const_int()),
            Some(3)
        );
        assert_eq!(
            ctx.get_box_replacement_operand_opt(l2.unwrap())
                .and_then(|cb| cb.const_int()),
            Some(3)
        );
    }

    /// vstring.py:341-347: copy_str_content uses getintbound().is_constant()
    /// for the inline threshold check. Verify intbound-based constant
    /// detection enables the same inlining as literal constant detection.
    #[test]
    fn test_copy_str_content_intbound_inline() {
        use crate::optimizeopt::intutils::IntBound;

        let mut ctx = OptContext::with_num_inputs_and_start_pos(10, 0, 0, 50);

        // srcbox (p0): non-null string, not virtual
        let p0 = OpRef::ref_op(0);
        let p0_box = ctx.materialize_operand_at(p0);
        ctx.set_ptr_info(
            &p0_box,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );
        // targetbox: non-virtual. Materialize so the inline force re-emits the
        // STRSETITEM target arg as a bound box (`Operand::Op`), not a
        // position-only `from_opref` box.
        let p1 = OpRef::ref_op(1);
        let p1_box = ctx.materialize_operand_at(p1);

        // lengthbox (i2): int with constant intbound = 2
        // Use an OpRef with IntBound set (not a literal constant)
        let i2 = OpRef::int_op(2);
        let i2_box = ctx.materialize_operand_at(i2);
        ctx.with_intbound_mut(&i2_box, |b| {
            *b = IntBound::from_constant(2);
        });

        // offsetbox and srcoffsetbox: constant 0
        let off = ctx.emit_constant_int(0);
        let off_op = ctx.materialize_operand_at(off);

        // Call copy_str_content. With intbound-constant length = 2 <= M=2,
        // it should inline to STRGETITEM+STRSETITEM instead of COPYSTRCONTENT.
        let _result = copy_str_content(
            &mut ctx, &p0_box, &p1_box, &off_op, &off_op, &i2_box, 0, true,
        );

        // emit_for_force routes to extra_operations_after; drain it.
        while let Some((_pass_idx, extra_op)) = ctx.extra_operations_after.pop_front() {
            ctx.new_operations.push(extra_op);
        }

        // Check that STRGETITEM ops were emitted (inline path) instead of
        // a single COPYSTRCONTENT (bulk path).
        let getitem_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::Strgetitem)
            .count();
        let copy_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::Copystrcontent)
            .count();
        assert!(
            getitem_count > 0 && copy_count == 0,
            "intbound-constant length should trigger inline path: \
             getitem={}, copy={}",
            getitem_count,
            copy_count,
        );
    }

    /// vstring.py:110-119 getstrlen_opref parity:
    /// getstrlen_opref(opref, mode) looks up info from opref and emits
    /// STRLEN(opref) on cache miss. Cached lgtop is returned on second call.
    #[test]
    fn test_getstrlen_opref_on_nonvirtual() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(10, 0, 0, 50);
        let arg2 = OpRef::ref_op(1);
        let arg2_box = ctx.materialize_operand_at(arg2);

        ctx.set_ptr_info(
            &arg2_box,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let strlen_ref = ctx.getstrlen_opref(arg2, 0);

        let (_pass_idx, strlen_op) = ctx
            .extra_operations_after
            .back()
            .expect("should have emitted STRLEN");
        assert_eq!(strlen_op.opcode, OpCode::Strlen);
        assert_eq!(
            strlen_op
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![arg2]
        );
        assert_eq!(strlen_ref, strlen_op.pos.get());

        // Subsequent call must return the cached lgtop.
        let strlen_ref2 = ctx.getstrlen_opref(arg2, 0);
        assert_eq!(strlen_ref, strlen_ref2);
    }

    #[test]
    fn test_force_then_strlen_reuse() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(10, 0, 0, 50);

        let p0 = OpRef::ref_op(0);
        // Virtual Plain string with 3 chars.
        set_vstring_plain(
            &mut ctx,
            p0,
            vec![
                Some(OpRef::int_op(10)),
                Some(OpRef::int_op(11)),
                Some(OpRef::int_op(12)),
            ],
        );

        // getstrlen_opref should cache lgtop = ConstInt(3).
        let len1 = ctx.getstrlen_opref(p0, 0);
        assert_eq!(
            ctx.get_box_replacement_operand_opt(len1)
                .and_then(|cb| cb.const_int()),
            Some(3)
        );

        // Query again — must return the same cached OpRef.
        let len2 = ctx.getstrlen_opref(p0, 0);
        assert_eq!(len1, len2, "force-then-strlen: lgtop must be reused");
    }
}
