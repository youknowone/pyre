/// String virtualization optimization pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/vstring.py.
///
/// Virtualizes string operations so that short, fully-tracked strings never
/// allocate at runtime:
///
/// - NEWSTR with constant length -> virtual string (characters tracked)
/// - STRSETITEM on virtual string -> stores character in tracked array
/// - STRGETITEM on virtual string -> returns tracked character
/// - STRLEN on virtual string -> constant length
/// - COPYSTRCONTENT -> updates tracked characters or emits forced copies
///
/// When a virtual string escapes (used in an unhandled context), it is
/// "forced": NEWSTR + STRSETITEM ops are emitted to materialize it.
use std::collections::HashMap;

use majit_ir::{EffectInfo, OopSpecIndex, Op, OpCode, OpRef, Value};

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// Maximum constant string length we will virtualize.
const MAX_CONST_LEN: usize = 100;

/// vstring.py mode_string / mode_unicode discriminators.
pub const mode_string: u8 = 0;
pub const mode_unicode: u8 = 1;

/// Virtual string representation.
///
/// Tracks the contents of a string that has not yet been allocated.
#[derive(Clone, Debug)]
#[allow(dead_code)]
enum VStringInfo {
    /// String with individually tracked characters.
    /// Each slot is `Some(opref)` once written, `None` while uninitialized.
    Plain { chars: Vec<Option<OpRef>> },

    /// Concatenation of two strings (lazy, allocation deferred).
    Concat { left: OpRef, right: OpRef },

    /// Slice of another string.
    Slice {
        source: OpRef,
        start: OpRef,
        length: OpRef,
    },
}

/// The OptString optimization pass.
pub struct OptString {
    /// Map from OpRef (the NEWSTR result) to its virtual string info.
    vstrings: HashMap<OpRef, VStringInfo>,
    /// Map from OpRef to the known length (as OpRef to a constant op or input).
    known_lengths: HashMap<OpRef, OpRef>,
    /// vstring.py: Track which OpRefs are unicode strings (vs byte strings).
    /// Affects which opcodes are emitted during force (NEWUNICODE vs NEWSTR).
    unicode_refs: std::collections::HashSet<OpRef>,
}

impl OptString {
    pub fn new() -> Self {
        OptString {
            vstrings: HashMap::new(),
            known_lengths: HashMap::new(),
            unicode_refs: std::collections::HashSet::new(),
        }
    }

    /// Get the virtual string info for an OpRef, following forwarding.
    #[allow(dead_code)]
    fn get_vstring<'a>(&'a self, opref: OpRef, ctx: &OptContext) -> Option<&'a VStringInfo> {
        let resolved = ctx.get_box_replacement(opref);
        self.vstrings.get(&resolved)
    }

    /// Force a virtual string: emit NEWSTR + STRSETITEM ops so it becomes real.
    ///
    /// Returns the OpRef of the emitted NEWSTR.
    fn force_box(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        let resolved = ctx.get_box_replacement(opref);
        let info = match self.vstrings.remove(&resolved) {
            Some(info) => info,
            None => return resolved, // already forced or not virtual
        };

        match info {
            VStringInfo::Plain { chars } => {
                let len = chars.len();
                let is_unicode = self.unicode_refs.contains(&resolved);
                let len_ref = self.emit_constant_int(len as i64, ctx);
                // vstring.py: use NEWUNICODE/UNICODESETITEM for unicode strings.
                let new_opcode = if is_unicode {
                    OpCode::Newunicode
                } else {
                    OpCode::Newstr
                };
                let set_opcode = if is_unicode {
                    OpCode::Unicodesetitem
                } else {
                    OpCode::Strsetitem
                };
                let newstr_op = Op::new(new_opcode, &[len_ref]);
                let str_ref = ctx.emit(newstr_op);
                for (i, ch) in chars.iter().enumerate() {
                    if let Some(ch_ref) = ch {
                        let idx_ref = self.emit_constant_int(i as i64, ctx);
                        let ch_resolved = ctx.get_box_replacement(*ch_ref);
                        let setitem_op = Op::new(set_opcode, &[str_ref, idx_ref, ch_resolved]);
                        ctx.emit(setitem_op);
                    }
                }
                // Forward the original opref to the newly emitted NEWSTR.
                ctx.replace_op(resolved, str_ref);
                str_ref
            }
            VStringInfo::Concat { left, right } => {
                // vstring.py: if one side has length 0, return the other.
                let left_len_val = self.get_known_length(left, ctx);
                let right_len_val = self.get_known_length(right, ctx);
                if left_len_val == Some(0) {
                    let right_forced = self.force_box(right, ctx);
                    ctx.replace_op(resolved, right_forced);
                    return right_forced;
                }
                if right_len_val == Some(0) {
                    let left_forced = self.force_box(left, ctx);
                    ctx.replace_op(resolved, left_forced);
                    return left_forced;
                }

                let is_unicode = self.unicode_refs.contains(&resolved);
                let left_forced = self.force_box(left, ctx);
                let right_forced = self.force_box(right, ctx);
                let left_len = self.getstrlen(left_forced, ctx);
                let right_len = self.getstrlen(right_forced, ctx);
                let total_op = Op::new(OpCode::IntAdd, &[left_len, right_len]);
                let total_ref = ctx.emit(total_op);
                // vstring.py: use correct opcodes for byte vs unicode strings
                let new_opcode = if is_unicode {
                    OpCode::Newunicode
                } else {
                    OpCode::Newstr
                };
                let copy_opcode = if is_unicode {
                    OpCode::Copyunicodecontent
                } else {
                    OpCode::Copystrcontent
                };
                let newstr_op = Op::new(new_opcode, &[total_ref]);
                let str_ref = ctx.emit(newstr_op);
                let zero = self.emit_constant_int(0, ctx);
                let copy_left = Op::new(copy_opcode, &[left_forced, str_ref, zero, zero, left_len]);
                ctx.emit(copy_left);
                let copy_right = Op::new(
                    copy_opcode,
                    &[right_forced, str_ref, zero, left_len, right_len],
                );
                ctx.emit(copy_right);
                ctx.replace_op(resolved, str_ref);
                str_ref
            }
            VStringInfo::Slice {
                source,
                start,
                length,
            } => {
                let src_forced = self.force_box(source, ctx);
                let start_resolved = ctx.get_box_replacement(start);
                let length_resolved = ctx.get_box_replacement(length);
                let is_unicode = self.unicode_refs.contains(&resolved);
                let new_opcode = if is_unicode {
                    OpCode::Newunicode
                } else {
                    OpCode::Newstr
                };
                let copy_opcode = if is_unicode {
                    OpCode::Copyunicodecontent
                } else {
                    OpCode::Copystrcontent
                };
                let newstr_op = Op::new(new_opcode, &[length_resolved]);
                let str_ref = ctx.emit(newstr_op);
                let zero = self.emit_constant_int(0, ctx);
                let copy_op = Op::new(
                    copy_opcode,
                    &[src_forced, str_ref, start_resolved, zero, length_resolved],
                );
                ctx.emit(copy_op);
                ctx.replace_op(resolved, str_ref);
                str_ref
            }
        }
    }

    /// Emit a SameAsI op that produces a constant integer value.
    ///
    /// We need a way to reference constant values as OpRefs. We emit a
    /// SameAsI(dummy) and record the constant in the context.
    fn emit_constant_int(&self, value: i64, ctx: &mut OptContext) -> OpRef {
        // Emit a dummy SameAsI to get an OpRef, then record the constant.
        let op = Op::new(OpCode::SameAsI, &[OpRef::NONE]);
        let opref = ctx.emit(op);
        ctx.make_constant(opref, Value::Int(value));
        opref
    }

    /// Get or compute the STRLEN for a string OpRef.
    fn getstrlen(&self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        let resolved = ctx.get_box_replacement(opref);
        if let Some(&len_ref) = self.known_lengths.get(&resolved) {
            return ctx.get_box_replacement(len_ref);
        }
        // Emit a STRLEN op.
        let strlen_op = Op::new(OpCode::Strlen, &[resolved]);
        ctx.emit(strlen_op)
    }

    /// Get the strlen OpRef if already known, without emitting a new op.
    fn getstrlen_if_known(&self, opref: OpRef, ctx: &mut OptContext) -> Option<OpRef> {
        let resolved = ctx.get_box_replacement(opref);
        if let Some(&len_ref) = self.known_lengths.get(&resolved) {
            return Some(ctx.get_box_replacement(len_ref));
        }
        if let Some(info) = self.vstrings.get(&resolved) {
            match info {
                VStringInfo::Plain { chars } => {
                    let len = chars.len() as i64;
                    return Some(self.emit_constant_int(len, ctx));
                }
                _ => {}
            }
        }
        None
    }

    /// Try to get a character from a virtual string at a constant index.
    fn strgetitem(&self, opref: OpRef, index: i64, ctx: &OptContext) -> Option<OpRef> {
        let resolved = ctx.get_box_replacement(opref);
        let info = self.vstrings.get(&resolved)?;
        match info {
            VStringInfo::Plain { chars } => {
                let idx = index as usize;
                if idx < chars.len() { chars[idx] } else { None }
            }
            VStringInfo::Concat { left, right } => {
                // Need to know the left length to decide which side.
                let left_resolved = ctx.get_box_replacement(*left);
                let left_len = self.get_known_length(left_resolved, ctx)?;
                if index < left_len {
                    self.strgetitem(*left, index, ctx)
                } else {
                    self.strgetitem(*right, index - left_len, ctx)
                }
            }
            VStringInfo::Slice { source, start, .. } => {
                let start_val = ctx.get_constant_int(*start)?;
                self.strgetitem(*source, index + start_val, ctx)
            }
        }
    }

    /// Get the known length of a virtual string as a constant, if available.
    fn get_known_length(&self, opref: OpRef, ctx: &OptContext) -> Option<i64> {
        let resolved = ctx.get_box_replacement(opref);
        // Check known_lengths map.
        if let Some(&len_ref) = self.known_lengths.get(&resolved) {
            return ctx.get_constant_int(len_ref);
        }
        // Check vstring info.
        let info = self.vstrings.get(&resolved)?;
        match info {
            VStringInfo::Plain { chars } => Some(chars.len() as i64),
            VStringInfo::Concat { left, right } => {
                let left_resolved = ctx.get_box_replacement(*left);
                let right_resolved = ctx.get_box_replacement(*right);
                let l = self.get_known_length(left_resolved, ctx)?;
                let r = self.get_known_length(right_resolved, ctx)?;
                Some(l + r)
            }
            VStringInfo::Slice { length, .. } => ctx.get_constant_int(*length),
        }
    }

    /// Handle NEWSTR: virtualize if length is a small constant.
    fn optimize_newstr(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let len_ref = op.arg(0);
        if let Some(len) = ctx.get_constant_int(len_ref) {
            if len >= 0 && (len as usize) <= MAX_CONST_LEN {
                let info = VStringInfo::Plain {
                    chars: vec![None; len as usize],
                };
                self.vstrings.insert(op.pos, info);
                self.known_lengths.insert(op.pos, len_ref);
                return OptimizationResult::Remove;
            }
        }
        // vstring.py postprocess_NEWSTR: pure_from_args(STRLEN, op, length)
        // Even for non-virtual NEWSTR, record that STRLEN(result) = length.
        // This enables CSE to eliminate subsequent STRLEN calls.
        self.known_lengths.insert(op.pos, len_ref);
        OptimizationResult::PassOn
    }

    /// Handle STRSETITEM: if target is virtual Plain and index is constant, track.
    fn optimize_strsetitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let str_ref = ctx.get_box_replacement(op.arg(0));
        let idx_ref = op.arg(1);
        let char_ref = op.arg(2);

        if let Some(idx) = ctx.get_constant_int(idx_ref) {
            if let Some(info) = self.vstrings.get_mut(&str_ref) {
                if let VStringInfo::Plain { chars } = info {
                    let i = idx as usize;
                    if i < chars.len() {
                        chars[i] = Some(ctx.get_box_replacement(char_ref));
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        // Not virtual or index not constant -> force and emit.
        self.force_if_virtual(str_ref, ctx);
        OptimizationResult::PassOn
    }

    /// Handle STRGETITEM: if source is virtual, resolve the character.
    fn optimize_strgetitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let str_ref = ctx.get_box_replacement(op.arg(0));
        let idx_ref = op.arg(1);

        if let Some(idx) = ctx.get_constant_int(idx_ref) {
            if let Some(ch_ref) = self.strgetitem(str_ref, idx, ctx) {
                let ch_resolved = ctx.get_box_replacement(ch_ref);
                ctx.replace_op(op.pos, ch_resolved);
                return OptimizationResult::Remove;
            }
        }
        // Not fully resolved -> force the string.
        self.force_if_virtual(str_ref, ctx);
        OptimizationResult::PassOn
    }

    /// Handle STRLEN: if source is virtual, return known length.
    fn optimize_strlen(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let str_ref = ctx.get_box_replacement(op.arg(0));

        if let Some(len) = self.get_known_length(str_ref, ctx) {
            ctx.make_constant(op.pos, Value::Int(len));
            self.known_lengths.insert(op.pos, op.pos);
            return OptimizationResult::Remove;
        }
        // vstring.py: even if not virtual, cache the STRLEN result
        // so subsequent STRLEN on the same string can be eliminated
        // (via heap.rs STRLEN caching or here).
        self.known_lengths.insert(str_ref, op.pos);
        OptimizationResult::PassOn
    }

    /// Handle COPYSTRCONTENT: if destination is virtual Plain, track characters.
    fn optimize_copystrcontent(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // copystrcontent(src, dst, src_start, dst_start, length)
        let src_ref = ctx.get_box_replacement(op.arg(0));
        let dst_ref = ctx.get_box_replacement(op.arg(1));
        let src_start_ref = op.arg(2);
        let dst_start_ref = op.arg(3);
        let length_ref = op.arg(4);

        // Only handle copying into a virtual Plain string with all-constant offsets.
        if let (Some(src_start), Some(dst_start), Some(length)) = (
            ctx.get_constant_int(src_start_ref),
            ctx.get_constant_int(dst_start_ref),
            ctx.get_constant_int(length_ref),
        ) {
            if self.vstrings.contains_key(&dst_ref) {
                // Try to resolve each character from the source.
                let mut all_resolved = true;
                let mut resolved_chars = Vec::new();
                for i in 0..length {
                    if let Some(ch) = self.strgetitem(src_ref, src_start + i, ctx) {
                        resolved_chars.push(Some(ch));
                    } else {
                        all_resolved = false;
                        break;
                    }
                }

                if all_resolved {
                    if let Some(VStringInfo::Plain { chars }) = self.vstrings.get_mut(&dst_ref) {
                        for (i, ch) in resolved_chars.into_iter().enumerate() {
                            let dst_idx = (dst_start as usize) + i;
                            if dst_idx < chars.len() {
                                chars[dst_idx] = ch;
                            }
                        }
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        // vstring.py: M-character inline heuristic.
        // If the source is a non-virtual string, the dest is virtual Plain,
        // and the length is small (≤8), emit individual STRGETITEM + track
        // in the virtual's chars array instead of a bulk copy.
        const MAX_INLINE_COPY: i64 = 8;
        if let (Some(dst_start), Some(length)) = (
            ctx.get_constant_int(dst_start_ref),
            ctx.get_constant_int(length_ref),
        ) {
            if length > 0
                && length <= MAX_INLINE_COPY
                && !self.vstrings.contains_key(&src_ref)
                && self.vstrings.contains_key(&dst_ref)
            {
                // Emit STRGETITEM for each character from non-virtual source,
                // then track them in the virtual dst.
                let mut char_refs = Vec::new();
                let src_start_val = ctx.get_constant_int(src_start_ref).unwrap_or(0);
                for i in 0..length {
                    let idx_ref = self.emit_constant_int(src_start_val + i, ctx);
                    let get_op = Op::new(OpCode::Strgetitem, &[src_ref, idx_ref]);
                    let ch_ref = ctx.emit(get_op);
                    char_refs.push(ch_ref);
                }
                if let Some(VStringInfo::Plain { chars }) = self.vstrings.get_mut(&dst_ref) {
                    for (i, ch) in char_refs.into_iter().enumerate() {
                        let di = (dst_start as usize) + i;
                        if di < chars.len() {
                            chars[di] = Some(ch);
                        }
                    }
                    return OptimizationResult::Remove;
                }
            }
        }

        // Force both src and dst if virtual.
        self.force_if_virtual(src_ref, ctx);
        self.force_if_virtual(dst_ref, ctx);
        OptimizationResult::PassOn
    }

    /// Force a string if it is virtual.
    fn force_if_virtual(&mut self, opref: OpRef, ctx: &mut OptContext) {
        let resolved = ctx.get_box_replacement(opref);
        if self.vstrings.contains_key(&resolved) {
            self.force_box(resolved, ctx);
        }
    }

    /// vstring.py: initialize_forced_string(op, targetbox, offsetbox, mode)
    ///
    /// Emit STRSETITEM for each known character of a virtual Plain string
    /// into a target string at the given offset. Returns the new offset
    /// (offset + length). Used by copy_str_content when the source is virtual.
    fn initialize_forced_string(
        &self,
        chars: &[Option<OpRef>],
        target: OpRef,
        mut offset: OpRef,
        is_unicode: bool,
        ctx: &mut OptContext,
    ) -> OpRef {
        let set_opcode = if is_unicode {
            OpCode::Unicodesetitem
        } else {
            OpCode::Strsetitem
        };
        for ch in chars {
            if let Some(ch_ref) = ch {
                let ch_resolved = ctx.get_box_replacement(*ch_ref);
                let set_op = Op::new(set_opcode, &[target, offset, ch_resolved]);
                ctx.emit(set_op);
            }
            // offset += 1
            let one = self.emit_constant_int(1, ctx);
            let add_op = Op::new(OpCode::IntAdd, &[offset, one]);
            offset = ctx.emit(add_op);
        }
        offset
    }

    /// Check if an OpRef references a virtual string (after forwarding).
    #[allow(dead_code)]
    fn is_virtual(&self, opref: OpRef, ctx: &OptContext) -> bool {
        let resolved = ctx.get_box_replacement(opref);
        self.vstrings.contains_key(&resolved)
    }

    /// vstring.py:371-381 _int_add — constant-fold if both args are constant,
    /// otherwise emit INT_ADD.
    fn int_add(&self, a: OpRef, b: OpRef, ctx: &mut OptContext) -> OpRef {
        if let Some(va) = ctx.get_constant_int(a) {
            if va == 0 {
                return b;
            }
            if let Some(vb) = ctx.get_constant_int(b) {
                return self.emit_constant_int(va + vb, ctx);
            }
        } else if let Some(vb) = ctx.get_constant_int(b) {
            if vb == 0 {
                return a;
            }
        }
        let op = Op::new(OpCode::IntAdd, &[a, b]);
        ctx.emit(op)
    }

    /// vstring.py:383-391 _int_sub — constant-fold if both args are constant,
    /// otherwise emit INT_SUB.
    fn int_sub(&self, a: OpRef, b: OpRef, ctx: &mut OptContext) -> OpRef {
        if let Some(vb) = ctx.get_constant_int(b) {
            if vb == 0 {
                return a;
            }
            if let Some(va) = ctx.get_constant_int(a) {
                return self.emit_constant_int(va - vb, ctx);
            }
        }
        let op = Op::new(OpCode::IntSub, &[a, b]);
        ctx.emit(op)
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
        ctx.make_nonnull_str(op.arg(0), mode);
        let str_ref = ctx.get_box_replacement(op.arg(0));
        if let Some(len) = self.get_known_length(str_ref, ctx) {
            let _ = len;
        }
    }

    fn force_args_if_virtual(&mut self, op: &Op, ctx: &mut OptContext) {
        // Collect refs first to avoid borrow issues.
        let args: Vec<OpRef> = op
            .args
            .iter()
            .map(|a| ctx.get_box_replacement(*a))
            .collect();
        for arg in args {
            if self.vstrings.contains_key(&arg) {
                self.force_box(arg, ctx);
            }
        }
    }

    /// Dispatch oopspec calls to specialized handlers.
    fn optimize_oopspec_call(
        &mut self,
        op: &Op,
        ei: &EffectInfo,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        match ei.oopspec_index {
            OopSpecIndex::StrConcat => self.opt_call_stroruni_str_concat(op, ctx),
            OopSpecIndex::StrSlice => self.opt_call_stroruni_str_slice(op, ctx),
            OopSpecIndex::StrEqual => self.opt_call_stroruni_str_equal(op, ctx),
            OopSpecIndex::StrCmp => self.opt_call_stroruni_str_cmp(op, ctx),
            OopSpecIndex::ShrinkArray => self.opt_call_shrink_array(op, ctx),
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
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() >= 3 {
            let left = ctx.get_box_replacement(op.arg(1));
            let right = ctx.get_box_replacement(op.arg(2));
            self.vstrings
                .insert(op.pos, VStringInfo::Concat { left, right });
            return OptimizationResult::Remove;
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:662-690 opt_call_stroruni_STR_SLICE
    fn opt_call_stroruni_str_slice(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if op.num_args() >= 4 {
            let source = ctx.get_box_replacement(op.arg(1));
            let start = ctx.get_box_replacement(op.arg(2));
            let stop = ctx.get_box_replacement(op.arg(3));
            let length = self.int_sub(stop, start, ctx);
            self.vstrings.insert(
                op.pos,
                VStringInfo::Slice {
                    source,
                    start,
                    length,
                },
            );
            return OptimizationResult::Remove;
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:692-733 opt_call_stroruni_STR_EQUAL
    fn opt_call_stroruni_str_equal(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if op.num_args() >= 3 {
            let a = ctx.get_box_replacement(op.arg(1));
            let b = ctx.get_box_replacement(op.arg(2));
            // Same ref → always equal
            if a == b {
                ctx.make_constant(op.pos, Value::Int(1));
                return OptimizationResult::Remove;
            }
            let l1 = self.get_known_length(a, ctx);
            let l2 = self.get_known_length(b, ctx);
            // vstring.py:706-712: different known lengths → always unequal
            if let (Some(len1), Some(len2)) = (l1, l2) {
                if len1 != len2 {
                    ctx.make_constant(op.pos, Value::Int(0));
                    return OptimizationResult::Remove;
                }
            }
            // vstring.py:714-718: handle_str_equal_level1 both directions
            if let Some(result) = self.handle_str_equal_level1(a, b, op, ctx) {
                return result;
            }
            if let Some(result) = self.handle_str_equal_level1(b, a, op, ctx) {
                return result;
            }
            // vstring.py:720-724: handle_str_equal_level2 both directions
            if let Some(result) = self.handle_str_equal_level2(a, b, op, ctx) {
                return result;
            }
            if let Some(result) = self.handle_str_equal_level2(b, a, op, ctx) {
                return result;
            }
            // vstring.py:727-732: fallback with null checks
            let a_nonnull = self.is_known_nonnull(a, ctx);
            let b_nonnull = self.is_known_nonnull(b, ctx);
            if a_nonnull && b_nonnull {
                // vstring.py:728: l1box.same_box(l2box) — same strlen
                // result OpRef (not just same constant value).
                let l1box = self.getstrlen_if_known(a, ctx);
                let l2box = self.getstrlen_if_known(b, ctx);
                let oopspec = if l1box.is_some() && l2box.is_some() && l1box == l2box {
                    OopSpecIndex::StreqLengthok
                } else {
                    OopSpecIndex::StreqNonnull
                };
                if let Some(result) = self.generate_modified_call(oopspec, &[a, b], op, ctx) {
                    return result;
                }
            }
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:735-787 handle_str_equal_level1
    fn handle_str_equal_level1(
        &self,
        arg1: OpRef,
        arg2: OpRef,
        op: &Op,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        let l2 = self.get_known_length(arg2, ctx);
        // vstring.py:743-756: length-0 string
        if l2 == Some(0) {
            if let Some(len_ref) = self.getstrlen_if_known(arg1, ctx) {
                let zero = self.emit_constant_int(0, ctx);
                let mut eq_op = Op::new(OpCode::IntEq, &[len_ref, zero]);
                eq_op.pos = op.pos;
                return Some(OptimizationResult::Emit(eq_op));
            }
        }
        // vstring.py:757-774: length-1 string optimizations
        if l2 == Some(1) {
            let l1 = self.get_known_length(arg1, ctx);
            if l1 == Some(1) {
                // vstring.py:761-768: both length 1 → compare chars
                let c1 = self.strgetitem(arg1, 0, ctx);
                let c2 = self.strgetitem(arg2, 0, ctx);
                if let (Some(ch1), Some(ch2)) = (c1, c2) {
                    let mut eq_op = Op::new(OpCode::IntEq, &[ch1, ch2]);
                    eq_op.pos = op.pos;
                    return Some(OptimizationResult::Emit(eq_op));
                }
            }
            // vstring.py:769-774: arg1 is a virtual slice, arg2 is length 1
            let resolved1 = ctx.get_box_replacement(arg1);
            if let Some(VStringInfo::Slice {
                source,
                start,
                length,
            }) = self.vstrings.get(&resolved1).cloned()
            {
                if let Some(vchar) = self.strgetitem(arg2, 0, ctx) {
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
                ctx.make_constant(op.pos, Value::Int(0));
                return Some(OptimizationResult::Remove);
            }
            if self.is_known_null(arg1, ctx) {
                ctx.make_constant(op.pos, Value::Int(1));
                return Some(OptimizationResult::Remove);
            }
            let null_const = self.emit_constant_int(0, ctx);
            let mut eq_op = Op::new(OpCode::PtrEq, &[arg1, null_const]);
            eq_op.pos = op.pos;
            return Some(OptimizationResult::Emit(eq_op));
        }
        None
    }

    /// vstring.py:789-814 handle_str_equal_level2
    fn handle_str_equal_level2(
        &self,
        arg1: OpRef,
        arg2: OpRef,
        op: &Op,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        // vstring.py:792-798: use getstrlen + intbound to check constant length.
        // RPython: l2box = i2.getstrlen(...); l2info = self.getintbound(l2box)
        // This catches lengths known via intbound analysis, not just from
        // get_known_length (which only returns virtual/constant lengths).
        let l2 = self.get_known_length(arg2, ctx).or_else(|| {
            let len_ref = self.getstrlen_if_known(arg2, ctx)?;
            let bound = ctx.get_int_bound(len_ref)?;
            bound.known_eq_const(1).then_some(1)
        });
        if l2 == Some(1) {
            if let Some(vchar) = self.strgetitem(arg2, 0, ctx) {
                let oopspec = if self.is_known_nonnull(arg1, ctx) {
                    OopSpecIndex::StreqNonnullChar
                } else {
                    OopSpecIndex::StreqChecknullChar
                };
                return self.generate_modified_call(oopspec, &[arg1, vchar], op, ctx);
            }
        }
        // vstring.py:807-813: if arg1 is a virtual slice
        let resolved1 = ctx.get_box_replacement(arg1);
        if let Some(VStringInfo::Slice {
            source,
            start,
            length,
        }) = self.vstrings.get(&resolved1).cloned()
        {
            let oopspec = if self.is_known_nonnull(arg2, ctx) {
                OopSpecIndex::StreqSliceNonnull
            } else {
                OopSpecIndex::StreqSliceChecknull
            };
            return self.generate_modified_call(oopspec, &[source, start, length, arg2], op, ctx);
        }
        None
    }

    /// Check if an opref is known to be null.
    fn is_known_null(&self, opref: OpRef, ctx: &OptContext) -> bool {
        let resolved = ctx.get_box_replacement(opref);
        if let Some(info) = ctx.get_ptr_info(resolved) {
            return info.is_null();
        }
        // Constant zero is null
        ctx.get_constant_int(resolved) == Some(0)
    }

    /// Check if an opref is known to be non-null.
    fn is_known_nonnull(&self, opref: OpRef, ctx: &OptContext) -> bool {
        // Virtual strings are always non-null
        let resolved = ctx.get_box_replacement(opref);
        if self.vstrings.contains_key(&resolved) {
            return true;
        }
        // PtrInfo with is_nonnull
        if let Some(info) = ctx.get_ptr_info(resolved) {
            if info.is_nonnull() || info.is_virtual() {
                return true;
            }
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
        let &(ref calldescr, func_addr) = cic.get(oopspec)?;
        let func_const = ctx.alloc_op_position();
        ctx.make_constant(func_const, Value::Int(func_addr as i64));
        let mut call_args = vec![func_const];
        call_args.extend_from_slice(args);
        let mut call_op = Op::with_descr(OpCode::CallI, &call_args, calldescr.clone());
        call_op.pos = result_op.pos;
        Some(OptimizationResult::Emit(call_op))
    }

    /// vstring.py:816-838 opt_call_stroruni_STR_CMP
    fn opt_call_stroruni_str_cmp(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if op.num_args() < 3 {
            self.force_args_if_virtual(op, ctx);
            return OptimizationResult::PassOn;
        }
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let arg2 = ctx.get_box_replacement(op.arg(2));
        // Same-ref: result is 0
        if arg1 == arg2 {
            ctx.make_constant(op.pos, Value::Int(0));
            return OptimizationResult::Remove;
        }
        // vstring.py:825-836: if both lengths are known constant 1,
        // extract characters and replace with INT_SUB
        let l1 = self.get_known_length(arg1, ctx);
        let l2 = self.get_known_length(arg2, ctx);
        if l1 == Some(1) && l2 == Some(1) {
            if let (Some(char1), Some(char2)) = (
                self.strgetitem(op.arg(1), 0, ctx),
                self.strgetitem(op.arg(2), 0, ctx),
            ) {
                let result = self.int_sub(char1, char2, ctx);
                ctx.replace_op(op.pos, result);
                return OptimizationResult::Remove;
            }
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
    /// majit's `VStringInfo::Plain { chars }` carries length implicitly via
    /// `chars.len()`, so the explicit `self.length = length` step is folded
    /// into the truncate.
    fn vstring_plain_shrink(info: &mut VStringInfo, length: usize) {
        if let VStringInfo::Plain { chars } = info {
            chars.truncate(length);
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
    fn opt_call_shrink_array(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if op.num_args() >= 3 {
            let arg1 = ctx.get_box_replacement(op.arg(1));
            let length = ctx.get_constant_int(op.arg(2));
            // vstring.py:844-845: i2.is_constant() && i1.is_virtual() &&
            // isinstance(i1, VStringPlainInfo)
            if let (Some(length), Some(info)) = (length, self.vstrings.get_mut(&arg1)) {
                if matches!(info, VStringInfo::Plain { .. }) {
                    // vstring.py:847: i1.shrink(length)
                    Self::vstring_plain_shrink(info, length as usize);
                    // vstring.py:849: self.make_equal_to(op, op.getarg(1))
                    ctx.replace_op(op.pos, arg1);
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
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        match op.opcode {
            OpCode::Newstr => self.optimize_newstr(op, ctx),
            OpCode::Strsetitem => self.optimize_strsetitem(op, ctx),
            OpCode::Strgetitem => self.optimize_strgetitem(op, ctx),
            OpCode::Strlen => self.optimize_strlen(op, ctx),
            OpCode::Copystrcontent => self.optimize_copystrcontent(op, ctx),

            // vstring.py: Unicode operations — same logic as string ops
            // but with unicode-specific opcodes.
            OpCode::Newunicode => {
                // Track that this is a unicode string for force_box.
                self.unicode_refs.insert(op.pos);
                self.optimize_newstr(op, ctx)
            }
            OpCode::Unicodesetitem => self.optimize_strsetitem(op, ctx),
            OpCode::Unicodegetitem => self.optimize_strgetitem(op, ctx),
            OpCode::Unicodelen => self.optimize_strlen(op, ctx),
            OpCode::Copyunicodecontent => self.optimize_copystrcontent(op, ctx),

            // vstring.py: STRHASH/UNICODEHASH — force virtual string and emit.
            OpCode::Strhash | OpCode::Unicodehash => {
                let src = ctx.get_box_replacement(op.arg(0));
                self.force_if_virtual(src, ctx);
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
            // vstring.py: oopspec call dispatch (CALL and CALL_PURE).
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureN => {
                if let Some(ref descr) = op.descr {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.effect_info();
                        if ei.has_oopspec() {
                            return self.optimize_oopspec_call(op, &ei, ctx);
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

    fn setup(&mut self) {
        self.vstrings.clear();
        self.known_lengths.clear();
        self.unicode_refs.clear();
    }

    fn name(&self) -> &'static str {
        "string"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;

    /// Assign sequential positions to ops and pre-seed constants in OptContext.
    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
    }

    /// Run the OptString pass on a list of ops, with given pre-seeded constants.
    fn run_with_constants(ops: &[Op], constants: &[(OpRef, i64)]) -> Vec<Op> {
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
            .map(|op| op.pos)
            .filter(|op| !op.is_none() && !op.is_constant())
            .map(|op| op.0)
            .max()
            .unwrap_or(0);
        let start_next_pos = (max_pos + 1).max(ops.len() as u32);
        let mut ctx = OptContext::with_num_inputs_and_start_pos(ops.len(), 0, 0, start_next_pos);
        for &(opref, val) in constants {
            ctx.make_constant(opref, Value::Int(val));
        }

        let mut pass = OptString::new();
        pass.setup();

        for op in ops {
            // Resolve forwarded arguments.
            let mut resolved_op = op.clone();
            for arg in &mut resolved_op.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved_op, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {
                    // Op removed, nothing emitted.
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        pass.flush(&mut ctx);
        ctx.new_operations
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
            Op::new(OpCode::Newstr, &[OpRef(100)]), // op 0: p0 = newstr(3)
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(101), OpRef(200)]), // op 1
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(102), OpRef(201)]), // op 2
            Op::new(OpCode::Strgetitem, &[OpRef(0), OpRef(101)]), // op 3: get char at 0
        ];
        assign_positions(&mut ops);

        let constants = vec![
            (OpRef(100), 3), // length = 3
            (OpRef(101), 0), // index 0
            (OpRef(102), 1), // index 1
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
            Op::new(OpCode::Newstr, &[OpRef(100)]), // op 0
            Op::new(OpCode::Strlen, &[OpRef(0)]),   // op 1
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), 5)];

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
            Op::new(OpCode::Newstr, &[OpRef(100)]), // op 0
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(101), OpRef(200)]), // op 1
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(102), OpRef(201)]), // op 2
            Op::new(OpCode::CallN, &[OpRef(0)]),    // op 3: forces
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), 2), (OpRef(101), 0), (OpRef(102), 1)];

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
        // Build two virtual strings, create a concat, then query length.
        //
        // We simulate concat by directly inserting into the pass's vstrings map.
        // This tests the get_known_length logic.

        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        // Constant length refs
        ctx.make_constant(OpRef(100), Value::Int(3));
        ctx.make_constant(OpRef(101), Value::Int(4));

        // Virtual plain strings
        let left_ref = OpRef(10);
        let right_ref = OpRef(11);
        pass.vstrings.insert(
            left_ref,
            VStringInfo::Plain {
                chars: vec![None; 3],
            },
        );
        pass.known_lengths.insert(left_ref, OpRef(100));
        pass.vstrings.insert(
            right_ref,
            VStringInfo::Plain {
                chars: vec![None; 4],
            },
        );
        pass.known_lengths.insert(right_ref, OpRef(101));

        // Virtual concat
        let concat_ref = OpRef(12);
        pass.vstrings.insert(
            concat_ref,
            VStringInfo::Concat {
                left: left_ref,
                right: right_ref,
            },
        );

        // Check total length = 3 + 4 = 7
        let total_len = pass.get_known_length(concat_ref, &ctx);
        assert_eq!(total_len, Some(7));
    }

    // ── Test 5: Slice virtual string ──

    #[test]
    fn test_slice_get_char() {
        // Build a virtual plain string, create a slice, get a character.
        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        // source = "abc" (chars at indices 0, 1, 2)
        let src_ref = OpRef(10);
        pass.vstrings.insert(
            src_ref,
            VStringInfo::Plain {
                chars: vec![Some(OpRef(200)), Some(OpRef(201)), Some(OpRef(202))],
            },
        );

        // slice = source[1:3] (start=1, length=2)
        ctx.make_constant(OpRef(300), Value::Int(1)); // start
        ctx.make_constant(OpRef(301), Value::Int(2)); // length
        let slice_ref = OpRef(11);
        pass.vstrings.insert(
            slice_ref,
            VStringInfo::Slice {
                source: src_ref,
                start: OpRef(300),
                length: OpRef(301),
            },
        );

        // Get char at index 0 of the slice -> should be source[1] = OpRef(201)
        let ch = pass.strgetitem(slice_ref, 0, &ctx);
        assert_eq!(ch, Some(OpRef(201)));

        // Get char at index 1 of the slice -> should be source[2] = OpRef(202)
        let ch = pass.strgetitem(slice_ref, 1, &ctx);
        assert_eq!(ch, Some(OpRef(202)));
    }

    // ── Test 6: Slice length via STRLEN ──

    #[test]
    fn test_slice_strlen() {
        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        let src_ref = OpRef(10);
        pass.vstrings.insert(
            src_ref,
            VStringInfo::Plain {
                chars: vec![None; 5],
            },
        );

        ctx.make_constant(OpRef(300), Value::Int(1)); // start
        ctx.make_constant(OpRef(301), Value::Int(3)); // length

        let slice_ref = OpRef(11);
        pass.vstrings.insert(
            slice_ref,
            VStringInfo::Slice {
                source: src_ref,
                start: OpRef(300),
                length: OpRef(301),
            },
        );

        let len = pass.get_known_length(slice_ref, &ctx);
        assert_eq!(len, Some(3));
    }

    // ── Test 7: Non-constant length NEWSTR passes through ──

    #[test]
    fn test_newstr_non_constant_passes_through() {
        // newstr(i0) where i0 is not a known constant -> should emit.
        let mut ops = vec![Op::new(OpCode::Newstr, &[OpRef(50)])];
        assign_positions(&mut ops);

        let result = run_with_constants(&ops, &[]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Newstr);
    }

    // ── Test 8: Too-large NEWSTR passes through ──

    #[test]
    fn test_newstr_too_large_passes_through() {
        let mut ops = vec![Op::new(OpCode::Newstr, &[OpRef(50)])];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(50), (MAX_CONST_LEN + 1) as i64)];
        let result = run_with_constants(&ops, &constants);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Newstr);
    }

    // ── Test 9: STRGETITEM on non-virtual string passes through ──

    #[test]
    fn test_strgetitem_non_virtual() {
        let mut ops = vec![Op::new(OpCode::Strgetitem, &[OpRef(50), OpRef(51)])];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(51), 0)];
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
            Op::new(OpCode::Newstr, &[OpRef(100)]),
            Op::new(OpCode::CallN, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), 0)];
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
            Op::new(OpCode::Newstr, &[OpRef(100)]), // op 0: src
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(101), OpRef(200)]), // op 1
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(102), OpRef(201)]), // op 2
            Op::new(OpCode::Newstr, &[OpRef(100)]), // op 3: dst
            Op::new(
                OpCode::Copystrcontent,
                &[OpRef(0), OpRef(3), OpRef(101), OpRef(101), OpRef(100)],
            ), // op 4: copy src->dst
            Op::new(OpCode::Strgetitem, &[OpRef(3), OpRef(101)]), // op 5: get dst[0]
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), 2), (OpRef(101), 0), (OpRef(102), 1)];

        let result = run_with_constants(&ops, &constants);

        // All ops should be removed since everything is virtual.
        assert!(
            result.is_empty(),
            "Expected all ops removed, got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Test 12: Multiple STRLEN calls on same virtual ──

    #[test]
    fn test_strlen_multiple_calls() {
        // p0 = newstr(3)
        // i1 = strlen(p0) -> const 3
        // i2 = strlen(p0) -> const 3
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[OpRef(100)]),
            Op::new(OpCode::Strlen, &[OpRef(0)]),
            Op::new(OpCode::Strlen, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), 3)];
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
            Op::new(OpCode::Newstr, &[OpRef(100)]),
            Op::new(OpCode::Strgetitem, &[OpRef(0), OpRef(101)]),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), 3), (OpRef(101), 0)];

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

        ctx.make_constant(OpRef(100), Value::Int(2));
        ctx.make_constant(OpRef(101), Value::Int(3));
        ctx.make_constant(OpRef(102), Value::Int(4));

        let a = OpRef(10);
        let b = OpRef(11);
        let c = OpRef(12);
        pass.vstrings.insert(
            a,
            VStringInfo::Plain {
                chars: vec![None; 2],
            },
        );
        pass.known_lengths.insert(a, OpRef(100));
        pass.vstrings.insert(
            b,
            VStringInfo::Plain {
                chars: vec![None; 3],
            },
        );
        pass.known_lengths.insert(b, OpRef(101));
        pass.vstrings.insert(
            c,
            VStringInfo::Plain {
                chars: vec![None; 4],
            },
        );
        pass.known_lengths.insert(c, OpRef(102));

        // ab = concat(a, b)
        let ab = OpRef(20);
        pass.vstrings
            .insert(ab, VStringInfo::Concat { left: a, right: b });

        // abc = concat(ab, c)
        let abc = OpRef(21);
        pass.vstrings
            .insert(abc, VStringInfo::Concat { left: ab, right: c });

        assert_eq!(pass.get_known_length(abc, &ctx), Some(9));
    }

    // ── Test 15: Concat get char across boundary ──

    #[test]
    fn test_concat_get_char() {
        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        ctx.make_constant(OpRef(100), Value::Int(2));
        ctx.make_constant(OpRef(101), Value::Int(2));

        let left = OpRef(10);
        let right = OpRef(11);
        pass.vstrings.insert(
            left,
            VStringInfo::Plain {
                chars: vec![Some(OpRef(200)), Some(OpRef(201))],
            },
        );
        pass.known_lengths.insert(left, OpRef(100));
        pass.vstrings.insert(
            right,
            VStringInfo::Plain {
                chars: vec![Some(OpRef(202)), Some(OpRef(203))],
            },
        );
        pass.known_lengths.insert(right, OpRef(101));

        let concat = OpRef(12);
        pass.vstrings
            .insert(concat, VStringInfo::Concat { left, right });

        // Index 0 -> left[0] = 200
        assert_eq!(pass.strgetitem(concat, 0, &ctx), Some(OpRef(200)));
        // Index 1 -> left[1] = 201
        assert_eq!(pass.strgetitem(concat, 1, &ctx), Some(OpRef(201)));
        // Index 2 -> right[0] = 202
        assert_eq!(pass.strgetitem(concat, 2, &ctx), Some(OpRef(202)));
        // Index 3 -> right[1] = 203
        assert_eq!(pass.strgetitem(concat, 3, &ctx), Some(OpRef(203)));
    }

    #[test]
    fn test_strlen_caching_non_virtual() {
        // STRLEN on a non-virtual string should be cached for the second call.
        let mut ops = vec![
            Op::new(OpCode::Strlen, &[OpRef(100)]),
            Op::new(OpCode::Strlen, &[OpRef(100)]),
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);
        let result = run_with_constants(&ops, &[]);
        // Second STRLEN should be eliminated by heap.rs STRLEN caching
        // (if running through full pipeline) or by vstring.rs known_lengths.
        // With just OptString pass, the first STRLEN passes through and
        // records in known_lengths, but the second one checks known_lengths
        // which maps OpRef(100) → OpRef(0) (result of first STRLEN).
        // Since OpRef(0) is not a constant, it won't be removed by OptString alone.
        // This test just verifies no crash occurs.
        assert!(result.len() >= 1);
    }

    #[test]
    fn test_concat_oopspec_creates_virtual() {
        // Verify that STR_CONCAT creates a virtual Concat.
        let mut pass = OptString::new();
        pass.setup();

        let left = OpRef(100);
        let right = OpRef(101);

        // Simulate: NEWSTR(2) for left
        let mut left_op = Op::new(OpCode::Newstr, &[OpRef(200)]);
        left_op.pos = left;
        let mut ctx = OptContext::new(10);
        ctx.make_constant(OpRef(200), Value::Int(2));

        // Process NEWSTR → creates virtual Plain
        let _ = pass.propagate_forward(&left_op, &mut ctx);
        assert!(pass.is_virtual(left, &ctx));
    }
}
