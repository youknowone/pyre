/// llsupport/regalloc.py + x86/regalloc.py: Register allocation.
///
/// Classes:
///   Lifetime           — regalloc.py:911
///   FixedRegisterPositions — regalloc.py:1025
///   LifetimeManager    — regalloc.py:1054
///   FrameManager       — regalloc.py:62
///   RegisterManager    — regalloc.py:356
///   RegAlloc           — x86/regalloc.py:169
///
/// Functions:
///   compute_vars_longevity — regalloc.py:1173
///   valid_addressing_size  — regalloc.py:1236
///   get_scale              — regalloc.py:1239
use std::collections::HashMap;

use crate::arch::*;
use crate::regloc::*;
use majit_ir::{InputArg, Op, OpCode, OpRef, Type};

// ── Constants ──────────────────────────────────────────────────────

/// regalloc.py:909
pub const UNDEF_POS: i32 = -42;

/// regalloc.py:26-28
pub const SAVE_DEFAULT_REGS: u8 = 0;
pub const SAVE_GCREF_REGS: u8 = 1;
pub const SAVE_ALL_REGS: u8 = 2;

// ── Lifetime ───────────────────────────────────────────────────────

/// regalloc.py:911 Lifetime — liveness information for a single variable.
pub struct Lifetime {
    /// regalloc.py:916 position where the variable is defined
    pub definition_pos: i32,
    /// regalloc.py:918 position where the variable is last used (including failargs/jump)
    pub last_usage: i32,
    /// regalloc.py:921 *real* usages (as op argument, excluding jump/failargs)
    pub real_usages: Option<Vec<i32>>,
    /// regalloc.py:925 positions requiring specific registers
    pub fixed_positions: Option<Vec<(i32, RegLoc)>>,
    /// regalloc.py:929 another Lifetime that wants to share a register
    pub share_with: Option<OpRef>,
    /// regalloc.py:934
    _definition_pos_shared: i32,
    /// regalloc.py:937 current register index into RegisterManager.all_regs (-1 = not in reg)
    pub current_register_index: i32,
    /// regalloc.py:940 frame location where the box currently lives
    pub current_frame_loc: Option<FrameLoc>,
    /// regalloc.py:943 hinted frame location (at the end of the trace)
    pub hint_frame_pos: i32,
}

impl Lifetime {
    pub fn new(definition_pos: i32, last_usage: i32) -> Self {
        Lifetime {
            definition_pos,
            last_usage,
            real_usages: None,
            fixed_positions: None,
            share_with: None,
            _definition_pos_shared: UNDEF_POS,
            current_register_index: -1,
            current_frame_loc: None,
            hint_frame_pos: -1,
        }
    }

    /// regalloc.py:946
    pub fn last_usage_including_sharing(&self, longevity: &LifetimeManager) -> i32 {
        let mut result = self.last_usage;
        let mut share = self.share_with;
        while let Some(opref) = share {
            if let Some(lt) = longevity.get(opref) {
                result = lt.last_usage;
                share = lt.share_with;
            } else {
                break;
            }
        }
        result
    }

    /// regalloc.py:951
    pub fn is_last_real_use_before(&self, position: i32) -> bool {
        match &self.real_usages {
            None => true,
            Some(usages) => *usages.last().unwrap() <= position,
        }
    }

    /// regalloc.py:956 binary search for next real usage after position.
    pub fn next_real_usage(&self, position: i32) -> i32 {
        debug_assert!(position >= self.definition_pos);
        let l = self.real_usages.as_ref().unwrap();
        if position >= *l.last().unwrap() {
            return -1;
        }
        let mut low: usize = 0;
        let mut high: usize = l.len();
        while low < high {
            let mid = low + (high - low) / 2;
            if position < l[mid] {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        l[low]
    }

    /// regalloc.py:972
    pub fn definition_pos_shared(&self) -> i32 {
        if self._definition_pos_shared != UNDEF_POS {
            self._definition_pos_shared
        } else {
            self.definition_pos
        }
    }

    /// regalloc.py:978
    pub fn fixed_register(&mut self, position: i32, reg: RegLoc) -> i32 {
        debug_assert!(self.definition_pos <= position && position <= self.last_usage);
        let res;
        if self.fixed_positions.is_none() {
            self.fixed_positions = Some(Vec::new());
            res = self.definition_pos_shared();
        } else {
            let positions = self.fixed_positions.as_ref().unwrap();
            debug_assert!(position > positions.last().unwrap().0);
            res = positions.last().unwrap().0;
        }
        self.fixed_positions.as_mut().unwrap().push((position, reg));
        res
    }

    /// regalloc.py:992
    pub fn find_fixed_register(&self, opindex: i32, longevity: &LifetimeManager) -> Option<RegLoc> {
        if let Some(ref positions) = self.fixed_positions {
            for &(index, reg) in positions {
                if opindex <= index {
                    return Some(reg);
                }
            }
        }
        if let Some(share_opref) = self.share_with {
            if let Some(share_lt) = longevity.get(share_opref) {
                return share_lt.find_fixed_register(opindex, longevity);
            }
        }
        None
    }

    /// regalloc.py:1001
    pub fn check_invariants(&self) {
        debug_assert!(self.definition_pos <= self.last_usage);
        if let Some(ref usages) = self.real_usages {
            let mut sorted = usages.clone();
            sorted.sort();
            debug_assert_eq!(usages, &sorted);
            debug_assert!(self.last_usage >= *usages.iter().max().unwrap());
            debug_assert!(self.definition_pos < *usages.iter().min().unwrap());
        }
    }
}

impl std::fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{:?}({})",
            self.definition_pos, self.real_usages, self.last_usage
        )?;
        if let Some(ref fp) = self.fixed_positions {
            for &(index, reg) in fp {
                write!(f, " @{} in {:?}", index, reg)?;
            }
        }
        if self.current_register_index >= 0 {
            write!(f, " reg_idx:{}", self.current_register_index)?;
        }
        if self.current_frame_loc.is_some() {
            write!(f, " frame:{:?}", self.current_frame_loc)?;
        }
        if self.hint_frame_pos >= 0 {
            write!(f, " hint:{}", self.hint_frame_pos)?;
        }
        Ok(())
    }
}

// ── FixedRegisterPositions ─────────────────────────────────────────

/// regalloc.py:1025
pub struct FixedRegisterPositions {
    pub register: RegLoc,
    pub index_lifetimes: Vec<(i32, i32)>, // (opindex, definition_pos)
}

impl FixedRegisterPositions {
    pub fn new(register: RegLoc) -> Self {
        FixedRegisterPositions {
            register,
            index_lifetimes: Vec::new(),
        }
    }

    /// regalloc.py:1031
    pub fn fixed_register(&mut self, opindex: i32, definition_pos: i32) {
        if let Some(last) = self.index_lifetimes.last() {
            debug_assert!(opindex > last.0);
        }
        self.index_lifetimes.push((opindex, definition_pos));
    }

    /// regalloc.py:1036
    pub fn free_until_pos(&self, opindex: i32) -> i32 {
        for &(index, definition_pos) in &self.index_lifetimes {
            if opindex <= index {
                if definition_pos >= opindex {
                    return definition_pos;
                } else {
                    return index;
                }
            }
        }
        i32::MAX
    }
}

// ── LifetimeManager ────────────────────────────────────────────────

/// regalloc.py:1054 LifetimeManager — manages Lifetime info for all variables.
pub struct LifetimeManager {
    lifetimes: HashMap<OpRef, Lifetime>,
    /// regalloc.py:1064 maps register → FixedRegisterPositions
    pub fixed_register_use: HashMap<RegLoc, FixedRegisterPositions>,
}

impl LifetimeManager {
    pub fn new() -> Self {
        LifetimeManager {
            lifetimes: HashMap::new(),
            fixed_register_use: HashMap::new(),
        }
    }

    pub fn contains(&self, v: OpRef) -> bool {
        self.lifetimes.contains_key(&v)
    }

    pub fn get(&self, v: OpRef) -> Option<&Lifetime> {
        self.lifetimes.get(&v)
    }

    pub fn get_mut(&mut self, v: OpRef) -> Option<&mut Lifetime> {
        self.lifetimes.get_mut(&v)
    }

    pub fn set(&mut self, v: OpRef, lifetime: Lifetime) {
        self.lifetimes.insert(v, lifetime);
    }

    /// regalloc.py:1066
    pub fn fixed_register(&mut self, opindex: i32, register: RegLoc, var: Option<OpRef>) {
        let definition_pos = match var {
            None => opindex,
            Some(v) => {
                let lt = self.lifetimes.get_mut(&v).unwrap();
                lt.fixed_register(opindex, register)
            }
        };
        self.fixed_register_use
            .entry(register)
            .or_insert_with(|| FixedRegisterPositions::new(register))
            .fixed_register(opindex, definition_pos);
    }

    /// regalloc.py:1079
    pub fn try_use_same_register(&mut self, v0: OpRef, v1: OpRef) {
        let def0 = self.lifetimes[&v0].definition_pos;
        let last0 = self.lifetimes[&v0].last_usage;
        let def1 = self.lifetimes[&v1].definition_pos;
        debug_assert!(def0 < def1);
        if last0 != def1 {
            return; // not supported for now
        }
        let def0_shared = self.lifetimes[&v0].definition_pos_shared();
        self.lifetimes.get_mut(&v0).unwrap().share_with = Some(v1);
        self.lifetimes.get_mut(&v1).unwrap()._definition_pos_shared = def0_shared;
    }

    /// regalloc.py:1091
    pub fn longest_free_reg(&self, position: i32, free_regs: &[RegLoc]) -> (Option<RegLoc>, i32) {
        let mut max_free_pos = position;
        let mut best_reg = None;
        // reverse for compatibility with old code
        for i in (0..free_regs.len()).rev() {
            let reg = free_regs[i];
            match self.fixed_register_use.get(&reg) {
                None => return (Some(reg), i32::MAX),
                Some(fixed_reg_pos) => {
                    let free_until_pos = fixed_reg_pos.free_until_pos(position);
                    if free_until_pos > max_free_pos {
                        best_reg = Some(reg);
                        max_free_pos = free_until_pos;
                    }
                }
            }
        }
        (best_reg, max_free_pos)
    }

    /// regalloc.py:1111
    pub fn free_reg_whole_lifetime(
        &self,
        position: i32,
        v: OpRef,
        free_regs: &[RegLoc],
    ) -> Option<RegLoc> {
        let longevity_var = &self.lifetimes[&v];
        let last_usage = longevity_var.last_usage_including_sharing(self);
        let mut min_fixed_use_after = i32::MAX;
        let mut best_reg = None;
        let mut unfixed_reg = None;
        for &reg in free_regs {
            match self.fixed_register_use.get(&reg) {
                None => {
                    unfixed_reg = Some(reg);
                    continue;
                }
                Some(fixed_reg_pos) => {
                    let use_after = fixed_reg_pos.free_until_pos(position);
                    if use_after <= last_usage {
                        continue; // can't fit
                    }
                    if use_after < min_fixed_use_after {
                        best_reg = Some(reg);
                        min_fixed_use_after = use_after;
                    }
                }
            }
        }
        if best_reg.is_some() {
            return best_reg;
        }
        unfixed_reg
    }

    /// regalloc.py:1138
    pub fn try_pick_free_reg(
        &self,
        position: i32,
        v: OpRef,
        free_regs: &[RegLoc],
    ) -> Option<RegLoc> {
        if free_regs.is_empty() {
            return None;
        }
        let longevity_var = &self.lifetimes[&v];
        // check whether there is a fixed register and whether it's free
        let fixed_reg = longevity_var.find_fixed_register(position, self);
        if let Some(reg) = fixed_reg {
            if free_regs.contains(&reg) {
                return Some(reg);
            }
        }
        // try to find a register that's free for the whole lifetime of v
        let loc = self.free_reg_whole_lifetime(position, v, free_regs);
        if loc.is_some() {
            return loc;
        }
        // can't fit v completely, pick the register free the longest
        let (loc, _free_until) = self.longest_free_reg(position, free_regs);
        loc
    }
}

// ── compute_vars_longevity ─────────────────────────────────────────

/// regalloc.py:1173 compute_vars_longevity — backward liveness analysis.
pub fn compute_vars_longevity(inputargs: &[InputArg], operations: &[Op]) -> LifetimeManager {
    let mut longevity = LifetimeManager::new();

    // regalloc.py:1179 iterate operations in REVERSE
    for i in (0..operations.len()).rev() {
        let op = &operations[i];
        let opnum = op.opcode;
        let opref = op.pos;
        let i = i as i32;

        if !longevity.contains(opref) {
            // regalloc.py:1183-1186: if op.type != 'v' and has_no_side_effect:
            // result not used, operation has no side-effect → dead code
            if opnum.result_type() != Type::Void && opnum.has_no_side_effect() {
                continue;
            }
            longevity.set(opref, Lifetime::new(i, i));
        } else {
            longevity.get_mut(opref).unwrap().definition_pos = i;
        }

        // regalloc.py:1190-1201 process arguments
        for j in 0..op.args.len() {
            let arg = op.args[j];
            if arg.is_constant() {
                continue;
            }
            if !longevity.contains(arg) {
                longevity.set(arg, Lifetime::new(UNDEF_POS, i));
            }
            let lifetime = longevity.get_mut(arg).unwrap();
            // real_usages: exclude JUMP and LABEL
            if opnum != OpCode::Jump && opnum != OpCode::Label {
                if lifetime.real_usages.is_none() {
                    lifetime.real_usages = Some(Vec::new());
                }
                lifetime.real_usages.as_mut().unwrap().push(i);
            }
        }

        // regalloc.py:1202-1208 guard failargs
        if opnum.is_guard() {
            if let Some(ref fail_args) = op.fail_args {
                for &arg in fail_args.iter() {
                    if arg.is_none() {
                        continue; // hole
                    }
                    debug_assert!(!arg.is_constant());
                    if !longevity.contains(arg) {
                        longevity.set(arg, Lifetime::new(UNDEF_POS, i));
                    }
                }
            }
        }
    }

    // regalloc.py:1210-1213 input arguments
    for iarg in inputargs {
        let opref = OpRef(iarg.index);
        if !longevity.contains(opref) {
            longevity.set(opref, Lifetime::new(-1, -1));
        }
    }

    // regalloc.py:1224-1231 reverse real_usages and check invariants
    // We need to iterate over all lifetimes; collect keys first to avoid borrow issues
    let keys: Vec<OpRef> = longevity.lifetimes.keys().copied().collect();
    for opref in keys {
        let lifetime = longevity.get_mut(opref).unwrap();
        if let Some(ref mut usages) = lifetime.real_usages {
            usages.reverse();
        }
        #[cfg(debug_assertions)]
        lifetime.check_invariants();
    }

    longevity
}

// ── FrameManager ───────────────────────────────────────────────────

/// regalloc.py:62 FrameManager — frame slot allocation.
/// Combined with x86/regalloc.py:132 X86FrameManager.
pub struct FrameManager {
    /// regalloc.py:69
    pub current_frame_depth: usize,
    /// regalloc.py:70
    pub boxes_in_frame: Vec<Option<OpRef>>,
    /// x86/regalloc.py:134 base_ofs (byte offset of frame field 0 from ebp)
    pub base_ofs: i32,
}

impl FrameManager {
    /// regalloc.py:68 + x86/regalloc.py:133
    pub fn new(base_ofs: i32) -> Self {
        FrameManager {
            current_frame_depth: 0,
            boxes_in_frame: Vec::new(),
            base_ofs,
        }
    }

    /// regalloc.py:81
    pub fn get_frame_depth(&self) -> usize {
        self.current_frame_depth
    }

    /// regalloc.py:84
    fn _increase_frame_depth(&mut self, incby: usize) {
        self.current_frame_depth += incby;
        for _ in 0..incby {
            self.boxes_in_frame.push(None);
        }
    }

    /// regalloc.py:89
    pub fn get(&self, v: OpRef, longevity: &LifetimeManager) -> Option<FrameLoc> {
        longevity.get(v).and_then(|lt| lt.current_frame_loc)
    }

    /// regalloc.py:95
    pub fn loc(
        &mut self,
        v: OpRef,
        tp: Type,
        must_exist: bool,
        longevity: &mut LifetimeManager,
    ) -> FrameLoc {
        if let Some(loc) = self.get(v, longevity) {
            return loc;
        }
        if must_exist {
            panic!("FrameManager.loc: box {:?} not found", v);
        }
        self.get_new_loc(v, tp, longevity)
    }

    /// regalloc.py:105
    pub fn get_new_loc(&mut self, v: OpRef, tp: Type, longevity: &mut LifetimeManager) -> FrameLoc {
        let size = Self::frame_size(tp);
        let hint = self.get_frame_pos_hint(v, longevity);
        let newloc = self._find_frame_location(size, tp, hint);
        let newloc = match newloc {
            Some(loc) => loc,
            None => {
                // regalloc.py:114-126
                let mut index = self.get_frame_depth();
                if size == 2 && index & 1 == 1 {
                    if index > 0 && self.boxes_in_frame[index - 1].is_none() {
                        index -= 1;
                    } else {
                        index += 1;
                    }
                    let loc = Self::frame_pos(index, tp, self.base_ofs);
                    let needed = index + size;
                    if needed > self.current_frame_depth {
                        self._increase_frame_depth(needed - self.current_frame_depth);
                    }
                    loc
                } else {
                    let loc = Self::frame_pos(index, tp, self.base_ofs);
                    self._increase_frame_depth(size);
                    loc
                }
            }
        };
        self.bind(v, newloc, tp, longevity);
        newloc
    }

    /// regalloc.py:142
    fn _find_frame_location(&self, size: usize, tp: Type, hint: i32) -> Option<FrameLoc> {
        debug_assert!(size == 1 || size == 2);
        if size == 1 {
            // regalloc.py:145 — check 0 <= hint < depth
            if 0 <= hint
                && (hint as usize) < self.current_frame_depth
                && self.boxes_in_frame[hint as usize].is_none()
            {
                return Some(Self::frame_pos(hint as usize, tp, self.base_ofs));
            }
            for i in 0..self.current_frame_depth {
                if self.boxes_in_frame[i].is_none() {
                    return Some(Self::frame_pos(i, tp, self.base_ofs));
                }
            }
            None
        } else {
            let limit = (self.current_frame_depth >> 1) << 1;
            let mut i = 0;
            while i < limit {
                if self.boxes_in_frame[i].is_none() && self.boxes_in_frame[i + 1].is_none() {
                    return Some(Self::frame_pos(i, tp, self.base_ofs));
                }
                i += 2;
            }
            None
        }
    }

    /// regalloc.py:164
    pub fn bind(&mut self, v: OpRef, loc: FrameLoc, tp: Type, longevity: &mut LifetimeManager) {
        let pos = loc.position;
        let size = Self::frame_size(tp);
        if pos + size > self.current_frame_depth {
            self._increase_frame_depth(pos + size - self.current_frame_depth);
        }
        debug_assert!(self.boxes_in_frame[pos].is_none());
        for index in pos..pos + size {
            self.boxes_in_frame[index] = Some(v);
        }
        let lifetime = longevity
            .get_mut(v)
            .expect("bind: variable not in longevity");
        lifetime.current_frame_loc = Some(loc);
    }

    /// regalloc.py:181
    pub fn finish_binding(&self) {
        #[cfg(debug_assertions)]
        self._check_invariants();
    }

    /// regalloc.py:185
    pub fn mark_as_free(&mut self, v: OpRef, tp: Type, longevity: &mut LifetimeManager) {
        let loc = match self.get(v, longevity) {
            Some(loc) => loc,
            None => return, // not in frame
        };
        let lifetime = longevity.get_mut(v).unwrap();
        debug_assert!(lifetime.current_frame_loc.is_some());
        lifetime.current_frame_loc = None;
        let pos = loc.position;
        let size = Self::frame_size(tp);
        debug_assert!(self.boxes_in_frame[pos] == Some(v));
        for index in pos..pos + size {
            self.boxes_in_frame[index] = None;
        }
    }

    /// regalloc.py:201
    pub fn add_frame_pos_hint(&self, v: OpRef, loc: FrameLoc, longevity: &mut LifetimeManager) {
        if let Some(lifetime) = longevity.get_mut(v) {
            lifetime.hint_frame_pos = loc.position as i32;
        }
    }

    /// regalloc.py:206
    pub fn get_frame_pos_hint(&self, v: OpRef, longevity: &LifetimeManager) -> i32 {
        longevity.get(v).map(|lt| lt.hint_frame_pos).unwrap_or(-1)
    }

    /// regalloc.py:212
    fn _check_invariants(&self) {
        debug_assert_eq!(self.boxes_in_frame.len(), self.current_frame_depth);
    }

    // ── x86/regalloc.py:136 X86FrameManager methods ──

    /// x86/regalloc.py:137 frame_pos(i, box_type)
    pub fn frame_pos(position: usize, tp: Type, base_ofs: i32) -> FrameLoc {
        let ebp_offset = get_ebp_ofs(base_ofs, position);
        FrameLoc::new(position, ebp_offset, tp == Type::Float)
    }

    /// x86/regalloc.py:141 frame_size(box_type)
    pub fn frame_size(tp: Type) -> usize {
        // x86_64: always 1 word (IS_X86_32 && FLOAT → 2, but we're 64-bit only)
        let _ = tp;
        1
    }

    /// x86/regalloc.py:145 get_loc_index(loc)
    pub fn get_loc_index(loc: &FrameLoc) -> usize {
        loc.position
    }
}

/// x86/regalloc.py:21 get_ebp_ofs(base_ofs, position)
///
/// RPython: rbp points past the frame header, slots grow downward:
///   -(position+1)*WORD + base_ofs
///
/// Dynasm: rbp points to jitframe start, slots grow upward:
///   (1+position)*WORD + base_ofs
///   jf_ptr[0] = jf_descr, jf_ptr[1+position] = frame slot
///
/// The sign difference is because RPython's GC-managed JITFRAME has
/// rbp pointing to the interior (after gc header + fixed fields),
/// while dynasm's malloc'd jitframe has rbp at the base.
pub fn get_ebp_ofs(base_ofs: i32, position: usize) -> i32 {
    ((1 + position) as i32 * WORD as i32) + base_ofs
}

// ── RegisterManager ────────────────────────────────────────────────

/// regalloc.py:356 RegisterManager — register allocation logic.
///
/// RPython stores `longevity` and `frame_manager` as attributes;
/// in Rust we pass them explicitly to methods to avoid borrow conflicts.
pub struct RegisterManager {
    // ── Class-level configuration (set at construction) ──
    /// regalloc.py:362
    pub all_regs: Vec<RegLoc>,
    /// regalloc.py:363
    pub no_lower_byte_regs: Vec<RegLoc>,
    /// regalloc.py:364
    pub save_around_call_regs: Vec<RegLoc>,
    /// regalloc.py:365
    pub frame_reg: RegLoc,
    /// Type filter: INT/REF for GPR, FLOAT for XMM
    pub box_types: Option<Vec<Type>>,
    /// x86/regalloc.py:52,120 call_result_location — hardcoded return register.
    pub call_result_reg: RegLoc,

    // ── Instance state ──
    /// regalloc.py:369
    pub free_regs: Vec<RegLoc>,
    /// regalloc.py:371 register index → OpRef (None = free)
    pub reg_bindings_list: Vec<Option<OpRef>>,
    /// regalloc.py:373
    pub temp_boxes: Vec<OpRef>,
    /// regalloc.py:375
    pub box_currently_in_frame_reg: Option<OpRef>,
    /// regalloc.py:376
    pub position: i32,
}

impl RegisterManager {
    /// regalloc.py:368
    pub fn new(
        all_regs: Vec<RegLoc>,
        no_lower_byte_regs: Vec<RegLoc>,
        save_around_call_regs: Vec<RegLoc>,
        frame_reg: RegLoc,
        box_types: Option<Vec<Type>>,
        call_result_reg: RegLoc,
    ) -> Self {
        let n = all_regs.len();
        let mut free_regs = all_regs.clone();
        free_regs.reverse(); // regalloc.py:370
        RegisterManager {
            all_regs,
            no_lower_byte_regs,
            save_around_call_regs,
            frame_reg,
            box_types,
            call_result_reg,
            free_regs,
            reg_bindings_list: vec![None; n],
            temp_boxes: Vec::new(),
            box_currently_in_frame_reg: None,
            position: -1,
        }
    }

    // ── RegBindingsDict equivalent ──

    fn _register_index(&self, reg: RegLoc) -> usize {
        self.all_regs
            .iter()
            .position(|r| *r == reg)
            .expect("register not in all_regs")
    }

    /// regalloc.py:393 / RegBindingsDict.get
    pub fn reg_bindings_get(&self, v: OpRef, longevity: &LifetimeManager) -> Option<RegLoc> {
        if let Some(lifetime) = longevity.get(v) {
            let index = lifetime.current_register_index;
            if index >= 0 {
                return Some(self.all_regs[index as usize]);
            }
        }
        None
    }

    /// RegBindingsDict.__contains__
    pub fn reg_bindings_contains(&self, v: OpRef, longevity: &LifetimeManager) -> bool {
        self.reg_bindings_get(v, longevity).is_some()
    }

    /// RegBindingsDict.__setitem__
    pub fn reg_bindings_set(&mut self, v: OpRef, reg: RegLoc, longevity: &mut LifetimeManager) {
        let index = self._register_index(reg);
        let lifetime = longevity
            .get_mut(v)
            .expect("reg_bindings_set: not in longevity");
        lifetime.current_register_index = index as i32;
        self.reg_bindings_list[index] = Some(v);
    }

    /// RegBindingsDict.__delitem__
    pub fn reg_bindings_del(&mut self, v: OpRef, longevity: &mut LifetimeManager) {
        let lifetime = longevity
            .get_mut(v)
            .expect("reg_bindings_del: not in longevity");
        let index = lifetime.current_register_index;
        debug_assert!(index >= 0, "reg_bindings_del: not in register");
        lifetime.current_register_index = -1;
        self.reg_bindings_list[index as usize] = None;
    }

    /// RegBindingsDict.pop
    pub fn reg_bindings_pop(&mut self, v: OpRef, longevity: &mut LifetimeManager) -> RegLoc {
        let lifetime = longevity
            .get_mut(v)
            .expect("reg_bindings_pop: not in longevity");
        let index = lifetime.current_register_index;
        debug_assert!(index >= 0, "reg_bindings_pop: not in register");
        lifetime.current_register_index = -1;
        let reg = self.all_regs[index as usize];
        self.reg_bindings_list[index as usize] = None;
        reg
    }

    /// Iterate (OpRef, RegLoc) bindings — RegBindingsIterItems equivalent.
    pub fn reg_bindings_iter(&self) -> impl Iterator<Item = (OpRef, RegLoc)> + '_ {
        self.reg_bindings_list
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.map(|v| (v, self.all_regs[i])))
    }

    /// Collect current register bindings as Vec.
    pub fn reg_bindings_items(&self) -> Vec<(OpRef, RegLoc)> {
        self.reg_bindings_iter().collect()
    }

    /// Count of active bindings.
    pub fn reg_bindings_len(&self) -> usize {
        self.reg_bindings_list
            .iter()
            .filter(|x| x.is_some())
            .count()
    }

    // ── Core methods ──

    /// regalloc.py:380
    pub fn is_still_alive(&self, v: OpRef, longevity: &LifetimeManager) -> bool {
        longevity
            .get(v)
            .map(|lt| lt.last_usage >= self.position)
            .unwrap_or(false)
    }

    /// regalloc.py:385
    pub fn stays_alive(&self, v: OpRef, longevity: &LifetimeManager) -> bool {
        longevity
            .get(v)
            .map(|lt| lt.last_usage > self.position)
            .unwrap_or(false)
    }

    /// regalloc.py:390
    pub fn next_instruction(&mut self, incr: i32) {
        self.position += incr;
    }

    /// regalloc.py:409
    pub fn possibly_free_var(
        &mut self,
        v: OpRef,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        v_type: Type,
    ) {
        if v.is_constant() {
            return;
        }
        let should_free =
            !longevity.contains(v) || longevity.get(v).unwrap().last_usage <= self.position;
        if should_free {
            if let Some(reg) = self.reg_bindings_get(v, longevity) {
                self.free_regs.push(reg);
                self.reg_bindings_del(v, longevity);
            }
            if self.box_currently_in_frame_reg == Some(v) {
                self.box_currently_in_frame_reg = None;
            }
            fm.mark_as_free(v, v_type, longevity);
        }
    }

    /// regalloc.py:427
    pub fn possibly_free_vars(
        &mut self,
        vars: &[OpRef],
        types: &[Type],
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) {
        for (v, tp) in vars.iter().zip(types.iter()) {
            self.possibly_free_var(*v, longevity, fm, *tp);
        }
    }

    /// regalloc.py:437
    pub fn free_temp_vars(&mut self, longevity: &mut LifetimeManager, fm: &mut FrameManager) {
        let temps: Vec<OpRef> = self.temp_boxes.drain(..).collect();
        for v in temps {
            // Temp vars are always INT type
            self.possibly_free_var(v, longevity, fm, Type::Int);
        }
    }

    /// regalloc.py:460
    pub fn try_allocate_reg(
        &mut self,
        v: OpRef,
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        longevity: &mut LifetimeManager,
    ) -> Option<RegLoc> {
        debug_assert!(!v.is_constant());

        if let Some(selected) = selected_reg {
            // regalloc.py:474-486
            let res = self.reg_bindings_get(v, longevity);
            if let Some(existing) = res {
                if existing == selected {
                    return Some(existing);
                } else {
                    self.reg_bindings_del(v, longevity);
                    self.free_regs.push(existing);
                }
            }
            if self.free_regs.contains(&selected) {
                self.free_regs.retain(|r| *r != selected);
                self.reg_bindings_set(v, selected, longevity);
                return Some(selected);
            }
            return None;
        }

        if need_lower_byte {
            // regalloc.py:487-501
            let loc = self.reg_bindings_get(v, longevity);
            if let Some(reg) = loc {
                if !self.no_lower_byte_regs.contains(&reg) {
                    return Some(reg);
                }
            }
            let free_regs: Vec<RegLoc> = self
                .free_regs
                .iter()
                .filter(|r| !self.no_lower_byte_regs.contains(r))
                .copied()
                .collect();
            let newloc = longevity.try_pick_free_reg(self.position, v, &free_regs);
            if newloc.is_none() {
                return None;
            }
            let newloc = newloc.unwrap();
            self.free_regs.retain(|r| *r != newloc);
            if let Some(old) = loc {
                self.free_regs.push(old);
            }
            self.reg_bindings_set(v, newloc, longevity);
            return Some(newloc);
        }

        // regalloc.py:502-512 — default path
        let res = self.reg_bindings_get(v, longevity);
        if res.is_some() {
            return res;
        }
        let loc = longevity.try_pick_free_reg(self.position, v, &self.free_regs);
        match loc {
            None => None,
            Some(reg) => {
                self.reg_bindings_set(v, reg, longevity);
                self.free_regs.retain(|r| *r != reg);
                Some(reg)
            }
        }
    }

    /// regalloc.py:514 _spill_var
    pub fn _spill_var(
        &mut self,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) -> RegLoc {
        let v_to_spill = self._pick_variable_to_spill(
            forbidden_vars,
            selected_reg,
            need_lower_byte,
            None,
            longevity,
        );
        let loc = self.reg_bindings_get(v_to_spill, longevity).unwrap();
        let tp = self._type_of(v_to_spill);
        self._sync_var_to_stack(v_to_spill, tp, longevity, fm);
        self.reg_bindings_del(v_to_spill, longevity);
        loc
    }

    /// regalloc.py:523 _pick_variable_to_spill
    pub fn _pick_variable_to_spill(
        &self,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        regs: Option<&[OpRef]>,
        longevity: &LifetimeManager,
    ) -> OpRef {
        let binding_keys: Vec<OpRef>;
        let regs = match regs {
            Some(r) => r,
            None => {
                binding_keys = self.reg_bindings_list.iter().filter_map(|x| *x).collect();
                &binding_keys
            }
        };

        let position = self.position;
        let mut cur_max_use_distance: i32 = -1;
        let mut candidate: Option<OpRef> = None;
        let mut cur_max_age_failargs: i32 = -1;
        let mut candidate_from_failargs: Option<OpRef> = None;

        for &next in regs {
            let reg = self.reg_bindings_get(next, longevity).unwrap();
            if forbidden_vars.contains(&next) {
                continue;
            }
            // regalloc.py:544
            if self.temp_boxes.contains(&next) {
                continue;
            }
            if let Some(sel) = selected_reg {
                if reg == sel {
                    return next;
                }
                continue;
            }
            if need_lower_byte && self.no_lower_byte_regs.contains(&reg) {
                continue;
            }
            let lifetime = longevity.get(next).unwrap();
            if lifetime.is_last_real_use_before(position) {
                // regalloc.py:554-561
                let max_age = lifetime.last_usage;
                if cur_max_age_failargs < max_age {
                    cur_max_age_failargs = max_age;
                    candidate_from_failargs = Some(next);
                }
            } else {
                // regalloc.py:563-566
                let use_distance = lifetime.next_real_usage(position) - position;
                if cur_max_use_distance < use_distance {
                    cur_max_use_distance = use_distance;
                    candidate = Some(next);
                }
            }
        }
        if let Some(c) = candidate_from_failargs {
            return c;
        }
        if let Some(c) = candidate {
            return c;
        }
        panic!("NoVariableToSpill");
    }

    /// regalloc.py:573
    pub fn force_allocate_reg(
        &mut self,
        v: OpRef,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) -> RegLoc {
        let loc = self.try_allocate_reg(v, selected_reg, need_lower_byte, longevity);
        if let Some(reg) = loc {
            return reg;
        }
        let spilled_loc =
            self._spill_var(forbidden_vars, selected_reg, need_lower_byte, longevity, fm);
        let prev_loc = self.reg_bindings_get(v, longevity);
        if let Some(prev) = prev_loc {
            self.free_regs.push(prev);
        }
        self.reg_bindings_set(v, spilled_loc, longevity);
        spilled_loc
    }

    /// regalloc.py:597
    pub fn force_allocate_frame_reg(&mut self, v: OpRef) {
        debug_assert!(self.box_currently_in_frame_reg.is_none());
        self.box_currently_in_frame_reg = Some(v);
    }

    /// regalloc.py:602
    pub fn force_spill_var(
        &mut self,
        var: OpRef,
        tp: Type,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) {
        self._sync_var_to_stack(var, tp, longevity, fm);
        if let Some(reg) = self.reg_bindings_get(var, longevity) {
            self.reg_bindings_del(var, longevity);
            self.free_regs.push(reg);
        }
    }

    /// regalloc.py:611
    pub fn loc(
        &self,
        v: OpRef,
        must_exist: bool,
        longevity: &LifetimeManager,
        fm: &FrameManager,
        constants: &HashMap<u32, i64>,
    ) -> Loc {
        if v.is_constant() {
            return self.convert_to_imm(v, constants);
        }
        if let Some(reg) = self.reg_bindings_get(v, longevity) {
            return Loc::Reg(reg);
        }
        if self.box_currently_in_frame_reg == Some(v) {
            return Loc::Reg(self.frame_reg);
        }
        // regalloc.py:622 → frame_manager.loc(box, must_exist)
        if let Some(frame_loc) = fm.get(v, longevity) {
            return Loc::Frame(frame_loc);
        }
        if must_exist {
            panic!("RegisterManager.loc: box {:?} not found", v);
        }
        Loc::Immed(ImmedLoc::new(0))
    }

    /// regalloc.py:625 return_constant
    pub fn return_constant(
        &mut self,
        v: OpRef,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        constants: &HashMap<u32, i64>,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        pending_moves: &mut Vec<(Loc, Loc)>,
    ) -> Loc {
        debug_assert!(v.is_constant());
        let immloc = self.convert_to_imm(v, constants);
        if let Some(selected) = selected_reg {
            if self.free_regs.contains(&selected) {
                pending_moves.push((immloc, Loc::Reg(selected)));
                return Loc::Reg(selected);
            }
            let spilled = self._spill_var(forbidden_vars, Some(selected), false, longevity, fm);
            self.free_regs.push(spilled);
            pending_moves.push((immloc, Loc::Reg(spilled)));
            return Loc::Reg(spilled);
        }
        immloc
    }

    /// regalloc.py:642
    pub fn make_sure_var_in_reg(
        &mut self,
        v: OpRef,
        _tp: Type,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        constants: &HashMap<u32, i64>,
        pending_moves: &mut Vec<(Loc, Loc)>,
    ) -> Loc {
        if v.is_constant() {
            return self.return_constant(
                v,
                forbidden_vars,
                selected_reg,
                constants,
                longevity,
                fm,
                pending_moves,
            );
        }
        let prev_loc = self.loc(v, true, longevity, fm, constants);
        if matches!(prev_loc, Loc::Reg(r) if r == self.frame_reg) && selected_reg.is_none() {
            return prev_loc;
        }
        let reg = self.force_allocate_reg(
            v,
            forbidden_vars,
            selected_reg,
            need_lower_byte,
            longevity,
            fm,
        );
        let new_loc = Loc::Reg(reg);
        if !loc_eq(&prev_loc, &new_loc) {
            pending_moves.push((prev_loc, new_loc));
        }
        new_loc
    }

    /// regalloc.py:661
    fn _reallocate_from_to(
        &mut self,
        from_v: OpRef,
        to_v: OpRef,
        longevity: &mut LifetimeManager,
    ) -> RegLoc {
        let reg = self.reg_bindings_pop(from_v, longevity);
        self.reg_bindings_set(to_v, reg, longevity);
        reg
    }

    /// regalloc.py:667
    pub fn force_result_in_reg(
        &mut self,
        result_v: OpRef,
        v: OpRef,
        tp: Type,
        forbidden_vars: &[OpRef],
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        constants: &HashMap<u32, i64>,
        pending_moves: &mut Vec<(Loc, Loc)>,
    ) -> Loc {
        if v.is_constant() {
            let result_loc =
                self.force_allocate_reg(result_v, forbidden_vars, None, false, longevity, fm);
            let imm = self.convert_to_imm(v, constants);
            pending_moves.push((imm, Loc::Reg(result_loc)));
            return Loc::Reg(result_loc);
        }
        let v_keeps_living = longevity.get(v).unwrap().last_usage > self.position;
        // regalloc.py:685 — two cases where we allocate a new register
        let not_in_reg = !self.reg_bindings_contains(v, longevity);
        if not_in_reg || (v_keeps_living && !self.free_regs.is_empty()) {
            let v_loc = self.loc(v, false, longevity, fm, constants);
            let result_loc =
                self.force_allocate_reg(result_v, forbidden_vars, None, false, longevity, fm);
            let new_loc = Loc::Reg(result_loc);
            pending_moves.push((v_loc, new_loc));
            return new_loc;
        }
        if v_keeps_living {
            // regalloc.py:692 — no free registers, spill v
            self._sync_var_to_stack(v, tp, longevity, fm);
        }
        let reg = self._reallocate_from_to(v, result_v, longevity);
        Loc::Reg(reg)
    }

    /// regalloc.py:696
    pub fn _sync_var_to_stack(
        &mut self,
        v: OpRef,
        tp: Type,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) {
        if fm.get(v, longevity).is_none() {
            // Not yet in frame — allocate frame slot and record move
            let _to = fm.loc(v, tp, false, longevity);
            // Note: The actual regalloc_mov emission is handled by the assembler.
            // RPython: self.assembler.regalloc_mov(reg, to)
        }
        // otherwise it's clean (already on stack)
    }

    /// regalloc.py:706 _bc_spill
    fn _bc_spill(
        &mut self,
        v: OpRef,
        tp: Type,
        new_free_regs: &mut Vec<RegLoc>,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) {
        self._sync_var_to_stack(v, tp, longevity, fm);
        let reg = self.reg_bindings_pop(v, longevity);
        new_free_regs.push(reg);
    }

    /// regalloc.py:710
    pub fn before_call(
        &mut self,
        force_store: &[OpRef],
        save_all_regs: u8,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        pending_moves: &mut Vec<(Loc, Loc)>,
        value_types: &HashMap<u32, Type>,
    ) {
        self.spill_or_move_registers_before_call(
            &self.save_around_call_regs.clone(),
            force_store,
            save_all_regs,
            longevity,
            fm,
            pending_moves,
            value_types,
        );
    }

    /// regalloc.py:714
    pub fn spill_or_move_registers_before_call(
        &mut self,
        save_sublist: &[RegLoc],
        force_store: &[OpRef],
        save_all_regs: u8,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        pending_moves: &mut Vec<(Loc, Loc)>,
        value_types: &HashMap<u32, Type>,
    ) {
        let mut new_free_regs: Vec<RegLoc> = Vec::new();
        let mut move_or_spill: Vec<OpRef> = Vec::new();

        // regalloc.py:774 iterate current bindings
        let items: Vec<(OpRef, RegLoc)> = self.reg_bindings_items();
        for (v, reg) in items {
            let max_age = longevity.get(v).unwrap().last_usage;
            let v_type = value_types.get(&v.0).copied().unwrap_or(Type::Int);
            if !force_store.contains(&v) && max_age <= self.position {
                // variable dies
                self.reg_bindings_del(v, longevity);
                new_free_regs.push(reg);
                continue;
            }

            if save_all_regs == SAVE_ALL_REGS {
                // regalloc.py:782-784
                self._bc_spill(v, v_type, &mut new_free_regs, longevity, fm);
            } else if save_all_regs == SAVE_GCREF_REGS && v_type == Type::Ref {
                // regalloc.py:786-788 — spill GC ptrs only
                self._bc_spill(v, v_type, &mut new_free_regs, longevity, fm);
            } else if !save_sublist.contains(&reg) {
                continue; // callee-saved: fine where it is
            } else {
                move_or_spill.push(v);
            }
        }

        if !move_or_spill.is_empty() {
            // regalloc.py:799
            let free_regs: Vec<RegLoc> = self
                .free_regs
                .iter()
                .filter(|r| !self.save_around_call_regs.contains(r))
                .copied()
                .collect();
            // regalloc.py:802-805
            while move_or_spill.len() > free_regs.len() {
                let v =
                    self._pick_variable_to_spill(&[], None, false, Some(&move_or_spill), longevity);
                self._bc_spill(v, Type::Int, &mut new_free_regs, longevity, fm);
                move_or_spill.retain(|x| *x != v);
            }
            // regalloc.py:807-821
            for v in &move_or_spill {
                let new_reg = loop {
                    let r = self.free_regs.pop().expect("no free register for move");
                    if self.save_around_call_regs.contains(&r) {
                        new_free_regs.push(r);
                        continue;
                    }
                    break r;
                };
                let old_reg = self.reg_bindings_get(*v, longevity).unwrap();
                pending_moves.push((Loc::Reg(old_reg), Loc::Reg(new_reg)));
                self.reg_bindings_set(*v, new_reg, longevity);
                new_free_regs.push(old_reg);
            }
        }

        // regalloc.py:826-827 re-add in reverse
        while let Some(r) = new_free_regs.pop() {
            self.free_regs.push(r);
        }
    }

    /// regalloc.py:829
    pub fn after_call(&mut self, v: OpRef, longevity: &mut LifetimeManager) -> RegLoc {
        let r = self.call_result_location();
        self.reg_bindings_set(v, r, longevity);
        self.free_regs.retain(|fr| *fr != r);
        r
    }

    // ── x86-specific methods ──

    /// x86/regalloc.py:55 convert_to_imm
    pub fn convert_to_imm(&self, v: OpRef, constants: &HashMap<u32, i64>) -> Loc {
        debug_assert!(v.is_constant());
        let val = constants.get(&v.const_index()).copied().unwrap_or(0);
        Loc::Immed(ImmedLoc::new(val))
    }

    /// x86/regalloc.py:52 call_result_location → eax (GPR) or xmm0 (XMM).
    pub fn call_result_location(&self) -> RegLoc {
        self.call_result_reg
    }

    /// Infer type for an OpRef (INT for GPR manager, FLOAT for XMM).
    fn _type_of(&self, _v: OpRef) -> Type {
        match &self.box_types {
            Some(types) if types.contains(&Type::Float) => Type::Float,
            _ => Type::Int,
        }
    }
}

// ── RegAlloc ───────────────────────────────────────────────────────

/// x86/regalloc.py:374 walk_operations output: one entry per operation.
/// The assembler uses these to generate machine code.
#[derive(Debug)]
pub enum RegAllocOp {
    /// regalloc_perform(op_index, arglocs, result_loc)
    Perform {
        op_index: usize,
        arglocs: Vec<Loc>,
        result_loc: Option<Loc>,
    },
    /// regalloc_perform_guard(op_index, arglocs, result_loc, faillocs)
    PerformGuard {
        op_index: usize,
        arglocs: Vec<Loc>,
        result_loc: Option<Loc>,
        faillocs: Vec<Option<Loc>>,
    },
    /// regalloc_perform_discard(op_index, arglocs)
    PerformDiscard { op_index: usize, arglocs: Vec<Loc> },
    /// Register move: src → dst (from spill/reload/register-register moves)
    Move { src: Loc, dst: Loc },
    /// Skip (dead operation, no-op)
    Skip,
}

/// x86/regalloc.py:169 RegAlloc — main x86 register allocator.
pub struct RegAlloc {
    /// Lifetime information for all variables.
    pub longevity: LifetimeManager,
    /// GPR register manager — x86/regalloc.py:45 X86RegisterManager
    pub rm: RegisterManager,
    /// XMM register manager — x86/regalloc.py:77 X86XMMRegisterManager
    pub xrm: RegisterManager,
    /// Frame manager — x86/regalloc.py:132 X86FrameManager
    pub fm: FrameManager,
    /// Constants map (OpRef const_index → i64 value).
    pub constants: HashMap<u32, i64>,
    /// Pending register moves to be emitted by the assembler.
    /// Each entry is (source_loc, dest_loc).
    pub pending_moves: Vec<(Loc, Loc)>,
    /// OpRef → Type mapping for type dispatch.
    pub value_types: HashMap<u32, Type>,
}

impl RegAlloc {
    /// x86/regalloc.py:170
    pub fn new(constants: HashMap<u32, i64>) -> Self {
        // We create with empty longevity/managers; they get initialized in _prepare.
        let longevity = LifetimeManager::new();
        let rm = Self::make_gpr_manager();
        let xrm = Self::make_xmm_manager();
        let fm = FrameManager::new(0);
        RegAlloc {
            longevity,
            rm,
            xrm,
            fm,
            constants,
            pending_moves: Vec::new(),
            value_types: HashMap::new(),
        }
    }

    /// x86/regalloc.py:65-75 X86_64_RegisterManager configuration.
    fn make_gpr_manager() -> RegisterManager {
        // x86/regalloc.py:67
        let all_regs = vec![
            ECX, EAX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R13, R14, R15,
        ];
        // x86/regalloc.py:71
        let no_lower_byte_regs = vec![];
        // x86/regalloc.py:72
        let save_around_call_regs = vec![EAX, ECX, EDX, ESI, EDI, R8, R9, R10];
        let frame_reg = EBP;
        RegisterManager::new(
            all_regs,
            no_lower_byte_regs,
            save_around_call_regs,
            frame_reg,
            Some(vec![Type::Int, Type::Ref]),
            EAX, // x86/regalloc.py:52 call_result_location → eax
        )
    }

    /// x86/regalloc.py:77-121,123-128 X86_64_XMMRegisterManager configuration.
    fn make_xmm_manager() -> RegisterManager {
        // x86/regalloc.py:79 all_regs = [xmm0..xmm7]
        // x86/regalloc.py:123-128 X86_64_XMMRegisterManager extends to xmm15,
        // but xmm15 is reserved for scratch.
        let all_regs = vec![
            XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13,
            XMM14,
        ];
        // x86/regalloc.py:81 save_around_call_regs = all_regs
        let save_around_call_regs = all_regs.clone();
        let frame_reg = EBP;
        RegisterManager::new(
            all_regs,
            vec![],
            save_around_call_regs,
            frame_reg,
            Some(vec![Type::Float]),
            XMM0, // x86/regalloc.py:120 call_result_location → xmm0
        )
    }

    /// x86/regalloc.py:176 _prepare
    pub fn _prepare(&mut self, inputargs: &[InputArg], operations: &[Op]) {
        self.longevity = compute_vars_longevity(inputargs, operations);
        let base_ofs = 0i32; // TODO: cpu.get_baseofs_of_frame_field()
        self.fm = FrameManager::new(base_ofs);
        self.rm = Self::make_gpr_manager();
        self.xrm = Self::make_xmm_manager();
        self.value_types.clear();
        for iarg in inputargs {
            self.value_types.insert(iarg.index, iarg.tp);
        }
    }

    /// x86/regalloc.py:209 prepare_loop
    pub fn prepare_loop(&mut self, inputargs: &[InputArg], operations: &[Op]) {
        self._prepare(inputargs, operations);
        self._set_initial_bindings(inputargs);
        // Free input variables that are dead at position 0
        for iarg in inputargs {
            let tp = iarg.tp;
            let opref = OpRef(iarg.index);
            if tp == Type::Float {
                self.xrm
                    .possibly_free_var(opref, &mut self.longevity, &mut self.fm, tp);
            } else {
                self.rm
                    .possibly_free_var(opref, &mut self.longevity, &mut self.fm, tp);
            }
        }
    }

    /// x86/regalloc.py:221 prepare_bridge
    pub fn prepare_bridge(&mut self, inputargs: &[InputArg], arglocs: &[Loc], operations: &[Op]) {
        self._prepare(inputargs, operations);
        self._update_bindings(arglocs, inputargs);
    }

    /// regalloc.py:861 _set_initial_bindings — place all inputargs in frame slots.
    fn _set_initial_bindings(&mut self, inputargs: &[InputArg]) {
        for iarg in inputargs {
            let opref = OpRef(iarg.index);
            let _loc = self.fm.get_new_loc(opref, iarg.tp, &mut self.longevity);
        }
    }

    /// x86/regalloc.py:291 _update_bindings — bind bridge inputargs to their locations.
    fn _update_bindings(&mut self, locs: &[Loc], inputargs: &[InputArg]) {
        let mut used: HashMap<RegLoc, ()> = HashMap::new();

        // x86/regalloc.py:295-312
        for (iarg, loc) in inputargs.iter().zip(locs.iter()) {
            let v = OpRef(iarg.index);
            let tp = iarg.tp;
            match loc {
                Loc::Reg(reg) => {
                    if tp == Type::Float {
                        self.xrm.reg_bindings_set(v, *reg, &mut self.longevity);
                        used.insert(*reg, ());
                    } else if *reg == EBP {
                        // x86/regalloc.py:305-307
                        debug_assert!(self.rm.box_currently_in_frame_reg.is_none());
                        self.rm.box_currently_in_frame_reg = Some(v);
                    } else {
                        self.rm.reg_bindings_set(v, *reg, &mut self.longevity);
                        used.insert(*reg, ());
                    }
                }
                Loc::Frame(floc) => {
                    self.fm.bind(v, *floc, tp, &mut self.longevity);
                }
                _ => {
                    self.fm.bind(
                        v,
                        FrameManager::frame_pos(0, tp, self.fm.base_ofs),
                        tp,
                        &mut self.longevity,
                    );
                }
            }
        }

        // x86/regalloc.py:313-320 rebuild free_regs from scratch
        self.rm.free_regs = Vec::new();
        for &reg in &self.rm.all_regs {
            if !used.contains_key(&reg) {
                self.rm.free_regs.push(reg);
            }
        }
        self.xrm.free_regs = Vec::new();
        for &reg in &self.xrm.all_regs {
            if !used.contains_key(&reg) {
                self.xrm.free_regs.push(reg);
            }
        }

        // x86/regalloc.py:321
        for iarg in inputargs {
            let opref = OpRef(iarg.index);
            self.possibly_free_var(opref, iarg.tp);
        }
        // x86/regalloc.py:322
        self.fm.finish_binding();
    }

    // ── Location dispatch (type-based routing to rm or xrm) ──

    /// x86/regalloc.py:291 loc(v)
    pub fn loc(&self, v: OpRef, tp: Type) -> Loc {
        if tp == Type::Float {
            self.xrm
                .loc(v, false, &self.longevity, &self.fm, &self.constants)
        } else {
            self.rm
                .loc(v, false, &self.longevity, &self.fm, &self.constants)
        }
    }

    /// x86/regalloc.py:299 possibly_free_var(v)
    pub fn possibly_free_var(&mut self, v: OpRef, tp: Type) {
        if tp == Type::Float {
            self.xrm
                .possibly_free_var(v, &mut self.longevity, &mut self.fm, tp);
        } else {
            self.rm
                .possibly_free_var(v, &mut self.longevity, &mut self.fm, tp);
        }
    }

    /// x86/regalloc.py:305 possibly_free_vars_for_op(op)
    pub fn possibly_free_vars_for_op(&mut self, op: &Op, value_types: &HashMap<u32, Type>) {
        for &arg in &op.args {
            if !arg.is_constant() && !arg.is_none() {
                let tp = value_types.get(&arg.0).copied().unwrap_or(Type::Int);
                self.possibly_free_var(arg, tp);
            }
        }
    }

    /// x86/regalloc.py:316 make_sure_var_in_reg
    pub fn make_sure_var_in_reg(
        &mut self,
        v: OpRef,
        tp: Type,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
    ) -> Loc {
        if tp == Type::Float {
            self.xrm.make_sure_var_in_reg(
                v,
                tp,
                forbidden_vars,
                selected_reg,
                need_lower_byte,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
                &mut self.pending_moves,
            )
        } else {
            self.rm.make_sure_var_in_reg(
                v,
                tp,
                forbidden_vars,
                selected_reg,
                need_lower_byte,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
                &mut self.pending_moves,
            )
        }
    }

    /// x86/regalloc.py:325 force_allocate_reg
    pub fn force_allocate_reg(
        &mut self,
        v: OpRef,
        tp: Type,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
    ) -> RegLoc {
        if tp == Type::Float {
            self.xrm.force_allocate_reg(
                v,
                forbidden_vars,
                selected_reg,
                need_lower_byte,
                &mut self.longevity,
                &mut self.fm,
            )
        } else {
            self.rm.force_allocate_reg(
                v,
                forbidden_vars,
                selected_reg,
                need_lower_byte,
                &mut self.longevity,
                &mut self.fm,
            )
        }
    }

    /// x86/regalloc.py:338 force_spill_var
    pub fn force_spill_var(&mut self, v: OpRef, tp: Type) {
        if tp == Type::Float {
            self.xrm
                .force_spill_var(v, tp, &mut self.longevity, &mut self.fm);
        } else {
            self.rm
                .force_spill_var(v, tp, &mut self.longevity, &mut self.fm);
        }
    }

    // ── Perform / emit ──

    /// Drain pending_moves as Move entries.
    fn flush_moves(&mut self, output: &mut Vec<RegAllocOp>) {
        for (src, dst) in self.pending_moves.drain(..) {
            output.push(RegAllocOp::Move { src, dst });
        }
    }

    /// x86/regalloc.py:667 perform
    fn perform(
        &mut self,
        op_index: usize,
        arglocs: Vec<Loc>,
        result_loc: Option<Loc>,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.flush_moves(output);
        output.push(RegAllocOp::Perform {
            op_index,
            arglocs,
            result_loc,
        });
    }

    /// x86/regalloc.py:671 perform_guard
    fn perform_guard(
        &mut self,
        op: &Op,
        op_index: usize,
        arglocs: Vec<Loc>,
        result_loc: Option<Loc>,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.flush_moves(output);
        let faillocs = self.locs_for_fail(op);
        output.push(RegAllocOp::PerformGuard {
            op_index,
            arglocs,
            result_loc,
            faillocs,
        });
    }

    /// x86/regalloc.py:675 perform_discard
    fn perform_discard(
        &mut self,
        op_index: usize,
        arglocs: Vec<Loc>,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.flush_moves(output);
        output.push(RegAllocOp::PerformDiscard { op_index, arglocs });
    }

    /// x86/regalloc.py:682 locs_for_fail
    pub fn locs_for_fail(&self, guard_op: &Op) -> Vec<Option<Loc>> {
        let fail_args = match &guard_op.fail_args {
            Some(fa) => fa,
            None => return Vec::new(),
        };
        let mut locs = Vec::with_capacity(fail_args.len());
        for &arg in fail_args.iter() {
            if arg.is_none() {
                locs.push(None);
                continue;
            }
            if let Some(reg) = self.rm.reg_bindings_get(arg, &self.longevity) {
                locs.push(Some(Loc::Reg(reg)));
            } else if let Some(reg) = self.xrm.reg_bindings_get(arg, &self.longevity) {
                locs.push(Some(Loc::Reg(reg)));
            } else if let Some(floc) = self.fm.get(arg, &self.longevity) {
                locs.push(Some(Loc::Frame(floc)));
            } else {
                locs.push(None);
            }
        }
        locs
    }

    // ── Type resolution ──

    /// Get the type of an OpRef from value_types, op result, or default.
    fn tp(&self, v: OpRef) -> Type {
        self.value_types.get(&v.0).copied().unwrap_or(Type::Int)
    }

    // ── walk_operations + consider_* ──

    /// x86/regalloc.py:374 walk_operations — main dispatch loop.
    pub fn walk_operations(
        &mut self,
        inputargs: &[InputArg],
        operations: &[Op],
    ) -> Vec<RegAllocOp> {
        let mut output = Vec::with_capacity(operations.len());

        for (i, op) in operations.iter().enumerate() {
            self.rm.position = i as i32;
            self.xrm.position = i as i32;

            // Record result type before dispatch (needed by tp() lookups)
            if !op.pos.is_none() && op.opcode.result_type() != Type::Void {
                self.value_types.insert(op.pos.0, op.opcode.result_type());
            }

            // x86/regalloc.py:383-386 skip dead ops
            if op.opcode.has_no_side_effect() && !self.longevity.contains(op.pos) {
                self._free_op_vars(op);
                output.push(RegAllocOp::Skip);
                continue;
            }

            // x86/regalloc.py:390 dispatch to consider_* method
            self._dispatch(op, i, &mut output);

            // x86/regalloc.py:391 possibly_free_vars_for_op
            self._free_op_vars(op);
        }

        // x86/regalloc.py:400-401 free inputargs
        for iarg in inputargs {
            let opref = OpRef(iarg.index);
            self.possibly_free_var(opref, iarg.tp);
        }

        output
    }

    /// Free args and result of an op (x86/regalloc.py:308).
    fn _free_op_vars(&mut self, op: &Op) {
        for &arg in &op.args {
            if !arg.is_constant() && !arg.is_none() {
                let tp = self.tp(arg);
                self.possibly_free_var(arg, tp);
            }
        }
        // Free the result itself if it's dead
        if !op.pos.is_none() && !op.pos.is_constant() {
            let tp = op.opcode.result_type();
            if tp != Type::Void {
                self.possibly_free_var(op.pos, tp);
            }
        }
    }

    /// x86/regalloc.py oplist dispatch
    fn _dispatch(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        match op.opcode {
            // ── Integer binary (symmetric) ──
            OpCode::IntMul
            | OpCode::IntAnd
            | OpCode::IntOr
            | OpCode::IntXor
            | OpCode::IntMulOvf
            | OpCode::IntAddOvf => {
                self.consider_binop_symm(op, i, output);
            }
            // ── Integer binary (non-symmetric) ──
            OpCode::IntSubOvf => {
                self.consider_binop(op, i, output);
            }
            // x86/regalloc.py:575 consider_int_sub — LEA when -const fits 32 bits.
            OpCode::IntSub => {
                self.consider_int_sub(op, i, output);
            }
            // ── Integer add (may use LEA) ──
            OpCode::IntAdd | OpCode::NurseryPtrIncrement => {
                self.consider_int_add(op, i, output);
            }
            // ── Shifts (need ecx) ──
            OpCode::IntLshift | OpCode::IntRshift | OpCode::UintRshift => {
                self.consider_int_lshift(op, i, output);
            }
            // ── Unary integer ──
            OpCode::IntNeg | OpCode::IntInvert => {
                self.consider_unary_int(op, i, output);
            }
            // ── Integer comparisons ──
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::IntEq
            | OpCode::IntNe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe
            | OpCode::PtrEq
            | OpCode::PtrNe
            | OpCode::InstancePtrEq
            | OpCode::InstancePtrNe => {
                self.consider_compop(op, i, output);
            }
            // ── Integer misc ──
            OpCode::IntIsTrue | OpCode::IntIsZero => {
                self.consider_unary_int(op, i, output);
            }
            OpCode::IntForceGeZero => {
                self.consider_unary_int(op, i, output);
            }
            OpCode::IntFloorDiv | OpCode::IntMod => {
                self.consider_binop(op, i, output);
            }
            // x86/regalloc.py:591 consider_uint_mul_high — uses eax and edx
            OpCode::UintMulHigh => {
                self.consider_uint_mul_high(op, i, output);
            }
            OpCode::IntSignext => {
                self.consider_int_signext(op, i, output);
            }

            // ── Float binary ──
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                self.consider_float_op(op, i, output);
            }
            // ── Float unary ──
            OpCode::FloatNeg | OpCode::FloatAbs => {
                self.consider_float_unary(op, i, output);
            }
            // ── Float comparisons ──
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe => {
                self.consider_float_cmp(op, i, output);
            }
            // ── Casts ──
            OpCode::CastIntToFloat => {
                self.consider_cast_int_to_float(op, i, output);
            }
            OpCode::CastFloatToInt => {
                self.consider_cast_float_to_int(op, i, output);
            }
            OpCode::CastFloatToSinglefloat | OpCode::CastSinglefloatToFloat => {
                self.consider_float_unary(op, i, output);
            }
            OpCode::ConvertFloatBytesToLonglong | OpCode::ConvertLonglongBytesToFloat => {
                self.consider_same_as(op, i, output);
            }
            OpCode::CastPtrToInt | OpCode::CastIntToPtr | OpCode::CastOpaquePtr => {
                self.consider_same_as(op, i, output);
            }

            // ── Guards ──
            // x86/regalloc.py:440-443
            OpCode::GuardTrue
            | OpCode::VecGuardTrue
            | OpCode::GuardFalse
            | OpCode::VecGuardFalse
            | OpCode::GuardNonnull
            | OpCode::GuardIsnull => {
                self.consider_guard_cc(op, i, output);
            }
            OpCode::GuardValue => {
                self.consider_guard_value(op, i, output);
            }
            OpCode::GuardClass | OpCode::GuardNonnullClass | OpCode::GuardGcType => {
                self.consider_guard_class(op, i, output);
            }
            // x86/regalloc.py:455,492-494
            OpCode::GuardNoException
            | OpCode::GuardNoOverflow
            | OpCode::GuardOverflow
            | OpCode::GuardNotForced
            | OpCode::GuardNotForced2 => {
                self.consider_guard_no_args(op, i, output);
            }
            // x86/regalloc.py:458
            OpCode::GuardNotInvalidated => {
                self.consider_guard_no_args(op, i, output);
            }
            // x86/regalloc.py:468
            OpCode::GuardException => {
                self.consider_guard_exception(op, i, output);
            }
            _ if op.opcode.is_guard() => {
                self.consider_guard_no_args(op, i, output);
            }

            // ── Same-as / identity ──
            OpCode::SameAsI
            | OpCode::SameAsR
            | OpCode::SameAsF
            | OpCode::LoadFromGcTable
            | OpCode::VirtualRefR => {
                self.consider_same_as(op, i, output);
            }

            // ── Memory loads ──
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldGcPureI
            | OpCode::GetfieldGcPureR
            | OpCode::GetfieldGcPureF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF => {
                self.consider_getfield(op, i, output);
            }
            OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF
            | OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawR
            | OpCode::GetarrayitemRawF => {
                self.consider_getarrayitem(op, i, output);
            }
            OpCode::ArraylenGc => {
                self.consider_getfield(op, i, output);
            }
            OpCode::GcLoadI | OpCode::GcLoadR | OpCode::GcLoadF => {
                self.consider_gc_load(op, i, output);
            }
            OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
                self.consider_gc_load_indexed(op, i, output);
            }
            OpCode::RawLoadI | OpCode::RawLoadF => {
                self.consider_gc_load(op, i, output);
            }
            OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF => {
                self.consider_getinteriorfield(op, i, output);
            }

            // ── Memory stores ──
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                self.consider_setfield(op, i, output);
            }
            OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                self.consider_setarrayitem(op, i, output);
            }
            OpCode::GcStore => {
                self.consider_gc_store(op, i, output);
            }
            OpCode::GcStoreIndexed => {
                self.consider_gc_store_indexed(op, i, output);
            }
            OpCode::RawStore => {
                self.consider_gc_store(op, i, output);
            }
            OpCode::SetinteriorfieldGc | OpCode::SetinteriorfieldRaw => {
                self.consider_setinteriorfield(op, i, output);
            }
            OpCode::ZeroArray => {
                self.consider_discard_3args(op, i, output);
            }
            OpCode::Strsetitem | OpCode::Unicodesetitem => {
                self.consider_discard_3args(op, i, output);
            }
            OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                self.consider_discard_nargs(op, i, output);
            }

            // ── Calls ──
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilR
            | OpCode::CallReleaseGilF
            | OpCode::CallReleaseGilN => {
                self.consider_call(op, i, output);
            }
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                self.consider_call(op, i, output);
            }
            OpCode::CondCallN => {
                self.consider_discard_nargs(op, i, output);
            }
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                self.consider_call(op, i, output);
            }

            // ── Allocation ──
            OpCode::New
            | OpCode::NewWithVtable
            | OpCode::NewArray
            | OpCode::NewArrayClear
            | OpCode::Newstr
            | OpCode::Newunicode => {
                self.consider_call(op, i, output);
            }

            // ── Control flow ──
            OpCode::Jump => {
                self.consider_jump(op, i, output);
            }
            OpCode::Label => {
                self.consider_label(op, i, output);
            }
            OpCode::Finish => {
                self.consider_finish(op, i, output);
            }

            // ── Misc result ops ──
            OpCode::ForceToken => {
                self.consider_force_token(op, i, output);
            }
            OpCode::Strlen | OpCode::Unicodelen => {
                self.consider_getfield(op, i, output);
            }
            OpCode::Strgetitem | OpCode::Unicodegetitem => {
                self.consider_getarrayitem(op, i, output);
            }
            OpCode::LoadEffectiveAddress => {
                self.consider_load_effective_address(op, i, output);
            }
            OpCode::SaveException | OpCode::SaveExcClass => {
                self.consider_no_arg_result(op, i, output);
            }
            // x86/regalloc.py:486
            OpCode::RestoreException => {
                self.consider_restore_exception(op, i, output);
            }

            // ── No-ops ──
            OpCode::JitDebug
            | OpCode::DebugMergePoint
            | OpCode::RecordExactClass
            | OpCode::RecordExactValueR
            | OpCode::RecordExactValueI
            | OpCode::RecordKnownResult
            | OpCode::QuasiimmutField
            | OpCode::AssertNotNone
            | OpCode::Keepalive
            | OpCode::EnterPortalFrame
            | OpCode::LeavePortalFrame
            | OpCode::IncrementDebugCounter
            | OpCode::VirtualRefFinish
            | OpCode::ForceSpill
            | OpCode::CondCallGcWb
            | OpCode::CondCallGcWbArray => {
                output.push(RegAllocOp::Skip);
            }

            _ => {
                // Unknown op: skip
                output.push(RegAllocOp::Skip);
            }
        }
    }

    // ── consider_* methods ──

    /// x86/regalloc.py:527 _consider_binop_part
    fn _consider_binop_part(&mut self, op: &Op, symm: bool) -> (Loc, Loc) {
        let mut x = op.args[0];
        let mut y = op.args[1];
        let xloc = self.loc(x, self.tp(x));
        let mut argloc = self.loc(y, self.tp(y));

        // x86/regalloc.py:536-542 symmetry optimization:
        // if x is not in a reg, but y is, and x lives longer while y dies,
        // swap the role of x and y
        if symm && !xloc.is_reg() && argloc.is_reg() {
            let x_lives_longer = !self.longevity.contains(x)
                || self.longevity.get(x).unwrap().last_usage > self.rm.position;
            let y_dies = self
                .longevity
                .get(y)
                .map(|lt| lt.last_usage == self.rm.position)
                .unwrap_or(false);
            if x_lives_longer && y_dies {
                std::mem::swap(&mut x, &mut y);
                argloc = self.loc(y, self.tp(y));
            }
        }

        let tp = self.tp(x);
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let loc = self.rm.force_result_in_reg(
            op.pos,
            x,
            tp,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        (loc, argloc)
    }

    /// x86/regalloc.py:548 _consider_binop
    fn consider_binop(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let (loc, argloc) = self._consider_binop_part(op, false);
        self.perform(i, vec![loc, argloc], Some(loc), output);
    }

    /// x86/regalloc.py:552 _consider_binop_symm
    fn consider_binop_symm(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let (loc, argloc) = self._consider_binop_part(op, true);
        self.perform(i, vec![loc, argloc], Some(loc), output);
    }

    /// x86/regalloc.py:556 _consider_lea
    fn _consider_lea(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let x = op.args[0];
        let loc = self.make_sure_var_in_reg(x, self.tp(x), &[], None, false);
        // make it possible to have argloc be == loc if x dies
        self.possibly_free_var(x, self.tp(x));
        let argloc = self.loc(op.args[1], self.tp(op.args[1]));
        let resloc = Loc::Reg(self.force_allocate_reg(op.pos, Type::Int, &[], None, false));
        self.perform(i, vec![loc, argloc], Some(resloc), output);
    }

    /// x86/regalloc.py:566 consider_int_add — LEA when const fits 32 bits.
    fn consider_int_add(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let y = op.args[1];
        if y.is_constant() {
            let val = self.constants.get(&y.const_index()).copied().unwrap_or(0);
            if fits_in_32bits(val) {
                return self._consider_lea(op, i, output);
            }
        }
        self.consider_binop_symm(op, i, output);
    }

    /// x86/regalloc.py:575 consider_int_sub
    fn consider_int_sub(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let y = op.args[1];
        if y.is_constant() {
            let val = self.constants.get(&y.const_index()).copied().unwrap_or(0);
            if fits_in_32bits(-val) {
                return self._consider_lea(op, i, output);
            }
        }
        self.consider_binop(op, i, output);
    }

    /// x86/regalloc.py:624 consider_int_lshift (shift operations need ecx)
    fn consider_int_lshift(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let y = op.args[1];
        let loc2 = if y.is_constant() {
            self.rm.convert_to_imm(y, &self.constants)
        } else {
            self.make_sure_var_in_reg(y, Type::Int, &[], Some(ECX), false)
        };
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let loc1 = self.rm.force_result_in_reg(
            op.pos,
            op.args[0],
            Type::Int,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, vec![loc1, loc2], Some(loc1), output);
    }

    /// x86/regalloc.py int_neg / int_invert / int_is_true / int_is_zero / int_signext
    fn consider_unary_int(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let loc = self.rm.force_result_in_reg(
            op.pos,
            op.args[0],
            Type::Int,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, vec![loc], Some(loc), output);
    }

    /// x86/regalloc.py:591 consider_uint_mul_high
    fn consider_uint_mul_high(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let mut arg1 = op.args[0];
        let mut arg2 = op.args[1];
        // x86/regalloc.py:594 — optimized for (box, const)
        if arg1.is_constant() {
            std::mem::swap(&mut arg1, &mut arg2);
        }
        // arg2 must be in eax
        self.make_sure_var_in_reg(arg2, Type::Int, &[], Some(EAX), false);
        let l1 = self.loc(arg1, Type::Int);
        // eax will be trash after the operation
        self.possibly_free_var(arg2, Type::Int);
        // allocate temporary in eax (will be trashed by MUL)
        let tmp = OpRef(u32::MAX - 1); // temp var
        if !self.longevity.contains(tmp) {
            self.longevity
                .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        }
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(EAX),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        // result in edx
        self.rm.force_allocate_reg(
            op.pos,
            &[],
            Some(EDX),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.perform(i, vec![l1], Some(Loc::Reg(EDX)), output);
    }

    /// x86/regalloc.py:618 consider_int_signext
    fn consider_int_signext(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let argloc = self.loc(op.args[0], Type::Int);
        let numbytesloc = self.loc(op.args[1], Type::Int);
        let resloc = Loc::Reg(self.force_allocate_reg(op.pos, Type::Int, &[], None, false));
        self.perform(i, vec![argloc, numbytesloc], Some(resloc), output);
    }

    /// x86/regalloc.py:636 _consider_compop
    fn consider_compop(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let vx = op.args[0];
        let vy = op.args[1];
        let mut arglocs = vec![self.loc(vx, self.tp(vx)), self.loc(vy, self.tp(vy))];
        // x86/regalloc.py:640-644
        let vx_in_reg = self.rm.reg_bindings_contains(vx, &self.longevity);
        let vy_in_reg = self.rm.reg_bindings_contains(vy, &self.longevity);
        if !vx_in_reg && !vy_in_reg && !vx.is_constant() && !vy.is_constant() {
            arglocs[0] = self.make_sure_var_in_reg(vx, Type::Int, &[], None, false);
        }
        // x86/regalloc.py:645 force_allocate_reg_or_cc
        // For now, allocate a register (CC optimization is assembler-level)
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, Type::Int, &[], None, false));
        self.perform(i, arglocs, Some(result_loc), output);
    }

    /// x86/regalloc.py:435 _consider_guard_cc
    fn consider_guard_cc(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let arg = op.args[0];
        let loc = self.make_sure_var_in_reg(arg, self.tp(arg), &[], None, false);
        self.perform_guard(op, i, vec![loc], None, output);
    }

    /// x86/regalloc.py:496 consider_guard_value
    fn consider_guard_value(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let x = self.make_sure_var_in_reg(op.args[0], self.tp(op.args[0]), &[], None, false);
        let y = self.loc(op.args[1], self.tp(op.args[1]));
        self.perform_guard(op, i, vec![x, y], None, output);
    }

    /// x86/regalloc.py:503 consider_guard_class
    fn consider_guard_class(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let x = self.make_sure_var_in_reg(op.args[0], Type::Ref, &[], None, false);
        let y = self.loc(op.args[1], Type::Int);
        self.perform_guard(op, i, vec![x, y], None, output);
    }

    /// x86/regalloc.py:468 consider_guard_exception
    fn consider_guard_exception(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let loc = self.make_sure_var_in_reg(op.args[0], Type::Ref, &[], None, false);
        // x86/regalloc.py:470-471 TempVar for scratch register
        let tmp = OpRef(u32::MAX - 2);
        if !self.longevity.contains(tmp) {
            self.longevity
                .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        }
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let loc1 = Loc::Reg(self.rm.force_allocate_reg(
            tmp,
            &args,
            None,
            false,
            &mut self.longevity,
            &mut self.fm,
        ));
        // x86/regalloc.py:473-477
        let resloc = if self.longevity.contains(op.pos) {
            let mut forbidden = args.clone();
            forbidden.push(tmp);
            Some(Loc::Reg(self.rm.force_allocate_reg(
                op.pos,
                &forbidden,
                None,
                false,
                &mut self.longevity,
                &mut self.fm,
            )))
        } else {
            None
        };
        self.perform_guard(op, i, vec![loc, loc1], resloc, output);
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
    }

    /// x86/regalloc.py:486 consider_restore_exception
    fn consider_restore_exception(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let loc0 = self.make_sure_var_in_reg(op.args[0], Type::Ref, &args, None, false);
        let loc1 = self.make_sure_var_in_reg(op.args[1], Type::Ref, &args, None, false);
        self.perform_discard(i, vec![loc0, loc1], output);
    }

    /// Guards with no arguments (guard_no_exception, guard_not_forced, etc.)
    fn consider_guard_no_args(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        self.perform_guard(op, i, vec![], None, output);
    }

    /// x86/regalloc.py:445 consider_finish
    fn consider_finish(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let locs = if !op.args.is_empty() {
            let tp = self.tp(op.args[0]);
            vec![self.make_sure_var_in_reg(op.args[0], tp, &[], None, false)]
        } else {
            vec![]
        };
        self.perform(i, locs, None, output);
    }

    /// x86/regalloc.py same_as / identity operations
    fn consider_same_as(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let tp = op.opcode.result_type();
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let loc = if tp == Type::Float {
            self.xrm.force_result_in_reg(
                op.pos,
                op.args[0],
                tp,
                &args,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
                &mut self.pending_moves,
            )
        } else {
            self.rm.force_result_in_reg(
                op.pos,
                op.args[0],
                tp,
                &args,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
                &mut self.pending_moves,
            )
        };
        self.perform(i, vec![loc], Some(loc), output);
    }

    /// x86/regalloc.py:661 _consider_float_op
    fn consider_float_op(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let loc1 = self.xrm.loc(
            op.args[1],
            false,
            &self.longevity,
            &self.fm,
            &self.constants,
        );
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let loc0 = self.xrm.force_result_in_reg(
            op.pos,
            op.args[0],
            Type::Float,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, vec![loc0, loc1], Some(loc0), output);
    }

    /// x86/regalloc.py float_neg / float_abs
    fn consider_float_unary(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let loc = self.xrm.force_result_in_reg(
            op.pos,
            op.args[0],
            Type::Float,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, vec![loc], Some(loc), output);
    }

    /// x86/regalloc.py:672 _consider_float_cmp
    fn consider_float_cmp(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let vx = op.args[0];
        let vy = op.args[1];
        let mut arglocs = vec![self.loc(vx, Type::Float), self.loc(vy, Type::Float)];
        let vx_in_reg = self.xrm.reg_bindings_contains(vx, &self.longevity);
        let vy_in_reg = self.xrm.reg_bindings_contains(vy, &self.longevity);
        if !vx_in_reg && !vy_in_reg && !vx.is_constant() {
            arglocs[0] = self.make_sure_var_in_reg(vx, Type::Float, &[], None, false);
        }
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, Type::Int, &[], None, false));
        self.perform(i, arglocs, Some(result_loc), output);
    }

    /// x86/regalloc.py cast_int_to_float
    fn consider_cast_int_to_float(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let loc0 = self.make_sure_var_in_reg(op.args[0], Type::Int, &[], None, false);
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, Type::Float, &[], None, false));
        self.perform(i, vec![loc0], Some(result_loc), output);
    }

    /// x86/regalloc.py cast_float_to_int
    fn consider_cast_float_to_int(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let loc0 = self.make_sure_var_in_reg(op.args[0], Type::Float, &[], None, false);
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, Type::Int, &[], None, false));
        self.perform(i, vec![loc0], Some(result_loc), output);
    }

    /// Memory load: getfield pattern (1 arg → result)
    fn consider_getfield(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let base_loc = self.make_sure_var_in_reg(op.args[0], Type::Ref, &[], None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, tp, &[], None, false));
        self.perform(i, vec![base_loc], Some(result_loc), output);
    }

    /// Memory load: getarrayitem pattern (2 args → result)
    fn consider_getarrayitem(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let base_loc = self.make_sure_var_in_reg(op.args[0], Type::Ref, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(op.args[1], Type::Int, &args, None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, tp, &[], None, false));
        self.perform(i, vec![base_loc, index_loc], Some(result_loc), output);
    }

    /// Memory load: getinteriorfield (3 args → result)
    fn consider_getinteriorfield(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let base_loc = self.make_sure_var_in_reg(op.args[0], Type::Ref, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(op.args[1], Type::Int, &args, None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, tp, &[], None, false));
        self.perform(i, vec![base_loc, index_loc], Some(result_loc), output);
    }

    /// x86/regalloc.py:1154 _consider_gc_load
    fn consider_gc_load(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let base_loc = self.make_sure_var_in_reg(op.args[0], Type::Ref, &args, None, false);
        let ofs_loc = self.loc(op.args[1], Type::Int);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, tp, &[], None, false));
        self.perform(i, vec![base_loc, ofs_loc], Some(result_loc), output);
    }

    /// x86/regalloc.py:1173 _consider_gc_load_indexed
    fn consider_gc_load_indexed(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let base_loc = self.make_sure_var_in_reg(op.args[0], Type::Ref, &args, None, false);
        let ofs_loc = self.make_sure_var_in_reg(op.args[1], Type::Int, &args, None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, tp, &[], None, false));
        let mut locs = vec![base_loc, ofs_loc];
        // scale, offset, size args passed as immediates
        for &arg in &op.args[2..] {
            locs.push(self.loc(arg, Type::Int));
        }
        self.perform(i, locs, Some(result_loc), output);
    }

    /// Memory store: setfield pattern (2 args: base, value)
    fn consider_setfield(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let base_loc = self.make_sure_var_in_reg(op.args[0], Type::Ref, &args, None, false);
        let tp_val = self.tp(op.args[1]);
        let val_loc = self.make_sure_var_in_reg(op.args[1], tp_val, &args, None, false);
        self.perform_discard(i, vec![base_loc, val_loc], output);
    }

    /// Memory store: setarrayitem pattern (3 args: base, index, value)
    fn consider_setarrayitem(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let base_loc = self.make_sure_var_in_reg(op.args[0], Type::Ref, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(op.args[1], Type::Int, &args, None, false);
        let tp_val = self.tp(op.args[2]);
        let val_loc = self.make_sure_var_in_reg(op.args[2], tp_val, &args, None, false);
        self.perform_discard(i, vec![base_loc, index_loc, val_loc], output);
    }

    /// Memory store: setinteriorfield
    fn consider_setinteriorfield(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        self.consider_setarrayitem(op, i, output);
    }

    /// x86/regalloc.py:1110 consider_gc_store
    fn consider_gc_store(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let base_loc = self.make_sure_var_in_reg(op.args[0], Type::Ref, &args, None, false);
        let ofs_loc = self.make_sure_var_in_reg(op.args[1], Type::Int, &args, None, false);
        let tp_val = self.tp(op.args[2]);
        let val_loc = self.make_sure_var_in_reg(op.args[2], tp_val, &args, None, false);
        let mut locs = vec![base_loc, ofs_loc, val_loc];
        // size arg
        if op.args.len() > 3 {
            locs.push(self.loc(op.args[3], Type::Int));
        }
        self.perform_discard(i, locs, output);
    }

    /// x86/regalloc.py:1127 consider_gc_store_indexed
    fn consider_gc_store_indexed(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let base_loc = self.make_sure_var_in_reg(op.args[0], Type::Ref, &args, None, false);
        let ofs_loc = self.make_sure_var_in_reg(op.args[1], Type::Int, &args, None, false);
        let tp_val = self.tp(op.args[2]);
        let val_loc = self.make_sure_var_in_reg(op.args[2], tp_val, &args, None, false);
        let mut locs = vec![base_loc, ofs_loc, val_loc];
        // scale, offset, size args
        for &arg in &op.args[3..] {
            locs.push(self.loc(arg, Type::Int));
        }
        self.perform_discard(i, locs, output);
    }

    /// x86/regalloc.py:824 _call — save registers around call.
    fn consider_call(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        // Save all caller-saved registers
        self.rm.before_call(
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &self.value_types,
        );
        self.xrm.before_call(
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &self.value_types,
        );

        let mut arglocs = Vec::new();
        for &arg in &op.args {
            let tp = self.tp(arg);
            arglocs.push(self.loc(arg, tp));
        }

        let result_tp = op.opcode.result_type();
        let result_loc = if result_tp != Type::Void {
            let r = if result_tp == Type::Float {
                self.xrm.after_call(op.pos, &mut self.longevity)
            } else {
                self.rm.after_call(op.pos, &mut self.longevity)
            };
            Some(Loc::Reg(r))
        } else {
            None
        };
        self.perform(i, arglocs, result_loc, output);
    }

    /// x86/regalloc.py:1303 consider_jump
    fn consider_jump(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let mut locs = Vec::new();
        for &arg in &op.args {
            let tp = self.tp(arg);
            locs.push(self.loc(arg, tp));
        }
        self.perform(i, locs, None, output);
    }

    /// x86/regalloc.py:1360 consider_label
    fn consider_label(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let mut locs = Vec::new();
        for &arg in &op.args {
            let tp = self.tp(arg);
            locs.push(self.loc(arg, tp));
        }
        self.perform(i, locs, None, output);
    }

    /// force_token: result = frame pointer (EBP)
    fn consider_force_token(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, Type::Ref, &[], None, false));
        self.perform(i, vec![], Some(result_loc), output);
    }

    /// load_effective_address: all args in regs
    fn consider_load_effective_address(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let mut locs = Vec::new();
        for &arg in &op.args {
            locs.push(self.loc(arg, Type::Int));
        }
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, Type::Int, &[], None, false));
        self.perform(i, locs, Some(result_loc), output);
    }

    /// No-arg result (save_exception, save_exc_class)
    fn consider_no_arg_result(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos, tp, &[], None, false));
        self.perform(i, vec![], Some(result_loc), output);
    }

    /// Discard op with 3 args (zero_array, strsetitem, etc.)
    fn consider_discard_3args(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args.iter().copied().collect();
        let mut locs = Vec::new();
        for &arg in &args {
            let tp = self.tp(arg);
            locs.push(self.make_sure_var_in_reg(arg, tp, &args, None, false));
        }
        self.perform_discard(i, locs, output);
    }

    /// Generic discard with N args
    fn consider_discard_nargs(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let mut locs = Vec::new();
        for &arg in &op.args {
            let tp = self.tp(arg);
            locs.push(self.loc(arg, tp));
        }
        self.perform_discard(i, locs, output);
    }

    /// Advance position in both register managers.
    pub fn next_instruction(&mut self) {
        self.rm.next_instruction(1);
        self.xrm.next_instruction(1);
    }

    /// Get final frame depth.
    pub fn get_final_frame_depth(&self) -> usize {
        self.fm.get_frame_depth()
    }

    /// Drain pending moves.
    pub fn take_pending_moves(&mut self) -> Vec<(Loc, Loc)> {
        std::mem::take(&mut self.pending_moves)
    }
}

// ── Utility functions ──────────────────────────────────────────────

/// regalloc.py:1236
pub fn valid_addressing_size(size: usize) -> bool {
    size == 1 || size == 2 || size == 4 || size == 8
}

/// regalloc.py:1239
pub fn get_scale(size: usize) -> u8 {
    debug_assert!(valid_addressing_size(size));
    if size < 4 {
        (size - 1) as u8 // 1, 2 => 0, 1
    } else {
        ((size >> 2) + 1) as u8 // 4, 8 => 2, 3
    }
}

/// rx86.py fits_in_32bits
pub fn fits_in_32bits(value: i64) -> bool {
    value == value as i32 as i64
}

/// Compare two Loc values for equality.
fn loc_eq(a: &Loc, b: &Loc) -> bool {
    match (a, b) {
        (Loc::Reg(ra), Loc::Reg(rb)) => ra == rb,
        (Loc::Frame(fa), Loc::Frame(fb)) => fa.position == fb.position,
        (Loc::Immed(ia), Loc::Immed(ib)) => ia.value == ib.value,
        _ => false,
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{InputArg, Op, OpCode, OpRef, Type};

    fn make_op(opcode: OpCode, pos: u32, args: &[OpRef]) -> Op {
        Op {
            opcode,
            args: args.iter().copied().collect(),
            descr: None,
            pos: OpRef(pos),
            fail_args: None,
            fail_arg_types: None,
            rd_pendingfields: None,
            rd_resume_position: -1,
            rd_numb: None,
            rd_consts: None,
            rd_virtuals: None,
            vecinfo: None,
        }
    }

    fn make_guard(opcode: OpCode, pos: u32, args: &[OpRef], fail_args: &[OpRef]) -> Op {
        let mut op = make_op(opcode, pos, args);
        op.fail_args = Some(fail_args.iter().copied().collect());
        op
    }

    #[test]
    fn test_compute_vars_longevity_simple() {
        // i0 = inputarg
        // i1 = int_add(i0, i0)  [pos 0]
        // guard_true(i1)         [pos 1, fail_args=[i0]]
        // jump(i1)               [pos 2]
        let i0 = OpRef(100);
        let i1 = OpRef(0);
        let _i2 = OpRef(1);
        let _i3 = OpRef(2);

        let inputargs = vec![InputArg {
            index: i0.0,
            tp: Type::Int,
        }];
        let ops = vec![
            make_op(OpCode::IntAdd, 0, &[i0, i0]),
            make_guard(OpCode::GuardTrue, 1, &[i1], &[i0]),
            make_op(OpCode::Jump, 2, &[i1]),
        ];

        let longevity = compute_vars_longevity(&inputargs, &ops);

        // i0: defined at -1 (input), last used at 1 (guard failarg), real usages: [0, 0]
        // (used twice as arg to int_add at position 0)
        let lt_i0 = longevity.get(i0).unwrap();
        assert_eq!(lt_i0.last_usage, 1);
        assert_eq!(lt_i0.real_usages.as_ref().unwrap(), &[0, 0]);

        // i1: defined at 0, last used at 2 (jump), real usages: [1]
        let lt_i1 = longevity.get(i1).unwrap();
        assert_eq!(lt_i1.definition_pos, 0);
        assert_eq!(lt_i1.last_usage, 2);
        assert_eq!(lt_i1.real_usages.as_ref().unwrap(), &[1]);
    }

    #[test]
    fn test_lifetime_next_real_usage() {
        let mut lt = Lifetime::new(0, 10);
        lt.real_usages = Some(vec![2, 5, 8]);

        assert_eq!(lt.next_real_usage(0), 2);
        assert_eq!(lt.next_real_usage(1), 2);
        assert_eq!(lt.next_real_usage(2), 5);
        assert_eq!(lt.next_real_usage(4), 5);
        assert_eq!(lt.next_real_usage(5), 8);
        assert_eq!(lt.next_real_usage(8), -1);
    }

    #[test]
    fn test_frame_manager_allocate() {
        let mut longevity = LifetimeManager::new();
        let v0 = OpRef(0);
        let v1 = OpRef(1);
        let v2 = OpRef(2);
        longevity.set(v0, Lifetime::new(0, 10));
        longevity.set(v1, Lifetime::new(1, 10));
        longevity.set(v2, Lifetime::new(2, 10));

        let mut fm = FrameManager::new(0);
        let loc0 = fm.get_new_loc(v0, Type::Int, &mut longevity);
        let loc1 = fm.get_new_loc(v1, Type::Int, &mut longevity);
        let loc2 = fm.get_new_loc(v2, Type::Int, &mut longevity);

        assert_eq!(loc0.position, 0);
        assert_eq!(loc1.position, 1);
        assert_eq!(loc2.position, 2);
        assert_eq!(fm.get_frame_depth(), 3);
    }

    #[test]
    fn test_register_manager_allocate() {
        let mut longevity = LifetimeManager::new();
        let v0 = OpRef(0);
        let v1 = OpRef(1);
        longevity.set(v0, Lifetime::new(0, 5));
        longevity.set(v1, Lifetime::new(1, 5));

        let mut rm = RegisterManager::new(
            vec![EAX, ECX, EDX],
            vec![],
            vec![EAX, ECX, EDX],
            EBP,
            Some(vec![Type::Int, Type::Ref]),
            EAX,
        );
        rm.position = 0;

        let reg0 = rm.try_allocate_reg(v0, None, false, &mut longevity);
        assert!(reg0.is_some());
        let reg0 = reg0.unwrap();

        let reg1 = rm.try_allocate_reg(v1, None, false, &mut longevity);
        assert!(reg1.is_some());
        let reg1 = reg1.unwrap();

        assert_ne!(reg0, reg1);
        assert_eq!(rm.reg_bindings_get(v0, &longevity), Some(reg0));
        assert_eq!(rm.reg_bindings_get(v1, &longevity), Some(reg1));
    }

    #[test]
    fn test_register_spill() {
        let mut longevity = LifetimeManager::new();
        let mut lt0 = Lifetime::new(0, 10);
        lt0.real_usages = Some(vec![5]);
        let mut lt1 = Lifetime::new(1, 10);
        lt1.real_usages = Some(vec![6]);
        let mut lt2 = Lifetime::new(2, 10);
        lt2.real_usages = Some(vec![7]);
        longevity.set(OpRef(0), lt0);
        longevity.set(OpRef(1), lt1);
        longevity.set(OpRef(2), lt2);

        // Only 2 registers available
        let mut rm = RegisterManager::new(vec![EAX, ECX], vec![], vec![EAX, ECX], EBP, None, EAX);
        rm.position = 3;
        let mut fm = FrameManager::new(0);

        let reg0 = rm.force_allocate_reg(OpRef(0), &[], None, false, &mut longevity, &mut fm);
        let reg1 = rm.force_allocate_reg(OpRef(1), &[], None, false, &mut longevity, &mut fm);
        assert!(reg0 != reg1);

        // Third allocation forces a spill
        let reg2 = rm.force_allocate_reg(OpRef(2), &[], None, false, &mut longevity, &mut fm);
        assert!(reg2 == reg0 || reg2 == reg1);
        // The spilled variable should now be in the frame
        assert!(fm.get_frame_depth() >= 1);
    }

    #[test]
    fn test_get_scale() {
        assert_eq!(get_scale(1), 0);
        assert_eq!(get_scale(2), 1);
        assert_eq!(get_scale(4), 2);
        assert_eq!(get_scale(8), 3);
    }
}
