//! `jump.py` — parallel assignment between location sets.
//!
//! Moving a set of values into a set of destinations is not a sequence of
//! independent moves: a destination may still be some other move's source, so
//! an order has to be found, and when the dependency graph has a cycle no
//! order exists and one value must be parked. `remap_frame_layout` is that
//! algorithm (`jump.py:4-64`), and `remap_frame_layout_mixed` (`jump.py:67-97`)
//! is the two-register-class variant used when integer and float arguments are
//! remapped together.
//!
//! ⚠ `dst_locations` and `src_locations` are swapped relative to upstream,
//! which spells the call `remap_frame_layout(assembler, dst_locations,
//! src_locations, tmpreg)`. Both are `&[Loc]`, so nothing catches a call
//! written from upstream's order — read the signature, not the memory of the
//! Python one.
//!
//! The algorithm is identical for every backend; only the three primitive
//! emitters it drives are not, which is why they are the trait and this is a
//! free function over it.

use crate::arch::WORD;
use crate::regloc::Loc;
use indexmap::IndexMap;

/// The three emitters `remap_frame_layout` drives, named as `jump.py` calls
/// them on the `assembler` it is handed.
///
/// `regalloc_push`/`regalloc_pop` exist only to serve the cycle-breaking arm
/// below — parking one value of a cycle on the machine stack is the whole
/// reason a cycle can be resolved at all.
pub(crate) trait RegallocMoves {
    /// `assembler.py:1145 regalloc_mov(from_loc, to_loc)`.
    fn regalloc_mov(&mut self, src: &Loc, dst: &Loc);
    fn regalloc_push(&mut self, loc: &Loc);
    fn regalloc_pop(&mut self, loc: &Loc);
}

/// A location's identity for the dependency bookkeeping.
///
/// Registers and frame slots share one number space, so the constants keep the
/// classes apart; an immediate has no identity because it is never anyone's
/// destination and can be re-materialised at will.
pub(crate) fn loc_as_key(loc: &Loc) -> i32 {
    match loc {
        Loc::Reg(r) if r.is_xmm => 0x2000 + i32::from(r.value),
        Loc::Reg(r) => 0x1000 + i32::from(r.value),
        Loc::Frame(f) => f.ebp_loc.value,
        Loc::Ebp(e) => e.value,
        Loc::Immed(_) => i32::MIN,
        Loc::Addr(a) => a.offset,
    }
}

pub(crate) fn loc_width(loc: &Loc) -> usize {
    match loc {
        Loc::Reg(r) => r.get_width(),
        Loc::Frame(f) => f.ebp_loc.get_width(),
        Loc::Ebp(e) => e.get_width(),
        _ => WORD,
    }
}

/// `jump.py:4 remap_frame_layout` — emit the moves that put `src_locations`
/// into `dst_locations`, in an order no move invalidates.
///
/// `tmpreg` is needed for a stack-to-stack pair, which no machine here can
/// move in one instruction.
pub(crate) fn remap_frame_layout<A: RegallocMoves + ?Sized>(
    asm: &mut A,
    src_locations: &[Loc],
    dst_locations: &[Loc],
    tmpreg: Loc,
) {
    let mut pending_dests = dst_locations.len() as i32;
    let mut srccount: IndexMap<i32, i32> = IndexMap::new();
    for dst in dst_locations {
        srccount.insert(loc_as_key(dst), 0);
    }
    for i in 0..dst_locations.len() {
        let src = src_locations[i];
        if src.is_immed() {
            continue;
        }
        let key = loc_as_key(&src);
        if let Some(cnt) = srccount.get_mut(&key) {
            if key == loc_as_key(&dst_locations[i]) {
                *cnt = -(dst_locations.len() as i32) - 1;
                pending_dests -= 1;
            } else {
                *cnt += 1;
            }
        }
    }

    while pending_dests > 0 {
        let mut progress = false;
        for i in 0..dst_locations.len() {
            let dst = dst_locations[i];
            let key = loc_as_key(&dst);
            if srccount.get(&key).copied().unwrap_or(-1) == 0 {
                srccount.insert(key, -1);
                pending_dests -= 1;
                let src = src_locations[i];
                if !src.is_immed() {
                    let src_key = loc_as_key(&src);
                    if let Some(cnt) = srccount.get_mut(&src_key) {
                        *cnt -= 1;
                    }
                }
                if dst.is_stack() && src.is_stack() {
                    asm.regalloc_mov(&src, &tmpreg);
                    asm.regalloc_mov(&tmpreg, &dst);
                } else {
                    asm.regalloc_mov(&src, &dst);
                }
                progress = true;
            }
        }
        if !progress {
            let mut sources: IndexMap<i32, Loc> = IndexMap::new();
            for i in 0..dst_locations.len() {
                sources.insert(loc_as_key(&dst_locations[i]), src_locations[i]);
            }
            for dst in dst_locations {
                let originalkey = loc_as_key(dst);
                if srccount.get(&originalkey).copied().unwrap_or(-1) >= 0 {
                    asm.regalloc_push(dst);
                    let mut cur_dst = *dst;
                    loop {
                        let key = loc_as_key(&cur_dst);
                        srccount.insert(key, -1);
                        pending_dests -= 1;
                        let src = sources[&key];
                        if loc_as_key(&src) == originalkey {
                            break;
                        }
                        if cur_dst.is_stack() && src.is_stack() {
                            asm.regalloc_mov(&src, &tmpreg);
                            asm.regalloc_mov(&tmpreg, &cur_dst);
                        } else {
                            asm.regalloc_mov(&src, &cur_dst);
                        }
                        cur_dst = src;
                    }
                    asm.regalloc_pop(&cur_dst);
                }
            }
        }
    }
}

/// `jump.py:67 remap_frame_layout_mixed` — two location sets remapped with a
/// temporary each, as integer and float arguments need.
///
/// The sets are not independent: a set-2 stack source may be a set-1
/// destination, and set 1 runs first. Those sources are pushed before either
/// remap and popped into place after, which is why they are dropped from the
/// set-2 lists rather than reordered.
pub(crate) fn remap_frame_layout_mixed<A: RegallocMoves + ?Sized>(
    asm: &mut A,
    src_locations1: &[Loc],
    dst_locations1: &[Loc],
    tmpreg1: Loc,
    src_locations2: &[Loc],
    dst_locations2: &[Loc],
    tmpreg2: Loc,
) {
    let mut extrapushes = Vec::new();
    let mut dst_keys = IndexMap::new();
    for loc in dst_locations1 {
        dst_keys.insert(loc_as_key(loc), ());
    }
    let mut src_locations2red = Vec::new();
    let mut dst_locations2red = Vec::new();
    for i in 0..src_locations2.len() {
        let loc = src_locations2[i];
        let dstloc = dst_locations2[i];
        if loc.is_stack() {
            let key = loc_as_key(&loc);
            if dst_keys.contains_key(&key)
                || (loc_width(&loc) > WORD && dst_keys.contains_key(&(key + WORD as i32)))
            {
                asm.regalloc_push(&loc);
                extrapushes.push(dstloc);
                continue;
            }
        }
        src_locations2red.push(loc);
        dst_locations2red.push(dstloc);
    }
    remap_frame_layout(asm, src_locations1, dst_locations1, tmpreg1);
    remap_frame_layout(asm, &src_locations2red, &dst_locations2red, tmpreg2);
    while let Some(loc) = extrapushes.pop() {
        asm.regalloc_pop(&loc);
    }
}
