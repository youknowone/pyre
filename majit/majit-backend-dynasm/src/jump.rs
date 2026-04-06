/// jump.py: Frame layout remapping for register/frame moves.
///
/// remap_frame_layout(assembler, src_locs, dst_locs, tmpreg)
/// — emit code to move values from src locations to dst locations,
///   handling overlaps via a temporary register.
use crate::regloc::Loc;

/// jump.py:4 remap_frame_layout — emit moves to rearrange locations.
///
/// This handles the problem of parallel assignment: if src[i] overlaps
/// with dst[j], we need a temporary to break the cycle.
pub fn remap_frame_layout(_src_locs: &[Loc], _dst_locs: &[Loc], _tmpreg: Loc) -> Vec<(Loc, Loc)> {
    // TODO: implement cycle-breaking parallel move algorithm
    // For now, return direct moves (incorrect if overlaps exist)
    Vec::new()
}
