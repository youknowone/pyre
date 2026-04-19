/// Canonical virtualizable hint kinds understood by the framework.
///
/// RPython equivalents: `hint(..., access_directly=True)`,
/// `hint(..., fresh_virtualizable=True)`, `hint(..., force_virtualizable=True)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualizableHintKind {
    AccessDirectly,
    FreshVirtualizable,
    ForceVirtualizable,
}

/// Classify a function-like symbol as a virtualizable hint.
pub fn classify_virtualizable_hint_segments<'a, I>(segments: I) -> Option<VirtualizableHintKind>
where
    I: IntoIterator<Item = &'a str>,
{
    match segments.into_iter().last().unwrap_or_default() {
        "hint_access_directly" => Some(VirtualizableHintKind::AccessDirectly),
        "hint_fresh_virtualizable" => Some(VirtualizableHintKind::FreshVirtualizable),
        "hint_force_virtualizable" => Some(VirtualizableHintKind::ForceVirtualizable),
        _ => None,
    }
}

pub fn classify_virtualizable_hint_path(path: &str) -> Option<VirtualizableHintKind> {
    classify_virtualizable_hint_segments(path.split("::"))
}
