/// Canonical hint kinds understood by the framework.
///
/// RPython mirrors a single `hint(x, **kwds)` operator (`rlib/jit.py:81`,
/// `flowspace/operation.py:521 add_operator('hint', None, dispatch=1)`)
/// whose kwarg dict picks the behaviour.  Pyre's helper layer cannot
/// carry kwarg dicts on a `Call` shape, so each kwarg-key gets its own
/// dispatch-by-name helper (`hint_access_directly`, `hint_promote`, …)
/// and the variant tag below mirrors the RPython kwarg key.
///
/// RPython equivalents (`rlib/jit.py:81-98`):
///   * `hint(x, access_directly=True)`     → [`AccessDirectly`]
///   * `hint(x, fresh_virtualizable=True)` → [`FreshVirtualizable`]
///   * `hint(x, force_virtualizable=True)` → [`ForceVirtualizable`]
///   * `hint(x, promote=True)`             → [`Promote`]
///     (also reached via `rlib/jit.py:101 promote(x) → hint(x,
///     promote=True)`, the user-facing wrapper).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HintKind {
    AccessDirectly,
    FreshVirtualizable,
    ForceVirtualizable,
    /// `rlib/jit.py:101 promote(x)` / `hint(x, promote=True)`.  Rewrite
    /// emits `[-live-, <kind>_guard_value(x), Identity(x)]` per
    /// `jit_codewriter/jtransform.py:608-614`; the `<kind>` char is
    /// resolved from the arg's value-kind at rewrite time.
    Promote,
    /// `rlib/jit.py:127 promote_string(x)` / `hint(x,
    /// promote_string=True)`.  Upstream emits the 3-input
    /// `str_guard_value/rid>r` op (`jit_codewriter/jtransform.py:
    /// 615-631`) calling `_ll_2_str_eq_nonnull`
    /// (`rpython/jit/codewriter/support.py:526-538`) on a
    /// `Ptr(rstr.STR)` arg.  Pyre's `jtransform::rewrite_op_hint`
    /// panics in this arm because pyre-object has no `rstr.STR`-
    /// equivalent GC layout (`rpython/rtyper/lltypesystem/rstr.py:
    /// 1226-1237 STR.become({hash, chars: Array(Char)})`).
    PromoteString,
    /// `rlib/jit.py:130 promote_unicode(x)` / `hint(x,
    /// promote_unicode=True)`.  Same upstream shape as
    /// `PromoteString` but on `Ptr(rstr.UNICODE)` arg
    /// (`jit_codewriter/jtransform.py:632-648`).  Pyre panics for
    /// the same reason: no `rstr.UNICODE`-equivalent GC layout
    /// (`rpython/rtyper/lltypesystem/rstr.py:1238-1246
    /// UNICODE.become({hash, chars: Array(UniChar)})`).
    PromoteUnicode,
    /// `rlib/jit.py:191-194` — `hint(arg, promote=True,
    /// promote_string=True)` carries both flags so jtransform's
    /// `jit_codewriter/jtransform.py:599-606` disambiguator can pick
    /// the right rewrite based on the arg's `concretetype`.  Pyre's
    /// per-kwarg dispatch surface lacks a way to declare "both
    /// kwargs at once", so the `elidable_promote` wrapper synthesiser
    /// emits the combined hint through a dedicated
    /// `hint_promote_or_string` helper; this variant carries the
    /// dual-hint shape into `rewrite_op_hint`.  Per
    /// `jit_codewriter/jtransform.py:601-606` the disambiguator picks
    /// `PromoteString` when `op.args[0].concretetype ==
    /// lltype.Ptr(rstr.STR)` and `Promote` otherwise.  Pyre has no
    /// `Ptr(rstr.STR)` GC layout (`rpython/rtyper/lltypesystem/
    /// rstr.py:1226-1237 STR.become(...)`), so the `if` branch is
    /// structurally unreachable and every dual-flag hint falls
    /// through to plain `Promote`.
    PromoteOrString,
}

/// Classify a function-like symbol as a hint kind.
pub fn classify_hint_segments<'a, I>(segments: I) -> Option<HintKind>
where
    I: IntoIterator<Item = &'a str>,
{
    match segments.into_iter().last().unwrap_or_default() {
        "hint_access_directly" => Some(HintKind::AccessDirectly),
        "hint_fresh_virtualizable" => Some(HintKind::FreshVirtualizable),
        "hint_force_virtualizable" => Some(HintKind::ForceVirtualizable),
        "hint_promote" => Some(HintKind::Promote),
        "hint_promote_string" => Some(HintKind::PromoteString),
        "hint_promote_unicode" => Some(HintKind::PromoteUnicode),
        "hint_promote_or_string" => Some(HintKind::PromoteOrString),
        _ => None,
    }
}

pub fn classify_hint_path(path: &str) -> Option<HintKind> {
    classify_hint_segments(path.split("::"))
}
