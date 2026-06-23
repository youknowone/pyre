//! RPython `rpython/rtyper/lltypesystem/rbuilder.py` parity module.
//!
//! This slice lands the low-level builder container shapes and repr
//! class names. The append/grow/build helper graphs are still pending,
//! but the public lltype names now match upstream:
//! `STRINGPIECE`, `STRINGBUILDER`, `UNICODEPIECE`, `UNICODEBUILDER`,
//! `BaseStringBuilderRepr`, `StringBuilderRepr`, and
//! `UnicodeBuilderRepr`.

use std::sync::LazyLock;

use crate::translator::rtyper::lltypesystem::lltype::{
    ForwardReference, LowLevelType, Ptr, PtrTarget, StructType,
};
use crate::translator::rtyper::lltypesystem::rstr::{STRPTR, UNICODEPTR};

fn ptr_to_lowlevel(target: LowLevelType) -> LowLevelType {
    match target {
        LowLevelType::ForwardReference(fwd) => LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(*fwd),
        })),
        LowLevelType::Struct(t) => LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(*t),
        })),
        other => panic!("expected container lowleveltype for Ptr(...), got {other:?}"),
    }
}

fn ptr_to_forward_reference(target: &LowLevelType) -> LowLevelType {
    let LowLevelType::ForwardReference(fwd) = target.clone() else {
        panic!("builder piece type must be a ForwardReference");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(*fwd),
    }))
}

/// RPython `STRINGPIECE = lltype.GcStruct('stringpiece', ...)`.
pub static STRINGPIECE: LazyLock<LowLevelType> = LazyLock::new(|| {
    let fwd = ForwardReference::gc();
    let body = StructType::gc(
        "stringpiece",
        vec![
            ("buf".into(), STRPTR.clone()),
            (
                "prev_piece".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd.clone()),
                })),
            ),
        ],
    );
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("STRINGPIECE.prev_piece.TO.become(STRINGPIECE)");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(STRINGPIECE)`.
pub static STRINGPIECEPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_forward_reference(&STRINGPIECE));

/// RPython `STRINGBUILDER = lltype.GcStruct('stringbuilder', ...)`.
pub static STRINGBUILDER: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(StructType::gc(
        "stringbuilder",
        vec![
            ("current_buf".into(), STRPTR.clone()),
            ("current_pos".into(), LowLevelType::Signed),
            ("current_end".into(), LowLevelType::Signed),
            ("total_size".into(), LowLevelType::Signed),
            ("extra_pieces".into(), STRINGPIECEPTR.clone()),
        ],
    )))
});

/// RPython `Ptr(STRINGBUILDER)`.
pub static STRINGBUILDERPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_lowlevel(STRINGBUILDER.clone()));

/// RPython `UNICODEPIECE = lltype.GcStruct('unicodepiece', ...)`.
pub static UNICODEPIECE: LazyLock<LowLevelType> = LazyLock::new(|| {
    let fwd = ForwardReference::gc();
    let body = StructType::gc(
        "unicodepiece",
        vec![
            ("buf".into(), UNICODEPTR.clone()),
            (
                "prev_piece".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd.clone()),
                })),
            ),
        ],
    );
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("UNICODEPIECE.prev_piece.TO.become(UNICODEPIECE)");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(UNICODEPIECE)`.
pub static UNICODEPIECEPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_forward_reference(&UNICODEPIECE));

/// RPython `UNICODEBUILDER = lltype.GcStruct('unicodebuilder', ...)`.
pub static UNICODEBUILDER: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(StructType::gc(
        "unicodebuilder",
        vec![
            ("current_buf".into(), UNICODEPTR.clone()),
            ("current_pos".into(), LowLevelType::Signed),
            ("current_end".into(), LowLevelType::Signed),
            ("total_size".into(), LowLevelType::Signed),
            ("extra_pieces".into(), UNICODEPIECEPTR.clone()),
        ],
    )))
});

/// RPython `Ptr(UNICODEBUILDER)`.
pub static UNICODEBUILDERPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_lowlevel(UNICODEBUILDER.clone()));

/// RPython `class BaseStringBuilderRepr(AbstractStringBuilderRepr)`.
#[derive(Debug, Default)]
pub struct BaseStringBuilderRepr;

/// RPython `class StringBuilderRepr(BaseStringBuilderRepr)`.
#[derive(Debug, Default)]
pub struct StringBuilderRepr;

impl StringBuilderRepr {
    /// RPython `StringBuilderRepr.lowleveltype = lltype.Ptr(STRINGBUILDER)`.
    pub fn lowleveltype(&self) -> &'static LowLevelType {
        &STRINGBUILDERPTR
    }

    /// RPython `StringBuilderRepr.basetp = STR`.
    pub fn basetp(&self) -> &'static LowLevelType {
        &crate::translator::rtyper::lltypesystem::rstr::STR
    }
}

/// RPython `class UnicodeBuilderRepr(BaseStringBuilderRepr)`.
#[derive(Debug, Default)]
pub struct UnicodeBuilderRepr;

impl UnicodeBuilderRepr {
    /// RPython `UnicodeBuilderRepr.lowleveltype = lltype.Ptr(UNICODEBUILDER)`.
    pub fn lowleveltype(&self) -> &'static LowLevelType {
        &UNICODEBUILDERPTR
    }

    /// RPython `UnicodeBuilderRepr.basetp = UNICODE`.
    pub fn basetp(&self) -> &'static LowLevelType {
        &crate::translator::rtyper::lltypesystem::rstr::UNICODE
    }
}

static STRINGBUILDER_REPR: LazyLock<StringBuilderRepr> = LazyLock::new(StringBuilderRepr::default);
static UNICODEBUILDER_REPR: LazyLock<UnicodeBuilderRepr> =
    LazyLock::new(UnicodeBuilderRepr::default);

/// RPython `stringbuilder_repr = StringBuilderRepr()`.
pub fn stringbuilder_repr() -> &'static StringBuilderRepr {
    &STRINGBUILDER_REPR
}

/// RPython `unicodebuilder_repr = UnicodeBuilderRepr()`.
pub fn unicodebuilder_repr() -> &'static UnicodeBuilderRepr {
    &UNICODEBUILDER_REPR
}

#[cfg(test)]
mod tests {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};

    #[test]
    fn stringpiece_prev_piece_points_back_to_stringpiece() {
        let LowLevelType::ForwardReference(piece_fwd) = super::STRINGPIECE.clone() else {
            panic!("STRINGPIECE must be a GcForwardReference");
        };
        let Some(LowLevelType::Struct(piece)) = piece_fwd.resolved() else {
            panic!("STRINGPIECE must resolve to a struct");
        };
        let Some(LowLevelType::Ptr(prev_ptr)) = piece.getattr_field_type("prev_piece") else {
            panic!("STRINGPIECE.prev_piece must be Ptr");
        };
        assert!(matches!(prev_ptr.TO, PtrTarget::ForwardReference(_)));
        assert_eq!(
            LowLevelType::ForwardReference(piece_fwd),
            LowLevelType::from(prev_ptr.TO)
        );
    }

    #[test]
    fn stringbuilder_fields_match_rpython_shape() {
        let LowLevelType::Struct(builder) = super::STRINGBUILDER.clone() else {
            panic!("STRINGBUILDER must be GcStruct");
        };
        assert_eq!(
            builder._names,
            vec![
                "current_buf",
                "current_pos",
                "current_end",
                "total_size",
                "extra_pieces",
            ]
        );
        assert_eq!(
            builder.getattr_field_type("current_pos"),
            Some(LowLevelType::Signed)
        );
        assert_eq!(
            builder.getattr_field_type("current_end"),
            Some(LowLevelType::Signed)
        );
        assert_eq!(
            builder.getattr_field_type("total_size"),
            Some(LowLevelType::Signed)
        );
    }

    #[test]
    fn repr_singletons_have_distinct_lowleveltypes() {
        assert_ne!(
            super::stringbuilder_repr().lowleveltype(),
            super::unicodebuilder_repr().lowleveltype()
        );
    }
}
