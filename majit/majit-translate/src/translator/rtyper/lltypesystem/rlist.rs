//! RPython `rpython/rtyper/lltypesystem/rlist.py` parity module.
//!
//! The list repr slice lives in [`crate::translator::rtyper::rlist`]
//! together with the abstract `rpython/rtyper/rlist.py` pieces.  This
//! module keeps the upstream lltypesystem import path available.

use crate::translator::rtyper::error::TyperError;

pub use crate::translator::rtyper::lltypesystem::lltype;
pub use crate::translator::rtyper::rlist::{AbstractListIteratorRepr, FixedSizeListRepr, ListRepr};
pub use crate::translator::rtyper::rmodel::Repr;

pub const INITIAL_EMPTY_LIST_ALLOCATION: i64 = 0;

/// RPython `class BaseListRepr(AbstractBaseListRepr)`
/// (`lltypesystem/rlist.py:52`).
///
/// The concrete list repr fields live in [`ListRepr`] and
/// [`FixedSizeListRepr`]. This placeholder keeps the lltypesystem-specific
/// inheritance surface visible until the full BaseListRepr split lands.
#[derive(Debug, Default)]
pub struct BaseListRepr;

/// RPython `class ListIteratorRepr(AbstractListIteratorRepr)`
/// (`lltypesystem/rlist.py:453`).
#[derive(Debug, Default)]
pub struct ListIteratorRepr;

/// RPython `class ReversedListIteratorRepr(AbstractListIteratorRepr)`
/// (`lltypesystem/rlist.py:497`).
#[derive(Debug, Default)]
pub struct ReversedListIteratorRepr;

fn lltypesystem_rlist_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!("lltypesystem.rlist.{name} — list helper deferred"))
}

#[allow(non_snake_case)]
pub fn LIST_OF() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("LIST_OF"))
}

pub fn _ll_list_resize_hint_really() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize_hint_really"))
}

pub fn _ll_list_resize_hint() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize_hint"))
}

pub fn _ll_list_resize_really() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize_really"))
}

pub fn _ll_list_resize() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize"))
}

pub fn _ll_list_resize_ge() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize_ge"))
}

pub fn _ll_list_resize_le() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize_le"))
}

pub fn ll_append_noresize() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_append_noresize"))
}

pub fn ll_both_none() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_both_none"))
}

pub fn ll_newlist() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_newlist"))
}

pub fn ll_newlist_hint() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_newlist_hint"))
}

pub fn _ll_prebuilt_empty_array() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_prebuilt_empty_array"))
}

pub fn _ll_new_empty_item_array() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_new_empty_item_array"))
}

pub fn ll_newemptylist() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_newemptylist"))
}

pub fn ll_length() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_length"))
}

pub fn ll_items() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_items"))
}

pub fn ll_getitem_fast() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_getitem_fast"))
}

pub fn ll_setitem_fast() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_setitem_fast"))
}

pub fn ll_fixed_newlist() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_newlist"))
}

pub fn ll_fixed_newemptylist() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_newemptylist"))
}

pub fn ll_fixed_length() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_length"))
}

pub fn ll_fixed_items() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_items"))
}

pub fn ll_fixed_getitem_fast() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_getitem_fast"))
}

pub fn ll_fixed_setitem_fast() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_setitem_fast"))
}

pub fn newlist() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("newlist"))
}

pub fn ll_set_maxlength() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_set_maxlength"))
}

pub fn ll_list2fixed() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_list2fixed"))
}

pub fn ll_list2fixed_exact() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_list2fixed_exact"))
}

pub fn ll_listiter() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_listiter"))
}

pub fn ll_listnext() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_listnext"))
}

pub fn ll_listnext_foldable() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_listnext_foldable"))
}

pub fn ll_getnextindex() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_getnextindex"))
}

pub fn ll_revlistiter() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_revlistiter"))
}

pub fn ll_revlistnext() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_revlistnext"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lltypesystem_rlist_exposes_concrete_list_repr_paths() {
        fn assert_repr<T: Repr>() {}

        assert_repr::<FixedSizeListRepr>();
        assert_repr::<ListRepr>();

        assert_eq!(
            std::any::type_name::<FixedSizeListRepr>()
                .rsplit("::")
                .next(),
            Some("FixedSizeListRepr")
        );
        assert_eq!(
            std::any::type_name::<ListRepr>().rsplit("::").next(),
            Some("ListRepr")
        );
    }

    #[test]
    fn lltypesystem_rlist_exposes_deferred_helper_surface() {
        let _base = BaseListRepr;
        let _iter = ListIteratorRepr;
        let _rev_iter = ReversedListIteratorRepr;
        let _abstract_iter = AbstractListIteratorRepr;
        assert_eq!(INITIAL_EMPTY_LIST_ALLOCATION, 0);

        let err = ll_newlist().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_newlist"));

        let err = ll_revlistnext().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_revlistnext"));
    }
}
