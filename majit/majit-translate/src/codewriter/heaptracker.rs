//! Struct descriptor helpers from `rpython/jit/codewriter/heaptracker.py`.
//!
//! Most descriptor construction already lives on [`CallControl`], because
//! Pyre's codewriter needs the same cache identity while lowering calls and
//! emitting bytecode.  This module restores the PyPy namespace and symbol
//! names, routing every descriptor operation through those existing caches
//! instead of adding a side table.

use std::collections::HashMap;

use crate::codewriter::call::{CallControl, extract_element_type_from_str, get_type_flag};
use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType, Struct};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedFieldExc(pub String);

#[derive(Debug, Clone, Default)]
pub struct GcStructVTableCache<V> {
    cache_gcstruct2vtable: HashMap<String, V>,
    testing_gcstruct2vtable: HashMap<String, V>,
}

impl<V> GcStructVTableCache<V> {
    pub fn insert_rtyper_vtable(&mut self, gcstruct: &Struct, vtable: V) {
        self.cache_gcstruct2vtable
            .insert(gcstruct._name.clone(), vtable);
    }
}

pub fn is_immutable_struct(s: &Struct) -> bool {
    s._gckind == GcKind::Gc && matches!(s._hints.get("immutable"), Some(ConstValue::Bool(true)))
}

pub fn has_gcstruct_a_vtable(gcstruct: &Struct) -> bool {
    if gcstruct._gckind != GcKind::Gc {
        return false;
    }
    if LowLevelType::Struct(Box::new(gcstruct.clone()))
        == *crate::translator::rtyper::rclass::OBJECT
    {
        return false;
    }

    let mut cursor = gcstruct.clone();
    loop {
        if matches!(cursor._hints.get("typeptr"), Some(ConstValue::Bool(true))) {
            return true;
        }
        let Some((_, first_struct)) = cursor._first_struct_owned() else {
            return false;
        };
        cursor = first_struct;
    }
}

pub fn get_vtable_for_gcstruct<V: Clone>(
    gccache: &mut GcStructVTableCache<V>,
    gcstruct: &Struct,
) -> Option<V> {
    if !has_gcstruct_a_vtable(gcstruct) {
        return None;
    }
    setup_cache_gcstruct2vtable(gccache);
    gccache
        .cache_gcstruct2vtable
        .get(&gcstruct._name)
        .or_else(|| gccache.testing_gcstruct2vtable.get(&gcstruct._name))
        .cloned()
}

pub fn setup_cache_gcstruct2vtable<V>(_gccache: &mut GcStructVTableCache<V>) {}

pub fn set_testing_vtable_for_gcstruct<V>(
    gccache: &mut GcStructVTableCache<V>,
    gcstruct: &Struct,
    vtable: V,
    _name: &str,
) {
    gccache
        .testing_gcstruct2vtable
        .insert(gcstruct._name.clone(), vtable);
}

pub fn all_fielddescrs(
    gccache: &CallControl,
    struct_name: &str,
    only_gc: bool,
) -> Vec<majit_ir::descr::DescrRef> {
    let mut res = Vec::new();
    all_fielddescrs_into(gccache, struct_name, only_gc, &mut res);
    res
}

pub fn all_interiorfielddescrs(
    gccache: &CallControl,
    array_type_id: &str,
) -> Result<Vec<majit_ir::descr::DescrRef>, UnsupportedFieldExc> {
    let elem_name =
        extract_element_type_from_str(array_type_id).unwrap_or_else(|| array_type_id.to_string());
    if let Some(layout) = gccache.struct_layout_for(&elem_name) {
        for field in &layout.fields {
            if field.field_type == majit_ir::value::Type::Void || field.name == "typeptr" {
                continue;
            }
            if field.flag == majit_ir::descr::ArrayFlag::Struct {
                return Err(UnsupportedFieldExc(
                    "unexpected array(struct(struct))".to_string(),
                ));
            }
        }
    } else if let Some(fields) = gccache.struct_field_entries(&elem_name) {
        for (name, field_type) in fields {
            if name == "typeptr" || name.starts_with("c__pad") {
                continue;
            }
            let (_, ir_type, _) = get_type_flag(field_type);
            if ir_type == majit_ir::value::Type::Void {
                continue;
            }
            if gccache.is_known_struct(field_type) {
                return Err(UnsupportedFieldExc(
                    "unexpected array(struct(struct))".to_string(),
                ));
            }
        }
    } else {
        return Ok(Vec::new());
    }

    let mut res = Vec::new();
    let Some(fields) = gccache.struct_field_entries(&elem_name) else {
        if let Some(layout) = gccache.struct_layout_for(&elem_name) {
            for field in &layout.fields {
                if field.field_type == majit_ir::value::Type::Void || field.name == "typeptr" {
                    continue;
                }
                let array_id = Some(array_type_id.to_string());
                let idx = gccache
                    .descr_indices
                    .interiorfield_index(&array_id, &field.name);
                if let Some(descr) = gccache.interiorfielddescrof(idx, &array_id, &field.name) {
                    res.push(descr);
                }
            }
        }
        return Ok(res);
    };

    for (field_name, field_type) in fields {
        if field_name == "typeptr" || field_name.starts_with("c__pad") {
            continue;
        }
        let (_, ir_type, _) = get_type_flag(field_type);
        if ir_type == majit_ir::value::Type::Void {
            continue;
        }
        let array_id = Some(array_type_id.to_string());
        let idx = gccache
            .descr_indices
            .interiorfield_index(&array_id, field_name);
        if let Some(descr) = gccache.interiorfielddescrof(idx, &array_id, field_name) {
            res.push(descr);
        }
    }
    Ok(res)
}

pub fn gc_fielddescrs(gccache: &CallControl, struct_name: &str) -> Vec<majit_ir::descr::DescrRef> {
    all_fielddescrs(gccache, struct_name, true)
}

pub fn get_fielddescr_index_in(
    gccache: &CallControl,
    struct_name: &str,
    fieldname: &str,
    cur_index: isize,
) -> isize {
    let mut cur_index = cur_index;
    let Some(fields) = gccache.struct_field_entries(struct_name) else {
        return -cur_index - 1;
    };
    for (name, field_type) in fields {
        let (_, ir_type, _) = get_type_flag(field_type);
        if ir_type == majit_ir::value::Type::Void {
            continue;
        }
        if name == "typeptr" {
            continue;
        }
        if gccache.is_known_struct(field_type) {
            let r = get_fielddescr_index_in(gccache, field_type, fieldname, cur_index);
            if r >= 0 {
                return r;
            }
            cur_index += -r - 1;
            continue;
        }
        if name == fieldname {
            return cur_index;
        }
        cur_index += 1;
    }
    -cur_index - 1
}

fn all_fielddescrs_into(
    gccache: &CallControl,
    struct_name: &str,
    only_gc: bool,
    res: &mut Vec<majit_ir::descr::DescrRef>,
) {
    let Some(fields) = gccache
        .struct_field_entries(struct_name)
        .map(|f| f.to_vec())
    else {
        return;
    };
    for (name, field_type) in fields {
        let (flag, ir_type, _) = get_type_flag(&field_type);
        if ir_type == majit_ir::value::Type::Void {
            continue;
        }
        if name.starts_with("c__pad") || name == "typeptr" {
            continue;
        }
        if gccache.is_known_struct(&field_type) {
            all_fielddescrs_into(gccache, &field_type, only_gc, res);
        } else if !only_gc || flag == majit_ir::descr::ArrayFlag::Pointer {
            let owner = Some(struct_name.to_string());
            let idx = gccache.descr_indices.field_index(&owner, &name);
            if let Some(descr) = gccache.fielddescrof(idx, struct_name, None, &name) {
                res.push(descr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_hint_helpers_match_heaptracker_predicates() {
        let raw = Struct::with_hints(
            "raw",
            vec![("value".into(), LowLevelType::Signed)],
            vec![("immutable".into(), ConstValue::Bool(true))],
        );
        assert!(!is_immutable_struct(&raw));

        let gc = Struct::gc_with_hints(
            "gc",
            vec![("value".into(), LowLevelType::Signed)],
            vec![("immutable".into(), ConstValue::Bool(true))],
        );
        assert!(is_immutable_struct(&gc));
        assert!(!has_gcstruct_a_vtable(&gc));

        let object_sub = Struct::gc_with_hints(
            "object_sub",
            vec![(
                "super".into(),
                crate::translator::rtyper::rclass::OBJECT.clone(),
            )],
            vec![],
        );
        assert!(has_gcstruct_a_vtable(&object_sub));
    }

    #[test]
    fn vtable_cache_uses_rtyper_then_testing_slot() {
        let gc = Struct::gc_with_hints(
            "instance",
            vec![("typeptr".into(), LowLevelType::Signed)],
            vec![("typeptr".into(), ConstValue::Bool(true))],
        );
        let mut cache = GcStructVTableCache::default();
        set_testing_vtable_for_gcstruct(&mut cache, &gc, "testing", "Instance");
        assert_eq!(get_vtable_for_gcstruct(&mut cache, &gc), Some("testing"));
        cache.insert_rtyper_vtable(&gc, "rtyper");
        assert_eq!(get_vtable_for_gcstruct(&mut cache, &gc), Some("rtyper"));
    }

    #[test]
    fn field_index_recurses_through_nested_structs() {
        let mut cc = CallControl::new();
        cc.set_known_struct_names(["Inner".to_string(), "Outer".to_string()].into());
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "Inner".to_string(),
            vec![
                ("a".to_string(), "i64".to_string()),
                ("b".to_string(), "i64".to_string()),
            ],
        );
        fields.fields.insert(
            "Outer".to_string(),
            vec![
                ("typeptr".to_string(), "usize".to_string()),
                ("inner".to_string(), "Inner".to_string()),
                ("c".to_string(), "i64".to_string()),
            ],
        );
        cc.set_struct_fields(fields);

        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "a", 0), 0);
        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "b", 0), 1);
        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "c", 0), 2);
        assert_eq!(get_fielddescr_index_in(&cc, "Outer", "missing", 0), -4);
    }
}
