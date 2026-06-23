//! RPython `rpython/rtyper/lltypesystem/llgroup.py`.
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::annotator::model::{KnownType, SomeInteger, SomeValue};
use crate::translator::rtyper::lltypesystem::llmemory::AddressOffset;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, _ptr_obj, GcKind, LowLevelType, PtrTarget, parentlink, typeOf,
};

thread_local! {
    /// RPython `_membership = weakref.WeakValueDictionary()` keyed by the
    /// struct container object. The translator keeps low-level containers for
    /// the process lifetime, so a strong map keyed by `_struct.identity()` is
    /// the equivalent Rust carrier.
    static MEMBERSHIP: RefCell<HashMap<usize, group>> = RefCell::new(HashMap::new());
}

/// RPython `GroupType(lltype.ContainerType)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupType;

/// RPython `Group = GroupType()`.
pub const Group: GroupType = GroupType;

/// RPython `LONG_BIT == 32 ? 16 : 32`.
#[cfg(target_pointer_width = "32")]
pub const HALFSHIFT: u32 = 16;

/// RPython `LONG_BIT == 32 ? 16 : 32`.
#[cfg(not(target_pointer_width = "32"))]
pub const HALFSHIFT: u32 = 32;

/// RPython `HALFWORD = rffi.USHORT` on 32-bit and `rffi.UINT` on 64-bit.
///
/// Pyre's lltype surface does not currently distinguish `USHORT` from
/// `UINT`; both flow as an unsigned primitive until the rffi scalar family is
/// split into width-specific low-level types.
pub fn HALFWORD() -> LowLevelType {
    LowLevelType::Unsigned
}

fn fresh_group_identity() -> usize {
    static NEXT_GROUP_ID: AtomicUsize = AtomicUsize::new(1);
    NEXT_GROUP_ID.fetch_add(1, Ordering::Relaxed)
}

fn fresh_symbolic_identity() -> usize {
    static NEXT_SYMBOLIC_ID: AtomicUsize = AtomicUsize::new(1);
    NEXT_SYMBOLIC_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
struct GroupCore {
    identity: usize,
    name: String,
    members: Mutex<Vec<_ptr>>,
    outdated: Mutex<Option<String>>,
}

/// RPython `class group(lltype._container)`.
#[derive(Clone, Debug)]
pub struct group(Arc<GroupCore>);

impl group {
    pub fn new(name: impl Into<String>) -> Self {
        group(Arc::new(GroupCore {
            identity: fresh_group_identity(),
            name: name.into(),
            members: Mutex::new(Vec::new()),
            outdated: Mutex::new(None),
        }))
    }

    pub fn _TYPE(&self) -> GroupType {
        Group
    }

    pub fn identity(&self) -> usize {
        self.0.identity
    }

    pub fn name(&self) -> String {
        self.0.name.clone()
    }

    pub fn outdated(&self) -> Option<String> {
        self.0.outdated.lock().unwrap().clone()
    }

    pub fn members(&self) -> Vec<_ptr> {
        self.0.members.lock().unwrap().clone()
    }

    /// RPython `group.add_member(self, structptr)`.
    pub fn add_member(&self, structptr: &_ptr) -> Result<GroupMemberOffset, String> {
        let TYPE = typeOf(structptr);
        let PtrTarget::Struct(struct_t) = &TYPE.TO else {
            return Err("group.add_member: expected pointer to Struct".to_string());
        };
        if struct_t._gckind != GcKind::Raw {
            return Err("group.add_member: expected raw Struct".to_string());
        }
        let struct_obj = match structptr
            ._obj()
            .map_err(|_| "delayed pointer".to_string())?
        {
            _ptr_obj::Struct(s) => s,
            other => {
                return Err(format!(
                    "group.add_member: expected struct object, got {other:?}"
                ));
            }
        };

        if let Some(prevgroup) =
            MEMBERSHIP.with(|m| m.borrow().get(&struct_obj.identity()).cloned())
        {
            *prevgroup.0.outdated.lock().unwrap() = Some(format!(
                "structure {:?} was inserted into another group",
                struct_obj
            ));
        }

        let (parent, _) = parentlink(&_ptr_obj::Struct(struct_obj.clone()));
        if parent.is_some() {
            return Err("group.add_member: expected a top-level structure".to_string());
        }

        let index = {
            let mut members = self.0.members.lock().unwrap();
            let index = members.len();
            members.push(structptr.clone());
            index
        };
        MEMBERSHIP.with(|m| {
            m.borrow_mut().insert(struct_obj.identity(), self.clone());
        });
        Ok(GroupMemberOffset::new(self, index))
    }
}

impl PartialEq for group {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for group {}

/// RPython `member_of_group(structptr)`.
pub fn member_of_group(structptr: &_ptr) -> Option<group> {
    let Ok(_ptr_obj::Struct(s)) = structptr._obj() else {
        return None;
    };
    MEMBERSHIP.with(|m| m.borrow().get(&s.identity()).cloned())
}

/// RPython `class GroupMemberOffset(llmemory.Symbolic)`.
#[derive(Clone, Debug)]
pub struct GroupMemberOffset {
    identity: usize,
    pub grpptr: group,
    pub index: usize,
    pub member: _ptr,
}

impl GroupMemberOffset {
    pub fn new(grp: &group, memberindex: usize) -> Self {
        let members = grp.0.members.lock().unwrap();
        let member = members
            .get(memberindex)
            .unwrap_or_else(|| panic!("group member index out of range: {memberindex}"))
            .clone();
        GroupMemberOffset {
            identity: fresh_symbolic_identity(),
            grpptr: grp.clone(),
            index: memberindex,
            member,
        }
    }

    pub fn annotation(&self) -> SomeValue {
        SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::Ruint))
    }

    pub fn lltype(&self) -> LowLevelType {
        HALFWORD()
    }

    pub fn nonzero(&self) -> bool {
        true
    }

    pub fn identity(&self) -> usize {
        self.identity
    }

    pub fn _get_group_member(&self, grpptr: &group) -> Result<_ptr, String> {
        if grpptr != &self.grpptr {
            return Err("get_group_member: wrong group!".to_string());
        }
        Ok(self.member.clone())
    }

    pub fn _get_next_group_member(
        &self,
        grpptr: &group,
        skipoffset: &AddressOffset,
    ) -> Result<_ptr, String> {
        if grpptr != &self.grpptr {
            return Err("get_next_group_member: wrong group!".to_string());
        }
        let AddressOffset::ItemOffset { TYPE, repeat } = skipoffset else {
            return Err("get_next_group_member: expected ItemOffset".to_string());
        };
        let member_type = typeOf(&self.member).TO.into();
        if TYPE != &member_type || *repeat != 1 {
            return Err("get_next_group_member: wrong skipoffset".to_string());
        }
        let members = self.grpptr.0.members.lock().unwrap();
        members
            .get(self.index + 1)
            .cloned()
            .ok_or_else(|| "get_next_group_member: no following member".to_string())
    }
}

impl PartialEq for GroupMemberOffset {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Eq for GroupMemberOffset {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CombinedLowPart {
    Int(u64),
    GroupMemberOffset(GroupMemberOffset),
}

impl CombinedLowPart {
    fn identity(&self) -> usize {
        match self {
            CombinedLowPart::Int(value) => *value as usize,
            CombinedLowPart::GroupMemberOffset(offset) => offset.identity(),
        }
    }
}

impl From<u64> for CombinedLowPart {
    fn from(value: u64) -> Self {
        CombinedLowPart::Int(value)
    }
}

impl From<GroupMemberOffset> for CombinedLowPart {
    fn from(value: GroupMemberOffset) -> Self {
        CombinedLowPart::GroupMemberOffset(value)
    }
}

/// RPython `class CombinedSymbolic(llmemory.Symbolic)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CombinedSymbolic {
    pub lowpart: CombinedLowPart,
    pub rest: i64,
}

impl CombinedSymbolic {
    pub const MASK: i64 = (1_i64 << HALFSHIFT) - 1;

    pub fn new(lowpart: impl Into<CombinedLowPart>, rest: i64) -> Self {
        assert_eq!(rest & Self::MASK, 0);
        CombinedSymbolic {
            lowpart: lowpart.into(),
            rest,
        }
    }

    pub fn annotation(&self) -> SomeValue {
        SomeValue::Integer(SomeInteger::default())
    }

    pub fn lltype(&self) -> LowLevelType {
        LowLevelType::Signed
    }

    pub fn nonzero(&self) -> bool {
        true
    }

    pub fn bitand(&self, other: i64) -> Result<CombinedAndResult, String> {
        if (other & Self::MASK) == 0 {
            return Ok(CombinedAndResult::Int(self.rest & other));
        }
        if (other & Self::MASK) == Self::MASK {
            return Ok(CombinedAndResult::Combined(CombinedSymbolic::new(
                self.lowpart.clone(),
                self.rest & other,
            )));
        }
        Err(format!("other=0x{other:x}"))
    }

    pub fn bitor(&self, other: i64) -> Self {
        assert_eq!(other & Self::MASK, 0);
        CombinedSymbolic::new(self.lowpart.clone(), self.rest | other)
    }

    pub fn add(&self, other: i64) -> Self {
        assert_eq!(other & Self::MASK, 0);
        CombinedSymbolic::new(self.lowpart.clone(), self.rest + other)
    }

    pub fn sub(&self, other: i64) -> Self {
        assert_eq!(other & Self::MASK, 0);
        CombinedSymbolic::new(self.lowpart.clone(), self.rest - other)
    }

    pub fn rshift(&self, other: u32) -> i64 {
        assert!(other >= HALFSHIFT);
        self.rest >> other
    }

    pub fn eq_same_lowpart(&self, other: &CombinedSymbolic) -> Option<bool> {
        if self.lowpart.identity() == other.lowpart.identity() {
            Some(self.rest == other.rest)
        } else {
            None
        }
    }

    pub fn ne_same_lowpart(&self, other: &CombinedSymbolic) -> Option<bool> {
        self.eq_same_lowpart(other).map(|same| !same)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CombinedAndResult {
    Int(i64),
    Combined(CombinedSymbolic),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::lltypesystem::lltype::{
        LowLevelType, MallocFlavor, StructType, malloc,
    };

    fn raw_struct_ptr(name: &str) -> _ptr {
        malloc(
            LowLevelType::Struct(Box::new(StructType::new(
                name,
                vec![("x".into(), LowLevelType::Signed)],
            ))),
            None,
            MallocFlavor::Raw,
            false,
        )
        .unwrap()
    }

    #[test]
    fn add_member_records_membership_and_offsets() {
        let grp = group::new("test");
        let first = raw_struct_ptr("A");
        let second = raw_struct_ptr("B");

        let first_offset = grp.add_member(&first).unwrap();
        let second_offset = grp.add_member(&second).unwrap();

        assert_eq!(first_offset.index, 0);
        assert_eq!(second_offset.index, 1);
        assert_eq!(member_of_group(&first), Some(grp.clone()));
        assert_eq!(first_offset._get_group_member(&grp).unwrap(), first);

        let skip = AddressOffset::ItemOffset {
            TYPE: typeOf(&first).TO.into(),
            repeat: 1,
        };
        assert_eq!(
            first_offset._get_next_group_member(&grp, &skip).unwrap(),
            second
        );
    }

    #[test]
    fn add_member_rejects_gc_structs() {
        let grp = group::new("test");
        let gc_ptr = malloc(
            LowLevelType::Struct(Box::new(StructType::gc(
                "G",
                vec![("x".into(), LowLevelType::Signed)],
            ))),
            None,
            MallocFlavor::Gc,
            false,
        )
        .unwrap();

        let err = grp.add_member(&gc_ptr).unwrap_err();
        assert!(err.contains("raw Struct"));
    }

    #[test]
    fn reinserting_member_marks_previous_group_outdated() {
        let first_group = group::new("first");
        let second_group = group::new("second");
        let ptr = raw_struct_ptr("A");

        first_group.add_member(&ptr).unwrap();
        second_group.add_member(&ptr).unwrap();

        assert!(
            first_group
                .outdated()
                .unwrap()
                .contains("was inserted into another group")
        );
        assert_eq!(member_of_group(&ptr), Some(second_group));
    }

    #[test]
    fn combined_symbolic_matches_upstream_operations() {
        let grp = group::new("test");
        let offset = grp.add_member(&raw_struct_ptr("A")).unwrap();
        let symbolic = CombinedSymbolic::new(offset.clone(), 1_i64 << HALFSHIFT);

        assert_eq!(
            symbolic.bitand(!CombinedSymbolic::MASK).unwrap(),
            CombinedAndResult::Int(1_i64 << HALFSHIFT)
        );
        assert_eq!(
            symbolic.bitand(CombinedSymbolic::MASK).unwrap(),
            CombinedAndResult::Combined(CombinedSymbolic::new(offset.clone(), 0))
        );
        assert_eq!(symbolic.bitor(2_i64 << HALFSHIFT).rest, 3_i64 << HALFSHIFT);
        assert_eq!(symbolic.add(2_i64 << HALFSHIFT).rest, 3_i64 << HALFSHIFT);
        assert_eq!(symbolic.sub(1_i64 << HALFSHIFT).rest, 0);
        assert_eq!(symbolic.rshift(HALFSHIFT), 1);

        let same_lowpart = CombinedSymbolic::new(offset, 1_i64 << HALFSHIFT);
        let other_lowpart = CombinedSymbolic::new(0_u64, 1_i64 << HALFSHIFT);
        assert_eq!(symbolic.eq_same_lowpart(&same_lowpart), Some(true));
        assert_eq!(symbolic.eq_same_lowpart(&other_lowpart), None);
    }
}
