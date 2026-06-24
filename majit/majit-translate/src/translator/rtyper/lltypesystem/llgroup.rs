//! Port of `rpython/rtyper/lltypesystem/llgroup.py`.
//!
//! RPython uses groups to pack raw static structs and refer to members through
//! compact half-word offsets. Pyre does not use the C backend group layout at
//! runtime, but the symbolic carriers are part of the lltypesystem surface and
//! are referenced by `opimpl.py` parity.

#![allow(non_camel_case_types, non_upper_case_globals)]

use std::fmt;
use std::ops::{Add, BitAnd, BitOr, Shr, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// RPython `GroupType(lltype.ContainerType)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupType {
    pub _gckind: &'static str,
}

/// RPython `Group = GroupType()`.
pub const Group: GroupType = GroupType { _gckind: "raw" };

#[cfg(target_pointer_width = "32")]
pub const HALFSHIFT: u32 = 16;
#[cfg(not(target_pointer_width = "32"))]
pub const HALFSHIFT: u32 = 32;

#[cfg(target_pointer_width = "32")]
pub type HALFWORD = u16;
#[cfg(not(target_pointer_width = "32"))]
pub type HALFWORD = u32;

#[cfg(target_pointer_width = "32")]
pub type r_halfword = u16;
#[cfg(not(target_pointer_width = "32"))]
pub type r_halfword = u32;

static NEXT_GROUP_ID: AtomicUsize = AtomicUsize::new(1);
static NEXT_MEMBER_ID: AtomicUsize = AtomicUsize::new(1);
static MEMBERSHIP: OnceLock<Mutex<majit_ir::VecMap<usize, GroupPtr>>> = OnceLock::new();
static OUTDATED: OnceLock<Mutex<majit_ir::VecMap<usize, String>>> = OnceLock::new();

fn membership() -> &'static Mutex<majit_ir::VecMap<usize, GroupPtr>> {
    MEMBERSHIP.get_or_init(|| Mutex::new(majit_ir::VecMap::new()))
}

fn outdated() -> &'static Mutex<majit_ir::VecMap<usize, String>> {
    OUTDATED.get_or_init(|| Mutex::new(majit_ir::VecMap::new()))
}

/// Stand-in for a raw struct pointer inserted into an llgroup.
///
/// RPython stores the actual lltype struct object. At this translator layer the
/// object identity is all the group operations need, so the carrier keeps a
/// stable id and debug name.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct GroupMember {
    pub id: usize,
    pub name: String,
    pub parent_structure: Option<String>,
}

impl GroupMember {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: NEXT_MEMBER_ID.fetch_add(1, Ordering::Relaxed),
            name: name.into(),
            parent_structure: None,
        }
    }

    pub fn with_parent_structure(
        name: impl Into<String>,
        parent_structure: impl Into<String>,
    ) -> Self {
        Self {
            id: NEXT_MEMBER_ID.fetch_add(1, Ordering::Relaxed),
            name: name.into(),
            parent_structure: Some(parent_structure.into()),
        }
    }
}

/// RPython `class group(lltype._container)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct group {
    pub name: String,
    pub members: Vec<GroupMember>,
    pub outdated: Option<String>,
    id: usize,
}

impl group {
    pub const _TYPE: GroupType = Group;

    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            members: Vec::new(),
            outdated: None,
            id: NEXT_GROUP_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn add_member(&mut self, structptr: GroupMember) -> GroupMemberOffset {
        let previous_group = membership()
            .lock()
            .expect("llgroup membership lock poisoned")
            .get(&structptr.id)
            .copied();
        if let Some(previous_group) = previous_group {
            let message = format!("structure {:?} was inserted into another group", structptr);
            outdated()
                .lock()
                .expect("llgroup outdated lock poisoned")
                .insert(previous_group.id, message.clone());
            if let Some(prevgroup) = group_by_ptr_mut(previous_group, self) {
                prevgroup.outdated = Some(message);
            }
        }
        assert!(
            structptr.parent_structure.is_none(),
            "llgroup.py: struct._parentstructure() is not None"
        );
        let index = self.members.len();
        self.members.push(structptr.clone());
        membership()
            .lock()
            .expect("llgroup membership lock poisoned")
            .insert(structptr.id, self._as_ptr());
        GroupMemberOffset::new(self, index, structptr)
    }

    pub fn _as_ptr(&self) -> GroupPtr {
        GroupPtr { id: self.id }
    }

    pub fn current_outdated(&self) -> Option<String> {
        self.outdated.clone().or_else(|| {
            outdated()
                .lock()
                .expect("llgroup outdated lock poisoned")
                .get(&self.id)
                .cloned()
        })
    }
}

fn group_by_ptr_mut<'a>(ptr: GroupPtr, current: &'a mut group) -> Option<&'a mut group> {
    if current._as_ptr() == ptr {
        Some(current)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GroupPtr {
    pub id: usize,
}

/// RPython `member_of_group(structptr)`.
///
/// RPython stores `_membership = weakref.WeakValueDictionary()`. Group members
/// here are stable symbolic carriers, so a `VecMap` keyed by member identity is
/// the direct dict-shaped counterpart without introducing a separate
/// Rust-native collection.
pub fn member_of_group(structptr: &GroupMember) -> Option<GroupPtr> {
    membership()
        .lock()
        .expect("llgroup membership lock poisoned")
        .get(&structptr.id)
        .copied()
}

/// RPython `GroupMemberOffset(llmemory.Symbolic)`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct GroupMemberOffset {
    pub grpptr: GroupPtr,
    pub index: usize,
    pub member: GroupMember,
}

impl GroupMemberOffset {
    pub fn new(grp: &group, memberindex: usize, member: GroupMember) -> Self {
        assert_eq!(grp._as_ptr(), GroupPtr { id: grp.id });
        Self {
            grpptr: grp._as_ptr(),
            index: memberindex,
            member,
        }
    }

    pub fn annotation(&self) -> &'static str {
        "SomeInteger(knowntype=r_halfword)"
    }

    pub fn lltype(&self) -> &'static str {
        "HALFWORD"
    }

    pub fn _get_group_member(&self, grpptr: GroupPtr) -> &GroupMember {
        assert_eq!(grpptr, self.grpptr, "get_group_member: wrong group!");
        &self.member
    }

    pub fn _get_next_group_member<'a>(
        &self,
        grpptr: GroupPtr,
        members: &'a [GroupMember],
    ) -> &'a GroupMember {
        assert_eq!(grpptr, self.grpptr, "get_next_group_member: wrong group!");
        &members[self.index + 1]
    }
}

impl fmt::Display for GroupMemberOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GroupMemberOffset({:?}, {})", self.grpptr, self.index)
    }
}

/// RPython `CombinedSymbolic(llmemory.Symbolic)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CombinedSymbolic<L = GroupMemberOffset> {
    pub lowpart: L,
    pub rest: u64,
}

impl<L> CombinedSymbolic<L> {
    pub const MASK: u64 = (1u64 << HALFSHIFT) - 1;

    pub fn new(lowpart: L, rest: u64) -> Self {
        assert_eq!(rest & Self::MASK, 0);
        Self { lowpart, rest }
    }

    pub fn annotation(&self) -> &'static str {
        "SomeInteger()"
    }

    pub fn lltype(&self) -> &'static str {
        "Signed"
    }
}

impl<L: fmt::Debug> fmt::Display for CombinedSymbolic<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<CombinedSymbolic {:?}|{}>", self.lowpart, self.rest)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CombinedAnd<L = GroupMemberOffset> {
    Rest(u64),
    Combined(CombinedSymbolic<L>),
}

impl<L: Clone> BitAnd<u64> for CombinedSymbolic<L> {
    type Output = CombinedAnd<L>;

    fn bitand(self, other: u64) -> Self::Output {
        if (other & Self::MASK) == 0 {
            return CombinedAnd::Rest(self.rest & other);
        }
        if (other & Self::MASK) == Self::MASK {
            return CombinedAnd::Combined(CombinedSymbolic::new(self.lowpart, self.rest & other));
        }
        panic!("other=0x{other:x}");
    }
}

impl<L> BitOr<u64> for CombinedSymbolic<L> {
    type Output = CombinedSymbolic<L>;

    fn bitor(self, other: u64) -> Self::Output {
        assert_eq!(other & Self::MASK, 0);
        CombinedSymbolic::new(self.lowpart, self.rest | other)
    }
}

impl<L> Add<u64> for CombinedSymbolic<L> {
    type Output = CombinedSymbolic<L>;

    fn add(self, other: u64) -> Self::Output {
        assert_eq!(other & Self::MASK, 0);
        CombinedSymbolic::new(self.lowpart, self.rest + other)
    }
}

impl<L> Sub<u64> for CombinedSymbolic<L> {
    type Output = CombinedSymbolic<L>;

    fn sub(self, other: u64) -> Self::Output {
        assert_eq!(other & Self::MASK, 0);
        CombinedSymbolic::new(self.lowpart, self.rest - other)
    }
}

impl<L> Shr<u32> for CombinedSymbolic<L> {
    type Output = u64;

    fn shr(self, other: u32) -> Self::Output {
        assert!(other >= HALFSHIFT);
        self.rest >> other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_member_offset() {
        let mut grp = group::new("grp");
        let first = GroupMember::new("first");
        let second = GroupMember::new("second");
        let offset = grp.add_member(first.clone());
        grp.add_member(second.clone());
        assert_eq!(member_of_group(&first), Some(grp._as_ptr()));
        assert_eq!(offset._get_group_member(grp._as_ptr()), &first);
        assert_eq!(
            offset._get_next_group_member(grp._as_ptr(), &grp.members),
            &second
        );
    }

    #[test]
    fn add_member_marks_previous_group_outdated() {
        let mut grp1 = group::new("grp1");
        let mut grp2 = group::new("grp2");
        let member = GroupMember::new("moved-member");

        grp1.add_member(member.clone());
        grp2.add_member(member.clone());

        assert_eq!(member_of_group(&member), Some(grp2._as_ptr()));
        assert!(
            grp1.current_outdated()
                .expect("previous group should be marked outdated")
                .contains("inserted into another group")
        );
    }

    #[test]
    fn add_member_rejects_nested_struct_member() {
        let mut grp = group::new("grp");
        let nested = GroupMember::with_parent_structure("nested", "parent");

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            grp.add_member(nested);
        }));

        assert!(result.is_err());
    }

    #[test]
    fn test_combined_symbolic() {
        let lowpart = GroupMemberOffset {
            grpptr: GroupPtr { id: 1 },
            index: 1,
            member: GroupMember::new("member"),
        };
        let symbolic = CombinedSymbolic::new(lowpart.clone(), 0x45u64 << HALFSHIFT);
        assert_eq!(
            symbolic.clone() & CombinedSymbolic::<GroupMemberOffset>::MASK,
            CombinedAnd::Combined(CombinedSymbolic::new(lowpart.clone(), 0))
        );
        assert_eq!(
            symbolic.clone() & !CombinedSymbolic::<GroupMemberOffset>::MASK,
            CombinedAnd::Rest(0x45u64 << HALFSHIFT)
        );
        assert_eq!(
            (symbolic.clone() | (0x01u64 << HALFSHIFT)).rest,
            0x45u64 << HALFSHIFT
        );
        assert_eq!((symbolic >> HALFSHIFT), 0x45);
    }
}
