//! An `rstr.STR` buffer the regex portals carry as a `str` red.
//!
//! Layout is `rstr.STR`: hash word, length word, chars, plus the extra NUL
//! `extra_item_after_alloc` counts in the token basesize. Offsets are the
//! backend constants `inject_builtin_string_descrs` compiles against.
//! `Strlen` / `Strgetitem` loads do not need a GC header.

use majit_backend::{
    BUILTIN_STR_TOKEN_BASE_SIZE, BUILTIN_STRING_CHARS_OFFSET, BUILTIN_STRING_HASH_OFFSET,
    BUILTIN_STRING_LEN_OFFSET,
};

/// Header in front of the character bytes. The extra NUL is past `chars`.
#[repr(C)]
struct Header {
    hash: isize,
    length: usize,
}

/// Owned `rstr.STR` bytes. The pointer stays valid for the life of this value.
pub struct RpyStr {
    buf: Vec<u8>,
}

impl RpyStr {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(
            std::mem::offset_of!(Header, hash),
            BUILTIN_STRING_HASH_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(Header, length),
            BUILTIN_STRING_LEN_OFFSET
        );
        assert_eq!(std::mem::size_of::<Header>(), BUILTIN_STRING_CHARS_OFFSET);
        assert_eq!(
            std::mem::size_of::<Header>() + 1,
            BUILTIN_STR_TOKEN_BASE_SIZE
        );
        let mut buf = vec![0u8; BUILTIN_STRING_CHARS_OFFSET + bytes.len() + 1];
        unsafe {
            let header = buf.as_mut_ptr() as *mut Header;
            (*header).hash = 0;
            (*header).length = bytes.len();
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.as_mut_ptr().add(BUILTIN_STRING_CHARS_OFFSET),
                bytes.len(),
            );
        }
        Self { buf }
    }

    pub fn as_usize(&self) -> usize {
        self.buf.as_ptr() as usize
    }
}

/// `len(s)` on an `rstr.STR`. The trace records `Strlen` for this call.
pub fn ll_strlen(s: usize) -> i64 {
    assert_ne!(s, 0, "ll_strlen: null string");
    unsafe { (s.wrapping_add(BUILTIN_STRING_LEN_OFFSET) as *const usize).read_unaligned() as i64 }
}

/// `s[i]` as a byte, zero-extended. The trace records `Strgetitem`.
pub fn ll_strgetitem(s: usize, index: i64) -> i64 {
    let len = ll_strlen(s);
    assert!(
        index >= 0 && index < len,
        "ll_strgetitem: index {index} outside 0..{len}"
    );
    unsafe {
        (s.wrapping_add(BUILTIN_STRING_CHARS_OFFSET)
            .wrapping_add(index as usize) as *const u8)
            .read_unaligned() as i64
    }
}
