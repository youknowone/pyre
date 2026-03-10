/// Object header layout for GC-managed objects.
///
/// Each GC object starts with a single machine word that stores both
/// the type ID (lower 32 bits) and GC flags (upper 32 bits), matching
/// incminimark's HDR.tid layout where `first_gcflag = 1 << (LONG_BIT//2)`.

/// Number of bits reserved for the type ID in the lower half.
pub const TYPE_ID_BITS: u32 = 32;
pub const TYPE_ID_MASK: u64 = (1u64 << TYPE_ID_BITS) - 1;

/// Shift applied to flag constants from `crate::flags` to position them
/// in the upper half of the header word. The flags module defines flags
/// at positions 0..N, but in the header they live at positions 32..32+N.
pub const FLAG_SHIFT: u32 = TYPE_ID_BITS;

/// The sentinel value written into a forwarded nursery object's header.
/// Equivalent to incminimark's tid = -42 marker (all bits set as i64 -42,
/// but we just use a distinctive pattern).
pub const FORWARDED_MARKER: u64 = 0xFFFF_FFFF_FFFF_FFD6; // -42 as u64

/// GC object header, placed immediately before the object payload.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GcHeader {
    /// Lower 32 bits: type ID. Upper 32 bits: GC flags (shifted).
    pub tid_and_flags: u64,
}

impl GcHeader {
    /// Size of the header in bytes.
    pub const SIZE: usize = std::mem::size_of::<GcHeader>();

    /// Minimum object size in the nursery: header + one pointer (for forwarding).
    pub const MIN_NURSERY_OBJ_SIZE: usize = Self::SIZE + std::mem::size_of::<usize>();

    /// Create a new header with the given type ID and no flags set.
    pub fn new(type_id: u32) -> Self {
        GcHeader {
            tid_and_flags: type_id as u64,
        }
    }

    /// Create a new header with the given type ID and initial flags.
    /// `raw_flags` are the flag constants from `crate::flags` (unshifted).
    pub fn with_flags(type_id: u32, raw_flags: u64) -> Self {
        GcHeader {
            tid_and_flags: ((raw_flags) << FLAG_SHIFT) | (type_id as u64),
        }
    }

    /// Extract the type ID from the lower bits.
    #[inline]
    pub fn type_id(self) -> u32 {
        (self.tid_and_flags & TYPE_ID_MASK) as u32
    }

    /// Extract the raw flags (shifted back to positions 0..N).
    #[inline]
    pub fn flags(self) -> u64 {
        self.tid_and_flags >> FLAG_SHIFT
    }

    /// Set a flag bit. `flag` is an unshifted constant from `crate::flags`.
    #[inline]
    pub fn set_flag(&mut self, flag: u64) {
        self.tid_and_flags |= flag << FLAG_SHIFT;
    }

    /// Clear a flag bit. `flag` is an unshifted constant from `crate::flags`.
    #[inline]
    pub fn clear_flag(&mut self, flag: u64) {
        self.tid_and_flags &= !(flag << FLAG_SHIFT);
    }

    /// Test whether a flag bit is set. `flag` is an unshifted constant from `crate::flags`.
    #[inline]
    pub fn has_flag(self, flag: u64) -> bool {
        self.tid_and_flags & (flag << FLAG_SHIFT) != 0
    }

    /// Check if this header indicates a forwarded object.
    #[inline]
    pub fn is_forwarded(self) -> bool {
        self.tid_and_flags == FORWARDED_MARKER
    }

    /// Mark this header as forwarded and store the forwarding address
    /// in the word immediately following the header.
    ///
    /// # Safety
    /// The caller must ensure that there is at least `size_of::<usize>()`
    /// bytes of writable memory after this header.
    #[inline]
    pub unsafe fn set_forwarding_address(&mut self, new_addr: usize) {
        self.tid_and_flags = FORWARDED_MARKER;
        let fwd_ptr = (self as *mut GcHeader).add(1) as *mut usize;
        fwd_ptr.write(new_addr);
    }

    /// Read the forwarding address from a forwarded object.
    ///
    /// # Safety
    /// Must only be called when `is_forwarded()` returns true.
    #[inline]
    pub unsafe fn forwarding_address(&self) -> usize {
        let fwd_ptr = (self as *const GcHeader).add(1) as *const usize;
        fwd_ptr.read()
    }
}

/// Read the GcHeader for an object at the given address.
///
/// # Safety
/// `obj_addr` must point to valid memory with a GcHeader immediately before it.
#[inline]
pub unsafe fn header_of(obj_addr: usize) -> &'static mut GcHeader {
    &mut *((obj_addr - GcHeader::SIZE) as *mut GcHeader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flags;

    #[test]
    fn test_new_header() {
        let hdr = GcHeader::new(42);
        assert_eq!(hdr.type_id(), 42);
        assert_eq!(hdr.flags(), 0);
    }

    #[test]
    fn test_with_flags() {
        let hdr = GcHeader::with_flags(7, flags::TRACK_YOUNG_PTRS | flags::VISITED);
        assert_eq!(hdr.type_id(), 7);
        assert!(hdr.has_flag(flags::TRACK_YOUNG_PTRS));
        assert!(hdr.has_flag(flags::VISITED));
        assert!(!hdr.has_flag(flags::PINNED));
    }

    #[test]
    fn test_set_clear_flag() {
        let mut hdr = GcHeader::new(1);
        assert!(!hdr.has_flag(flags::TRACK_YOUNG_PTRS));

        hdr.set_flag(flags::TRACK_YOUNG_PTRS);
        assert!(hdr.has_flag(flags::TRACK_YOUNG_PTRS));
        assert_eq!(hdr.type_id(), 1);

        hdr.clear_flag(flags::TRACK_YOUNG_PTRS);
        assert!(!hdr.has_flag(flags::TRACK_YOUNG_PTRS));
        assert_eq!(hdr.type_id(), 1);
    }

    #[test]
    fn test_forwarded_marker() {
        let mut hdr = GcHeader::new(100);
        assert!(!hdr.is_forwarded());

        hdr.tid_and_flags = FORWARDED_MARKER;
        assert!(hdr.is_forwarded());
    }

    #[test]
    fn test_header_size() {
        assert_eq!(GcHeader::SIZE, 8);
    }

    #[test]
    fn test_flags_dont_clobber_type_id() {
        let mut hdr = GcHeader::new(0xDEAD);
        hdr.set_flag(flags::TRACK_YOUNG_PTRS);
        hdr.set_flag(flags::VISITED);
        hdr.set_flag(flags::PINNED);
        assert_eq!(hdr.type_id(), 0xDEAD);
        assert!(hdr.has_flag(flags::TRACK_YOUNG_PTRS));
        assert!(hdr.has_flag(flags::VISITED));
        assert!(hdr.has_flag(flags::PINNED));
    }

    #[test]
    fn test_multiple_flags() {
        let mut hdr = GcHeader::new(0);
        hdr.set_flag(flags::TRACK_YOUNG_PTRS);
        hdr.set_flag(flags::HAS_CARDS);
        hdr.set_flag(flags::CARDS_SET);

        assert!(hdr.has_flag(flags::TRACK_YOUNG_PTRS));
        assert!(hdr.has_flag(flags::HAS_CARDS));
        assert!(hdr.has_flag(flags::CARDS_SET));
        assert!(!hdr.has_flag(flags::VISITED));

        hdr.clear_flag(flags::HAS_CARDS);
        assert!(!hdr.has_flag(flags::HAS_CARDS));
        assert!(hdr.has_flag(flags::TRACK_YOUNG_PTRS));
    }

    #[test]
    fn test_forwarding_address() {
        // Allocate enough space for header + forwarding pointer
        let mut buf = [0u8; 16];
        let hdr = unsafe { &mut *(buf.as_mut_ptr() as *mut GcHeader) };
        *hdr = GcHeader::new(42);

        let target_addr: usize = 0xCAFEBABE;
        unsafe {
            hdr.set_forwarding_address(target_addr);
        }
        assert!(hdr.is_forwarded());
        assert_eq!(unsafe { hdr.forwarding_address() }, target_addr);
    }
}
