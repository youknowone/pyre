/// Object tracing for GC reference discovery.
///
/// During collection, the GC needs to find all GC references within
/// a live object so it can update them (for copying collection) or
/// mark the targets (for mark-sweep).
///
/// Instead of a closure-based trace approach (which causes lifetime issues
/// with the borrow checker), we use an offset-based approach: each type
/// declares the offsets of its GC pointer fields relative to the object
/// payload start. The collector reads/writes GcRef values at these offsets
/// directly.
use majit_ir::GcRef;

/// Registry mapping type IDs to their type descriptors and object sizes.
pub struct TypeRegistry {
    entries: Vec<TypeInfo>,
}

/// Custom trace function type.
///
/// RPython parity: `rgc.register_custom_trace_hook(TYPE, trace_fn)`.
/// When set, the GC calls this instead of the generic offset-based
/// tracing. Used for types with dynamic GC reference layouts
/// (e.g. JitFrame with gcmap bitmap).
///
/// `obj_addr` is the object payload start. The callback `f` must be
/// called for each GC reference slot address.
pub type CustomTraceFn = unsafe fn(obj_addr: usize, f: &mut dyn FnMut(*mut GcRef));

/// Information about a GC-managed type.
pub struct TypeInfo {
    /// Fixed size of the object (excluding header), or base size for varsize objects.
    pub size: usize,
    /// Whether this type contains GC pointers.
    pub has_gc_ptrs: bool,
    /// Byte offsets of GC pointer fields within the object payload (fixed part).
    pub gc_ptr_offsets: Vec<usize>,
    /// For variable-size objects: item size. 0 for fixed-size objects.
    pub item_size: usize,
    /// For variable-size objects: offset from object start to the length field.
    pub length_offset: usize,
    /// For variable-size objects: whether items contain GC pointers.
    /// If true, each item is treated as a GcRef.
    pub items_have_gc_ptrs: bool,
    /// RPython `rgc.register_custom_trace_hook` parity.
    /// When set, overrides offset-based tracing entirely.
    pub custom_trace: Option<CustomTraceFn>,
}

impl TypeInfo {
    /// Create a type info for a fixed-size object with no GC pointers.
    pub fn simple(size: usize) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: false,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            custom_trace: None,
        }
    }

    /// Create a type info for a fixed-size object with GC pointer fields.
    /// `offsets` lists byte offsets of GcRef fields within the payload.
    pub fn with_gc_ptrs(size: usize, offsets: Vec<usize>) -> Self {
        let has_gc_ptrs = !offsets.is_empty();
        TypeInfo {
            size,
            has_gc_ptrs,
            gc_ptr_offsets: offsets,
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            custom_trace: None,
        }
    }

    /// Create a type info for a variable-size object.
    pub fn varsize(
        base_size: usize,
        item_size: usize,
        length_offset: usize,
        items_have_gc_ptrs: bool,
        gc_ptr_offsets: Vec<usize>,
    ) -> Self {
        TypeInfo {
            size: base_size,
            has_gc_ptrs: !gc_ptr_offsets.is_empty() || items_have_gc_ptrs,
            gc_ptr_offsets,
            item_size,
            length_offset,
            items_have_gc_ptrs,
            custom_trace: None,
        }
    }

    /// Create a type info with a custom trace hook (fixed-size).
    ///
    /// RPython parity: `rgc.register_custom_trace_hook(TYPE, trace_fn)`.
    pub fn with_custom_trace(size: usize, trace_fn: CustomTraceFn) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: true,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            custom_trace: Some(trace_fn),
        }
    }

    /// Create a varsize type info with a custom trace hook.
    ///
    /// RPython parity: `rgc.register_custom_trace_hook` on a
    /// `GcStruct(..., Array(Signed))` — e.g. JITFRAME.
    pub fn varsize_with_custom_trace(
        base_size: usize,
        item_size: usize,
        length_offset: usize,
        trace_fn: CustomTraceFn,
    ) -> Self {
        TypeInfo {
            size: base_size,
            has_gc_ptrs: true,
            gc_ptr_offsets: Vec::new(),
            item_size,
            length_offset,
            items_have_gc_ptrs: false, // custom_trace handles ref tracing
            custom_trace: Some(trace_fn),
        }
    }

    /// Compute the total size of an instance (excluding GC header).
    /// RPython lltypelayout.py:93-100 sizeof(TYPE, i):
    ///   fixedsize = get_fixed_size(TYPE)
    ///   varsize = get_variable_size(TYPE)
    ///   return fixedsize + i * varsize
    ///
    /// Both `get_fixed_size(lltype.Array)` and the `_size` field of a
    /// `get_layout(Struct-with-array)` already account for the length
    /// word, so no extra WORD is added here.
    pub fn total_instance_size(&self, length: usize) -> usize {
        self.size + self.item_size * length
    }

    /// Iterate all GC pointer slot addresses for a given object.
    ///
    /// # Safety
    /// `obj_addr` must point to valid memory of at least `self.size` bytes (plus
    /// variable part if applicable).
    pub unsafe fn for_each_gc_ptr(&self, obj_addr: usize, mut f: impl FnMut(*mut GcRef)) {
        // RPython custom_trace_hook parity: if a custom trace function
        // is registered, use it instead of generic offset-based tracing.
        if let Some(trace_fn) = self.custom_trace {
            unsafe { trace_fn(obj_addr, &mut f) };
            return;
        }

        // Fixed-part GC pointer fields.
        for &offset in &self.gc_ptr_offsets {
            f((obj_addr + offset) as *mut GcRef);
        }

        // Variable-part GC pointer items.
        if self.items_have_gc_ptrs && self.item_size > 0 {
            let length = unsafe { *((obj_addr + self.length_offset) as *const usize) };
            let items_start = obj_addr + self.size;
            for i in 0..length {
                f((items_start + i * self.item_size) as *mut GcRef);
            }
        }
    }
}

impl TypeRegistry {
    pub fn new() -> Self {
        TypeRegistry {
            entries: Vec::new(),
        }
    }

    /// Register a new type and return its type ID.
    pub fn register(&mut self, info: TypeInfo) -> u32 {
        let id = self.entries.len() as u32;
        self.entries.push(info);
        id
    }

    /// Look up type info by ID.
    pub fn get(&self, type_id: u32) -> &TypeInfo {
        &self.entries[type_id as usize]
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_registry() {
        let mut reg = TypeRegistry::new();
        let id0 = reg.register(TypeInfo::simple(16));
        let id1 = reg.register(TypeInfo::simple(32));

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(reg.get(id0).size, 16);
        assert_eq!(reg.get(id1).size, 32);
        assert!(!reg.get(id0).has_gc_ptrs);
    }

    #[test]
    fn test_type_with_gc_ptrs() {
        let mut reg = TypeRegistry::new();
        let id = reg.register(TypeInfo::with_gc_ptrs(24, vec![0, 8, 16]));
        assert!(reg.get(id).has_gc_ptrs);
        assert_eq!(reg.get(id).gc_ptr_offsets.len(), 3);
    }

    #[test]
    fn test_varsize_type() {
        let info = TypeInfo::varsize(8, 8, 0, true, Vec::new());
        assert_eq!(info.total_instance_size(10), 88); // 8 + 8*10
        assert_eq!(info.total_instance_size(0), 8);
    }

    #[test]
    fn test_for_each_gc_ptr() {
        // Object with two GcRef fields at offsets 0 and 8.
        let info = TypeInfo::with_gc_ptrs(16, vec![0, 8]);

        let mut data = [0u8; 16];
        let obj_addr = data.as_mut_ptr() as usize;

        let mut visited = Vec::new();
        unsafe {
            info.for_each_gc_ptr(obj_addr, |ptr| {
                visited.push(ptr as usize);
            });
        }

        assert_eq!(visited.len(), 2);
        assert_eq!(visited[0], obj_addr);
        assert_eq!(visited[1], obj_addr + 8);
    }

    #[test]
    fn test_for_each_gc_ptr_varsize() {
        // Variable-size object: base_size=8 (length field at offset 0),
        // items are GcRef (item_size=8).
        let info = TypeInfo::varsize(8, 8, 0, true, Vec::new());

        // Layout: [length: usize][item0: GcRef][item1: GcRef][item2: GcRef]
        let mut data = [0u8; 32]; // 8 base + 3*8 items
        let obj_addr = data.as_mut_ptr() as usize;

        // Write length = 3.
        unsafe {
            *(obj_addr as *mut usize) = 3;
        }

        let mut visited = Vec::new();
        unsafe {
            info.for_each_gc_ptr(obj_addr, |ptr| {
                visited.push(ptr as usize);
            });
        }

        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], obj_addr + 8);
        assert_eq!(visited[1], obj_addr + 16);
        assert_eq!(visited[2], obj_addr + 24);
    }
}
