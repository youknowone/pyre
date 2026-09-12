//! llsupport/gcmap.py: GC bitmap allocation helpers.

use crate::arch::WORD;

/// Two-word maps (`malloc_size == 2`) are 16 B. They are never freed
/// (`Box::into_raw` leak), so mint them from reserved chunks instead
/// of the 16-byte malloc class — one per guard on the compile path.
const GCMAP16_WORDS: usize = 2;
const GCMAP16_CHUNK: usize = 4096;

struct Gcmap16Heap {
    chunks: Vec<(*mut usize, usize)>,
}

unsafe impl Send for Gcmap16Heap {}
unsafe impl Sync for Gcmap16Heap {}

static GCMAP16_HEAP: std::sync::Mutex<Gcmap16Heap> =
    std::sync::Mutex::new(Gcmap16Heap { chunks: Vec::new() });

fn alloc_gcmap16() -> *mut usize {
    let mut heap = GCMAP16_HEAP.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((base, used)) = heap.chunks.last_mut()
        && *used + GCMAP16_WORDS <= GCMAP16_CHUNK * GCMAP16_WORDS
    {
        let p = unsafe { (*base).add(*used) };
        *used += GCMAP16_WORDS;
        unsafe {
            p.write(1);
            p.add(1).write(0);
        }
        return p;
    }
    let layout = std::alloc::Layout::array::<usize>(GCMAP16_CHUNK * GCMAP16_WORDS)
        .expect("16-byte gcmap chunk");
    let base = unsafe { std::alloc::alloc_zeroed(layout) as *mut usize };
    assert!(!base.is_null(), "16-byte gcmap chunk alloc failed");
    heap.chunks.push((base, GCMAP16_WORDS));
    unsafe {
        base.write(1);
    }
    base
}

/// llsupport/gcmap.py allocate_gcmap.
pub fn allocate_gcmap(frame_depth: usize, fixed_size: usize) -> *mut usize {
    let size = frame_depth + fixed_size;
    let malloc_size = (size / WORD / 8 + 1) + 1;
    if malloc_size == GCMAP16_WORDS {
        return alloc_gcmap16();
    }
    let mut gcmap = vec![0usize; malloc_size].into_boxed_slice();
    gcmap[0] = malloc_size - 1;
    Box::into_raw(gcmap) as *mut usize
}

#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "this mirrors RPython's GC-map primitive: callers receive the pointer only from allocate_gcmap, and the null check is intentionally part of the total safe helper contract"
)]
pub fn gcmap_set_bit(gcmap: *mut usize, index: usize) {
    if gcmap.is_null() {
        return;
    }
    let word_index = index / (WORD * 8);
    let bit_index = index % (WORD * 8);
    unsafe {
        *gcmap.add(1 + word_index) |= 1usize << bit_index;
    }
}
