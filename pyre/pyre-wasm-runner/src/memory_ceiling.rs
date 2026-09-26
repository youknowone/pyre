//! `PYRE_MAX_MEMORY` for the wasm host. Same spelling as the native launchers:
//! default 8 GiB, `0` unbounded, `K`/`M`/`G` suffix. The host allocator is a
//! live-byte counter around mimalloc: v3 `mi_option_limit_os_alloc` does not
//! cover segment `_mi_os_alloc`. Guest linear memory is capped separately
//! (`wasmtime` / `wasmi` `ResourceLimiter`).

use std::alloc::{GlobalAlloc, Layout};
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const DEFAULT_MAX_MEMORY: u64 = 8 * 1024 * 1024 * 1024;

static INSTALLED: Once = Once::new();
static ARMED: AtomicUsize = AtomicUsize::new(0);
static CHARGED: AtomicUsize = AtomicUsize::new(0);
static LINE: std::sync::Mutex<Vec<u8>> = std::sync::Mutex::new(Vec::new());

pub fn parse_max_memory(raw: Option<&str>) -> Result<u64, String> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_MAX_MEMORY);
    };
    let text = raw.trim();
    if text.is_empty() {
        return Err("empty value".into());
    }
    if text.starts_with('-') {
        return Err(format!("negative value {text:?}"));
    }
    let mut body = text;
    if body.len() > 1 && matches!(body.as_bytes().last(), Some(b'b' | b'B')) {
        body = &body[..body.len() - 1];
    }
    if body.len() > 1 && matches!(body.as_bytes().last(), Some(b'i' | b'I')) {
        body = &body[..body.len() - 1];
    }
    let (number, factor) = match body.as_bytes().last().copied() {
        Some(b'k' | b'K') => (&body[..body.len() - 1], 1024u64),
        Some(b'm' | b'M') => (&body[..body.len() - 1], 1024 * 1024),
        Some(b'g' | b'G') => (&body[..body.len() - 1], 1024 * 1024 * 1024),
        _ => (body, 1u64),
    };
    if number.is_empty() {
        return Err(format!("missing number in {text:?}"));
    }
    let value: u64 = number
        .parse()
        .map_err(|_| format!("not a byte count: {text:?}"))?;
    if value == 0 {
        return Ok(0);
    }
    value
        .checked_mul(factor)
        .ok_or_else(|| format!("overflows a byte count: {text:?}"))
}

pub fn install() {
    let raw = std::env::var("PYRE_MAX_MEMORY").ok();
    let limit = match parse_max_memory(raw.as_deref()) {
        Ok(limit) => limit,
        Err(error) => {
            eprintln!("pyre-wasm-runner: PYRE_MAX_MEMORY: {error}");
            std::process::exit(2);
        }
    };
    INSTALLED.call_once(|| {
        let bytes = usize::try_from(limit).unwrap_or(usize::MAX);
        ARMED.store(bytes, Ordering::Relaxed);
        if bytes == 0 {
            return;
        }
        let line = format!(
            "pyre: process memory limit of {bytes} bytes exceeded (PYRE_MAX_MEMORY; 0 = unbounded)\n"
        );
        *LINE.lock().expect("ceiling line") = line.into_bytes();
    });
}

pub fn bytes() -> usize {
    ARMED.load(Ordering::Relaxed)
}

pub fn emit_once() {
    static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if ONCE.swap(true, Ordering::Relaxed) {
        return;
    }
    let line = LINE.lock().expect("ceiling line");
    let bytes: &[u8] = if line.is_empty() {
        b"pyre: process memory limit exceeded (PYRE_MAX_MEMORY; 0 = unbounded)\n"
    } else {
        &line
    };
    unsafe {
        libc::write(2, bytes.as_ptr().cast(), bytes.len() as _);
    }
}

fn try_charge(size: usize) -> bool {
    if size == 0 {
        return true;
    }
    let limit = ARMED.load(Ordering::Relaxed);
    if limit == 0 {
        return true;
    }
    let prev = CHARGED.fetch_add(size, Ordering::Relaxed);
    if prev.saturating_add(size) > limit {
        let _ = CHARGED.fetch_sub(size, Ordering::Relaxed);
        return false;
    }
    true
}

fn uncharge(size: usize) {
    if size == 0 || ARMED.load(Ordering::Relaxed) == 0 {
        return;
    }
    // Saturating: a block allocated before the ceiling was armed was never
    // charged, and freeing it must not wrap the counter.
    let _ = CHARGED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |charged| {
        Some(charged.saturating_sub(size))
    });
}

fn refuse() -> *mut u8 {
    emit_once();
    std::ptr::null_mut()
}

pub struct ProcessAllocator;

impl ProcessAllocator {
    pub const fn new() -> Self {
        Self
    }
}

unsafe impl GlobalAlloc for ProcessAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let charge = layout.size();
        if !try_charge(charge) {
            return refuse();
        }
        let ptr = unsafe { mimalloc::MiMalloc.alloc(layout) };
        if ptr.is_null() {
            uncharge(charge);
            return std::ptr::null_mut();
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let charge = layout.size();
        if !try_charge(charge) {
            return refuse();
        }
        let ptr = unsafe { mimalloc::MiMalloc.alloc_zeroed(layout) };
        if ptr.is_null() {
            uncharge(charge);
            return std::ptr::null_mut();
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { mimalloc::MiMalloc.dealloc(ptr, layout) };
        uncharge(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old = layout.size();
        let new = new_size;
        if new > old && !try_charge(new - old) {
            return refuse();
        }
        let new_ptr = unsafe { mimalloc::MiMalloc.realloc(ptr, layout, new_size) };
        if new_ptr.is_null() {
            if new > old {
                uncharge(new - old);
            }
            return std::ptr::null_mut();
        }
        if new < old {
            uncharge(old - new);
        }
        new_ptr
    }
}

/// Caps one guest linear memory at the process ceiling.
///
/// wasmtime maps linear memory itself, outside the global allocator, so its
/// limiter charges each growth delta to the same `CHARGED` total the host
/// allocator uses. wasmi keeps linear memory in a `Vec<u8>`, which the global
/// allocator already charges.
#[derive(Default)]
pub struct GuestMemoryLimit {
    pub ceiling: usize,
    /// Bytes charged for the growth wasmtime is attempting, returned if the
    /// growth then fails.
    pending: usize,
}

impl wasmtime::ResourceLimiter for GuestMemoryLimit {
    fn memory_growing(
        &mut self,
        current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        if self.ceiling != 0 && desired > self.ceiling {
            emit_once();
            return Ok(false);
        }
        let delta = desired.saturating_sub(current);
        if !try_charge(delta) {
            emit_once();
            return Ok(false);
        }
        self.pending = delta;
        Ok(true)
    }

    fn memory_grow_failed(&mut self, _error: wasmtime::Error) -> wasmtime::Result<()> {
        uncharge(std::mem::take(&mut self.pending));
        Ok(())
    }

    fn table_growing(
        &mut self,
        _current: usize,
        _desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        Ok(true)
    }
}

impl wasmi::ResourceLimiter for GuestMemoryLimit {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> Result<bool, wasmi_core::LimiterError> {
        if self.ceiling != 0 && desired > self.ceiling {
            emit_once();
            return Ok(false);
        }
        Ok(true)
    }

    fn table_growing(
        &mut self,
        _current: usize,
        _desired: usize,
        _maximum: Option<usize>,
    ) -> Result<bool, wasmi_core::LimiterError> {
        Ok(true)
    }

    fn instances(&self) -> usize {
        10_000
    }

    fn tables(&self) -> usize {
        10_000
    }

    fn memories(&self) -> usize {
        10_000
    }
}
