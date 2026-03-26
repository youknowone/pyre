use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

/// Notifier for quasi-immutable fields.
///
/// When a quasi-immutable field changes, call `invalidate()` to mark
/// all compiled loops that depend on this field's value as invalid.
/// The next time a `GuardNotInvalidated` check runs in those loops,
/// it will fail and execution will fall back to the interpreter.
pub struct QuasiImmut {
    /// Weak references to JitCellToken invalidation flags.
    watchers: Vec<Weak<AtomicBool>>,
    /// quasiimmut.py: compress_limit — threshold for compressing dead refs.
    compress_limit: usize,
    /// Statistics: total number of invalidations performed.
    pub invalidation_count: u64,
}

impl std::fmt::Debug for QuasiImmut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuasiImmut")
            .field("num_watchers", &self.num_watchers())
            .finish()
    }
}

impl QuasiImmut {
    pub fn new() -> Self {
        Self {
            watchers: Vec::new(),
            compress_limit: 30,
            invalidation_count: 0,
        }
    }

    /// Register a compiled loop's invalidation flag.
    /// quasiimmut.py: register_loop_token(wref)
    pub fn register(&mut self, flag: &Arc<AtomicBool>) {
        if self.watchers.len() > self.compress_limit {
            self.compress();
        }
        self.watchers.push(Arc::downgrade(flag));
    }

    /// quasiimmut.py: compress_looptokens_list()
    /// Remove dead weak references and update compress_limit.
    pub fn compress(&mut self) {
        self.watchers.retain(|w| w.strong_count() > 0);
        self.compress_limit = (self.watchers.len() + 15) * 2;
    }

    /// Invalidate all registered loops.
    /// quasiimmut.py: invalidate(descr_repr)
    pub fn invalidate(&mut self) {
        let mut invalidated = 0u64;
        for watcher in &self.watchers {
            if let Some(flag) = watcher.upgrade() {
                invalidated += 1;
                flag.store(true, Ordering::Release);
            }
        }
        self.invalidation_count += invalidated;
        self.watchers.clear();
    }

    /// Number of live watchers.
    pub fn num_watchers(&self) -> usize {
        self.watchers
            .iter()
            .filter(|w| w.strong_count() > 0)
            .count()
    }

    /// Check if any watchers are still alive.
    pub fn has_watchers(&self) -> bool {
        self.watchers.iter().any(|w| w.strong_count() > 0)
    }

    /// Remove all dead references without invalidating.
    pub fn cleanup(&mut self) {
        self.watchers.retain(|w| w.strong_count() > 0);
    }
}

impl Default for QuasiImmut {
    fn default() -> Self {
        Self::new()
    }
}

/// quasiimmut.py: QuasiImmutDescr — descriptor binding a field to a QuasiImmut.
/// Associates a specific object field with a quasi-immutable notifier
/// and the cached constant value.
#[derive(Clone, Debug)]
pub struct QuasiImmutDescr {
    /// The object whose field is quasi-immutable.
    pub obj_ref: u64,
    /// The field descriptor index.
    pub field_descr_idx: u32,
    /// The cached constant value (snapshot at guard time).
    pub cached_value: i64,
    /// Reference to the QuasiImmut notifier.
    pub notifier: Arc<std::sync::Mutex<QuasiImmut>>,
}

impl QuasiImmutDescr {
    /// Create a new QuasiImmutDescr.
    pub fn new(obj_ref: u64, field_descr_idx: u32, cached_value: i64) -> Self {
        QuasiImmutDescr {
            obj_ref,
            field_descr_idx,
            cached_value,
            notifier: Arc::new(std::sync::Mutex::new(QuasiImmut::new())),
        }
    }

    /// Register a compiled loop that depends on this quasi-immutable value.
    pub fn register_loop(&self, flag: &Arc<AtomicBool>) {
        if let Ok(mut qi) = self.notifier.lock() {
            qi.register(flag);
        }
    }

    /// Invalidate all loops depending on this quasi-immutable value.
    pub fn invalidate(&self) {
        if let Ok(mut qi) = self.notifier.lock() {
            qi.invalidate();
        }
    }

    /// quasiimmut.py: get_parent_descr()
    /// Return the field descriptor index.
    pub fn get_parent_descr(&self) -> u32 {
        self.field_descr_idx
    }

    /// quasiimmut.py: get_index()
    /// Return the descriptor index (delegates to field_descr_idx).
    pub fn get_index(&self) -> u32 {
        self.field_descr_idx
    }

    /// quasiimmut.py: get_current_constant_fieldvalue()
    ///
    /// Read the current value of the quasi-immutable field from the
    /// concrete object. Returns the raw value at the field offset.
    pub fn get_current_constant_fieldvalue(&self, field_offset: usize) -> i64 {
        if self.obj_ref == 0 {
            return 0;
        }
        unsafe { *((self.obj_ref as *const u8).add(field_offset) as *const i64) }
    }

    /// quasiimmut.py: is_still_valid_for(structconst)
    ///
    /// Check if this descriptor is still valid for the given object:
    /// same object identity AND same field value as cached.
    pub fn is_still_valid_for(&self, struct_ref: u64, field_offset: usize) -> bool {
        if self.obj_ref != struct_ref {
            return false;
        }
        let current = self.get_current_constant_fieldvalue(field_offset);
        current == self.cached_value
    }
}

/// quasiimmut.py: do_force_quasi_immutable(cpu, p, mutatefielddescr)
/// Force a quasi-immutable mutation: clear the mutate field and
/// invalidate all dependent compiled loops.
///
/// Called by the interpreter when a quasi-immutable field is written.
pub fn do_force_quasi_immutable(descr: &QuasiImmutDescr) {
    descr.invalidate();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_invalidate() {
        let mut qi = QuasiImmut::new();
        let flag1 = Arc::new(AtomicBool::new(false));
        let flag2 = Arc::new(AtomicBool::new(false));
        let flag3 = Arc::new(AtomicBool::new(false));

        qi.register(&flag1);
        qi.register(&flag2);
        qi.register(&flag3);
        assert_eq!(qi.num_watchers(), 3);

        qi.invalidate();
        assert!(flag1.load(Ordering::Acquire));
        assert!(flag2.load(Ordering::Acquire));
        assert!(flag3.load(Ordering::Acquire));
    }

    #[test]
    fn test_dead_refs_cleaned() {
        let mut qi = QuasiImmut::new();
        let flag1 = Arc::new(AtomicBool::new(false));
        qi.register(&flag1);

        {
            let flag2 = Arc::new(AtomicBool::new(false));
            qi.register(&flag2);
        }
        // flag2 is dropped

        // quasiimmut.py: compress_looptokens_list() removes dead refs
        qi.compress();
        assert_eq!(qi.num_watchers(), 1); // dead ref removed

        qi.invalidate();
        assert!(flag1.load(Ordering::Acquire));
        // quasiimmut.py: invalidate() clears the list
        assert_eq!(qi.num_watchers(), 0);
    }

    #[test]
    fn test_multiple_invalidations() {
        let mut qi = QuasiImmut::new();
        let flag = Arc::new(AtomicBool::new(false));
        qi.register(&flag);

        qi.invalidate();
        assert!(flag.load(Ordering::Acquire));

        // Reset and re-register
        flag.store(false, Ordering::Release);
        qi.register(&flag);
        qi.invalidate();
        assert!(flag.load(Ordering::Acquire));
    }

    #[test]
    fn test_quasi_immut_descr() {
        let descr = QuasiImmutDescr::new(0x1000, 42, 99);
        assert_eq!(descr.obj_ref, 0x1000);
        assert_eq!(descr.field_descr_idx, 42);
        assert_eq!(descr.cached_value, 99);

        // Register and invalidate through the descr
        let flag = Arc::new(AtomicBool::new(false));
        descr.register_loop(&flag);
        descr.invalidate();
        assert!(flag.load(Ordering::Acquire));
    }

    #[test]
    fn test_has_watchers() {
        let mut qi = QuasiImmut::new();
        assert!(!qi.has_watchers());

        let flag = Arc::new(AtomicBool::new(false));
        qi.register(&flag);
        assert!(qi.has_watchers());

        drop(flag);
        qi.cleanup();
        assert!(!qi.has_watchers());
    }

    #[test]
    fn test_quasi_immut_descr_multi_loop() {
        let descr = QuasiImmutDescr::new(0x2000, 10, 55);
        let f1 = Arc::new(AtomicBool::new(false));
        let f2 = Arc::new(AtomicBool::new(false));
        descr.register_loop(&f1);
        descr.register_loop(&f2);
        // Only invalidate — both flags should be set.
        descr.invalidate();
        assert!(f1.load(Ordering::Acquire));
        assert!(f2.load(Ordering::Acquire));
    }

    #[test]
    fn test_num_watchers_after_invalidate() {
        let mut qi = QuasiImmut::new();
        let f1 = Arc::new(AtomicBool::new(false));
        let f2 = Arc::new(AtomicBool::new(false));
        qi.register(&f1);
        qi.register(&f2);
        assert_eq!(qi.num_watchers(), 2);
        qi.invalidate();
        // quasiimmut.py: invalidate() clears the watcher list.
        assert_eq!(qi.num_watchers(), 0);
        assert_eq!(qi.invalidation_count, 2);
    }

    #[test]
    fn test_debug_format() {
        let qi = QuasiImmut::new();
        let debug = format!("{:?}", qi);
        assert!(debug.contains("QuasiImmut"));
    }
}
