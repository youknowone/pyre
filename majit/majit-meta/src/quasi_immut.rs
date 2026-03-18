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
        }
    }

    /// Register a compiled loop's invalidation flag.
    /// Called after compile_loop() when the trace contains QUASIIMMUT_FIELD.
    pub fn register(&mut self, flag: &Arc<AtomicBool>) {
        self.watchers.push(Arc::downgrade(flag));
    }

    /// Invalidate all registered loops.
    /// Called when the quasi-immutable field's value changes.
    pub fn invalidate(&mut self) {
        for watcher in &self.watchers {
            if let Some(flag) = watcher.upgrade() {
                flag.store(true, Ordering::Release);
            }
        }
        // Remove dead weak references
        self.watchers.retain(|w| w.strong_count() > 0);
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
    pub fn new(
        obj_ref: u64,
        field_descr_idx: u32,
        cached_value: i64,
    ) -> Self {
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

        qi.invalidate();
        assert!(flag1.load(Ordering::Acquire));
        assert_eq!(qi.num_watchers(), 1); // dead ref removed
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
}
