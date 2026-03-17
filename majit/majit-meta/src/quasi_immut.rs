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
}

impl Default for QuasiImmut {
    fn default() -> Self {
        Self::new()
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
