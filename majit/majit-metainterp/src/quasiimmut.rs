use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// quasiimmut.py: get_mutate_field_name(fieldname).
pub fn get_mutate_field_name(fieldname: &str) -> String {
    if let Some(rest) = fieldname.strip_prefix("inst_") {
        format!("mutate_{rest}")
    } else {
        panic!("{fieldname}")
    }
}

/// quasiimmut.py: get_current_qmut_instance(cpu, gcref, mutatefielddescr).
///
/// PyPy stores the `QuasiImmut` object in the object's mutate field via CPU
/// descriptor reads/writes. Pyre's current runtime-facing shape represents
/// that mutate field as a Rust cell; this helper preserves the same
/// get-or-create semantics without inventing a side table.
pub fn get_current_qmut_instance(
    mutate_field: &Mutex<Option<Arc<Mutex<QuasiImmut>>>>,
) -> Arc<Mutex<QuasiImmut>> {
    let mut field = mutate_field
        .lock()
        .expect("quasi-immutable mutate field mutex poisoned");
    if let Some(qmut) = field.as_ref() {
        return qmut.clone();
    }
    let qmut = Arc::new(Mutex::new(QuasiImmut::new()));
    *field = Some(qmut.clone());
    qmut
}

/// quasiimmut.py: make_invalidation_function(STRUCT, mutatefieldname).
///
/// The returned closure mirrors PyPy's invalidation function: if the mutate
/// field currently holds a `QuasiImmut`, clear the field and invalidate it.
pub fn make_invalidation_function(
    mutate_field: Arc<Mutex<Option<Arc<Mutex<QuasiImmut>>>>>,
) -> impl Fn() + Send + Sync + 'static {
    move || {
        let qmut = mutate_field
            .lock()
            .expect("quasi-immutable mutate field mutex poisoned")
            .take();
        if let Some(qmut) = qmut {
            qmut.lock()
                .expect("quasi-immutable instance mutex poisoned")
                .invalidate();
        }
    }
}

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
    /// quasiimmut.py:121 `self.struct` — the object whose field is
    /// quasi-immutable.
    pub obj_ref: u64,
    /// quasiimmut.py:122 `self.fielddescr` — the field descriptor index.
    pub field_descr_idx: u32,
    /// quasiimmut.py:125 `self.constantfieldbox` — the cached constant value
    /// (snapshot at descr creation).
    pub cached_value: i64,
    /// quasiimmut.py:123 `self.mutatefielddescr` — a handle to the object's
    /// mutate field (read upstream via `cpu.bh_getfield_gc_r(struct,
    /// mutatefielddescr)`), modeled as the clearable cell that holds the
    /// current `QuasiImmut` (or NULL).  This cell is owned by the object, so
    /// every descriptor for the same `(struct, mutatefielddescr)` is built
    /// from the SAME cell and shares one `QuasiImmut`; the caller supplies it.
    /// `do_force_quasi_immutable` clears this cell before invalidating.
    pub mutate_field: Arc<Mutex<Option<Arc<Mutex<QuasiImmut>>>>>,
    /// quasiimmut.py:124 `self.qmut` — the `QuasiImmut` captured at descr
    /// creation; `is_still_valid_for` compares the field's current qmut
    /// against this identity.
    pub qmut: Arc<Mutex<QuasiImmut>>,
}

impl QuasiImmutDescr {
    /// quasiimmut.py:119-125 `QuasiImmutDescr.__init__` — capture the current
    /// QuasiImmut instance from the object's mutate field.  `mutate_field` is
    /// the object's shared cell (`get_current_qmut_instance(cpu, struct,
    /// mutatefielddescr)`), so two descriptors for the same field share a qmut.
    pub fn new(
        obj_ref: u64,
        field_descr_idx: u32,
        cached_value: i64,
        mutate_field: Arc<Mutex<Option<Arc<Mutex<QuasiImmut>>>>>,
    ) -> Self {
        let qmut = get_current_qmut_instance(&mutate_field);
        QuasiImmutDescr {
            obj_ref,
            field_descr_idx,
            cached_value,
            mutate_field,
            qmut,
        }
    }

    /// Register a compiled loop that depends on this quasi-immutable value.
    /// quasiimmut.py: `descr.qmut.register_loop_token(wref)`.
    pub fn register_loop(&self, flag: &Arc<AtomicBool>) {
        if let Ok(mut qi) = self.qmut.lock() {
            qi.register(flag);
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

    /// quasiimmut.py:146-158 is_still_valid_for(structconst)
    ///
    /// Same object identity, same mutate-field qmut identity, AND same field
    /// value as cached.  After `do_force_quasi_immutable` clears the mutate
    /// field, `get_current_qmut_instance` mints a fresh `QuasiImmut`, so the
    /// identity check below fails and the descriptor is no longer valid.
    pub fn is_still_valid_for(&self, struct_ref: u64, field_offset: usize) -> bool {
        if self.obj_ref != struct_ref {
            return false;
        }
        let qmut = get_current_qmut_instance(&self.mutate_field);
        if !Arc::ptr_eq(&qmut, &self.qmut) {
            return false;
        }
        let current = self.get_current_constant_fieldvalue(field_offset);
        current == self.cached_value
    }
}

/// quasiimmut.py:46-51 do_force_quasi_immutable(cpu, p, mutatefielddescr)
///
/// Read the mutate field; if it holds a `QuasiImmut`, clear the field (set to
/// NULL) BEFORE invalidating all dependent compiled loops.  Clearing first is
/// what makes a later `is_still_valid_for` see a fresh qmut and fail.
///
/// Called by the interpreter when a quasi-immutable field is written.
pub fn do_force_quasi_immutable(descr: &QuasiImmutDescr) {
    let qmut = descr
        .mutate_field
        .lock()
        .expect("quasi-immutable mutate field mutex poisoned")
        .take();
    if let Some(qmut) = qmut {
        qmut.lock()
            .expect("quasi-immutable instance mutex poisoned")
            .invalidate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_mutate_field_name() {
        assert_eq!(get_mutate_field_name("inst_value"), "mutate_value");
    }

    #[test]
    #[should_panic(expected = "value")]
    fn test_get_mutate_field_name_rejects_non_instance_field() {
        get_mutate_field_name("value");
    }

    #[test]
    fn test_get_current_qmut_instance_reuses_mutate_field() {
        let mutate_field = Mutex::new(None);
        let qmut1 = get_current_qmut_instance(&mutate_field);
        let qmut2 = get_current_qmut_instance(&mutate_field);
        assert!(Arc::ptr_eq(&qmut1, &qmut2));
    }

    #[test]
    fn test_make_invalidation_function_clears_and_invalidates() {
        let mutate_field = Arc::new(Mutex::new(None));
        let qmut = get_current_qmut_instance(&mutate_field);
        let flag = Arc::new(AtomicBool::new(false));
        qmut.lock().unwrap().register(&flag);

        let invalidate = make_invalidation_function(mutate_field.clone());
        invalidate();

        assert!(flag.load(Ordering::Acquire));
        assert!(mutate_field.lock().unwrap().is_none());
    }

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
        let mutate_field = Arc::new(Mutex::new(None));
        let descr = QuasiImmutDescr::new(0x1000, 42, 99, mutate_field);
        assert_eq!(descr.obj_ref, 0x1000);
        assert_eq!(descr.field_descr_idx, 42);
        assert_eq!(descr.cached_value, 99);

        // Register and force through the descr
        let flag = Arc::new(AtomicBool::new(false));
        descr.register_loop(&flag);
        do_force_quasi_immutable(&descr);
        assert!(flag.load(Ordering::Acquire));
        // The mutate field is cleared, so the descr is no longer valid.
        assert!(!descr.is_still_valid_for(0x1000, 0));
    }

    #[test]
    fn test_quasi_immut_descr_shares_object_mutate_field() {
        // Two descriptors for the same object field share the mutate-field
        // cell, hence the same QuasiImmut (quasiimmut.py:124 via
        // get_current_qmut_instance reading the object's field).
        let mutate_field = Arc::new(Mutex::new(None));
        let descr1 = QuasiImmutDescr::new(0x3000, 7, 1, mutate_field.clone());
        let descr2 = QuasiImmutDescr::new(0x3000, 7, 1, mutate_field);
        assert!(Arc::ptr_eq(&descr1.qmut, &descr2.qmut));

        // Forcing through one clears the shared field, invalidating both.
        let f1 = Arc::new(AtomicBool::new(false));
        let f2 = Arc::new(AtomicBool::new(false));
        descr1.register_loop(&f1);
        descr2.register_loop(&f2);
        do_force_quasi_immutable(&descr1);
        assert!(f1.load(Ordering::Acquire));
        assert!(f2.load(Ordering::Acquire));
        assert!(!descr2.is_still_valid_for(0x3000, 0));
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
        let mutate_field = Arc::new(Mutex::new(None));
        let descr = QuasiImmutDescr::new(0x2000, 10, 55, mutate_field);
        let f1 = Arc::new(AtomicBool::new(false));
        let f2 = Arc::new(AtomicBool::new(false));
        descr.register_loop(&f1);
        descr.register_loop(&f2);
        // Force the mutate field — both flags should be set.
        do_force_quasi_immutable(&descr);
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
