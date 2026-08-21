use majit_ir::IndexMapExt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// counter.py: JitCounter — float-based 5-way associative timetable.
///
/// Direct port of rpython/jit/metainterp/counter.py.
/// Uses f32 time values (0.0 to 1.0) instead of integer counts.
/// tick(hash, increment) adds increment; fires when >= 1.0.
/// 5-way associative cache indexed by _get_index(hash), matched by
/// _get_subhash(hash). MRU promotion via _swap.
///
/// counter.py:82 DEFAULT_SIZE = 2048
pub const DEFAULT_SIZE: usize = 2048;

/// counter.py ENTRY: 5 (f32 time, u16 subhash) pairs per bucket.
const ASSOCIATIVITY: usize = 5;

/// counter.py UINT32MAX = 2 ** 32 - 1
const UINT32MAX: u64 = 0xFFFF_FFFF;

static MINOR_COLLECTION_STEP: AtomicUsize = AtomicUsize::new(0);
static DECAY_GENERATION: AtomicUsize = AtomicUsize::new(0);

/// counter.py invoke_after_minor_collection
///
/// This runs inside a minor collection, so it must remain allocation-free and
/// must not touch the counter table or acquire a lock.
fn invoke_after_minor_collection() {
    let step = MINOR_COLLECTION_STEP.fetch_add(1, Ordering::Relaxed) + 1;
    if step == 32 {
        MINOR_COLLECTION_STEP.store(0, Ordering::Relaxed);
        DECAY_GENERATION.fetch_add(1, Ordering::Relaxed);
    }
}

/// One timetable entry: 5-way associative (time, subhash) pairs.
/// counter.py ENTRY struct.
#[derive(Clone)]
struct Entry {
    /// counter.py: times — f32 timing values, 0.0 to 1.0.
    times: [f32; ASSOCIATIVITY],
    /// counter.py: subhashes — lower 16 bits of the hash.
    subhashes: [u16; ASSOCIATIVITY],
}

impl Default for Entry {
    fn default() -> Self {
        Entry {
            times: [0.0; ASSOCIATIVITY],
            subhashes: [0; ASSOCIATIVITY],
        }
    }
}

/// counter.py JitCounter
pub struct JitCounter {
    /// counter.py:86 size
    size: usize,
    /// counter.py:87 shift
    shift: u32,
    /// counter.py:97 timetable
    timetable: Vec<Entry>,
    /// counter.py:100 _nexthash
    _nexthash: u64,
    /// counter.py:264 decay_by_mult — f64 (Python float).
    decay_by_mult: f64,
    /// Last `DECAY_GENERATION` this counter applied. Each counter tracks its
    /// own, so one thread's tick cannot consume another counter's decay.
    last_decay_generation: usize,
}

impl JitCounter {
    /// counter.py __init__(self, size=DEFAULT_SIZE, translator=None)
    pub fn new(size: usize) -> Self {
        majit_gc::register_after_minor_collection_hook(invoke_after_minor_collection);
        let mut shift = 16u32;
        while (UINT32MAX >> shift) != (size as u64 - 1) {
            shift += 1;
            assert!(shift < 999, "size is not a power of two <= 2**16");
        }
        JitCounter {
            size,
            shift,
            timetable: vec![Entry::default(); size],
            _nexthash: 0,
            decay_by_mult: 1.0,
            last_decay_generation: DECAY_GENERATION.load(Ordering::Relaxed),
        }
    }

    /// counter.py compute_threshold
    pub fn compute_threshold(&self, threshold: u32) -> f64 {
        if threshold == 0 {
            return 0.0;
        }
        1.0_f64 / (threshold as f64 - 0.001)
    }

    /// counter.py `self.size = size` — the entry count both tables are
    /// indexed with. `counter.py:97,103` size the timetable and the celltable
    /// from this one number, so a table living outside this struct has to read
    /// it from here rather than pick its own.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size
    }

    /// counter.py _get_index
    ///
    /// ```text
    ///  def _get_index(self, hash):
    ///      hash32 = r_uint(r_uint32(hash))  # mask off the bits higher than 32
    ///      index = hash32 >> self.shift     # shift, resulting in a value < size
    ///      return index                     # return the result as a r_uint
    /// ```
    ///
    /// Public because the timetable is not the only table this indexes:
    /// `counter.py:239-240` reads the celltable through the same call, and the
    /// two tables must agree about which entry a hash names.
    #[inline(always)]
    pub fn _get_index(&self, hash: u64) -> usize {
        let hash32 = hash as u32 as u64;
        (hash32 >> self.shift) as usize
    }

    /// counter.py _get_subhash
    #[inline(always)]
    fn _get_subhash(hash: u64) -> u16 {
        (hash & 0xFFFF) as u16
    }

    /// counter.py fetch_next_hash
    pub fn fetch_next_hash(&mut self) -> u64 {
        let result = self._nexthash;
        self._nexthash =
            result.wrapping_add(1 | (1u64 << self.shift) | (1u64 << (self.shift - 16)));
        result
    }

    /// counter.py _swap
    #[inline(always)]
    fn _swap(entry: &mut Entry, n: usize) -> usize {
        if entry.times[n] > entry.times[n + 1] {
            n + 1
        } else {
            entry.times.swap(n, n + 1);
            entry.subhashes.swap(n, n + 1);
            n
        }
    }

    /// counter.py _tick_slowpath
    fn _tick_slowpath(entry: &mut Entry, subhash: u16) -> usize {
        if entry.subhashes[1] == subhash {
            Self::_swap(entry, 0)
        } else if entry.subhashes[2] == subhash {
            Self::_swap(entry, 1)
        } else if entry.subhashes[3] == subhash {
            Self::_swap(entry, 2)
        } else if entry.subhashes[4] == subhash {
            Self::_swap(entry, 3)
        } else {
            let mut n = 4;
            while n > 0 && entry.times[n - 1] == 0.0 {
                n -= 1;
            }
            entry.subhashes[n] = subhash;
            entry.times[n] = 0.0;
            n
        }
    }

    /// TODO: no RPython counterpart. Read-only peek
    /// used by warmstate's cold fast path to avoid GreenKey allocation.
    pub fn would_tick_fire(&self, hash: u64, increment: f64) -> bool {
        let elapsed = DECAY_GENERATION
            .load(Ordering::Relaxed)
            .wrapping_sub(self.last_decay_generation);
        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &self.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                // This predicate is &self, so decay-adjust the read instead of
                // mutating the table to apply the pending generations. Step
                // through them one at a time rather than raising the multiplier
                // to `elapsed`: every step rounds back to f32, and this answer
                // has to be the one a `tick` would give once it drains the same
                // generations. Narrow the multiplier the way decay_all_counters
                // does, for the same reason.
                let mult = self.decay_by_mult as f32;
                let mut time = entry.times[i];
                for _ in 0..elapsed {
                    time *= mult;
                }
                return time as f64 + increment >= 1.0;
            }
        }
        increment >= 1.0
    }

    /// counter.py tick(self, hash, increment)
    #[inline(always)]
    pub fn tick(&mut self, hash: u64, increment: f64) -> bool {
        // counter.py:104-121 applies the decay synchronously inside the minor
        // collection, where the hook closes over the process's one JitCounter.
        // pyre defers it to the next table access instead: the counter is
        // reached through the `JIT_DRIVER` cell in eval.rs, whose accessor
        // mints a `&'static mut JitDriverPair`, and a minor collection can be
        // triggered by an allocation the metainterp makes while already holding
        // one. Decaying from inside the collector would alias it.
        //
        // Each JitCounter keeps its own last-seen generation because JIT_DRIVER
        // is thread-local, giving each mutator thread its own counter.
        // Every elapsed interval is applied before every mutating table access;
        // would_tick_fire decay-adjusts its read. A value written after a
        // collection is therefore not retro-decayed.
        self.apply_pending_decay();

        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &mut self.timetable[index];

        let n = if entry.subhashes[0] == subhash {
            0
        } else {
            Self::_tick_slowpath(entry, subhash)
        };

        // counter.py:194: counter = float(p_entry.times[n]) + increment
        let counter: f64 = entry.times[n] as f64 + increment;
        if counter < 1.0 {
            // counter.py:196: p_entry.times[n] = r_singlefloat(counter)
            entry.times[n] = counter as f32;
            false
        } else {
            // counter.py: self.reset(hash); return True
            self.reset(hash);
            true
        }
    }

    /// counter.py change_current_fraction(hash, new_fraction)
    pub fn change_current_fraction(&mut self, hash: u64, new_fraction: f64) {
        self.apply_pending_decay();

        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &mut self.timetable[index];

        let mut n = 0;
        while n < 4 && entry.subhashes[n] != subhash && entry.times[n] != 0.0 {
            n += 1;
        }
        while n > 0 {
            n -= 1;
            entry.subhashes[n + 1] = entry.subhashes[n];
            entry.times[n + 1] = entry.times[n];
        }
        entry.subhashes[0] = subhash;
        entry.times[0] = new_fraction as f32;
    }

    /// counter.py reset(hash)
    pub fn reset(&mut self, hash: u64) {
        self.apply_pending_decay();

        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &mut self.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                entry.times[i] = 0.0;
            }
        }
    }

    /// TODO: no RPython equivalent.
    /// Zero all timetable entries.
    pub fn reset_all(&mut self) {
        self.apply_pending_decay();

        for entry in &mut self.timetable {
            *entry = Entry::default();
        }
    }

    /// counter.py set_decay(decay)
    pub fn set_decay(&mut self, decay: i32) {
        self.apply_pending_decay();

        let clamped = decay.clamp(0, 1000);
        self.decay_by_mult = 1.0_f64 - (clamped as f64 * 0.001);
    }

    /// counter.py decay_all_counters()
    ///
    /// counter.py hands `decay_by_mult` to `pypy__decay_jit_counters`,
    /// whose C body narrows it with `float f = (float)f1` once and then
    /// multiplies each entry in single precision. Narrowing the multiplier here
    /// rather than the product keeps those bits: widening the entry to f64,
    /// multiplying, and narrowing back rounds twice, and the two disagree
    /// whenever the exact product lands near an f32 tie.
    pub fn decay_all_counters(&mut self) {
        let mult = self.decay_by_mult as f32;
        for entry in &mut self.timetable {
            for time in &mut entry.times {
                *time *= mult;
            }
        }
    }

    /// Apply every 32-collection interval that elapsed since this counter
    /// last looked. counter.py:104-121 decays inside the collection, so a
    /// pending decay must land before anything reads or writes the table —
    /// otherwise it would decay values written after the collection.
    fn apply_pending_decay(&mut self) {
        let generation = DECAY_GENERATION.load(Ordering::Relaxed);
        while self.last_decay_generation != generation {
            self.last_decay_generation = self.last_decay_generation.wrapping_add(1);
            self.decay_all_counters();
        }
    }
}

/// counter.py DeterministicJitCounter — test-only, NOT_RPYTHON.
///
/// RPython: subclasses JitCounter, overrides _get_index to return the
/// raw hash (identity — no collision), uses a defaultdict timetable.
/// Rust: uses a IndexMap<u64, Entry> to mirror the defaultdict approach.
pub struct DeterministicJitCounter {
    entries: indexmap::IndexMap<u64, Entry>,
}

impl Default for DeterministicJitCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl DeterministicJitCounter {
    /// counter.py DeterministicJitCounter.__init__
    pub fn new() -> Self {
        DeterministicJitCounter {
            entries: indexmap::IndexMap::new(),
        }
    }

    /// counter.py _get_index — identity (no hash collision).
    #[inline(always)]
    fn _get_index(hash: u64) -> u64 {
        hash
    }

    /// counter.py _get_subhash
    #[inline(always)]
    fn _get_subhash(hash: u64) -> u16 {
        (hash & 0xFFFF) as u16
    }

    /// counter.py compute_threshold
    pub fn compute_threshold(&self, threshold: u32) -> f64 {
        if threshold == 0 {
            return 0.0;
        }
        1.0_f64 / (threshold as f64 - 0.001)
    }

    /// counter.py tick — same logic but using identity _get_index.
    pub fn tick(&mut self, hash: u64, increment: f64) -> bool {
        let key = Self::_get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = self.entries.entry_or_insert_with(key, Entry::default);

        let n = if entry.subhashes[0] == subhash {
            0
        } else if entry.subhashes[1] == subhash {
            JitCounter::_swap(entry, 0)
        } else if entry.subhashes[2] == subhash {
            JitCounter::_swap(entry, 1)
        } else if entry.subhashes[3] == subhash {
            JitCounter::_swap(entry, 2)
        } else if entry.subhashes[4] == subhash {
            JitCounter::_swap(entry, 3)
        } else {
            let mut n = 4;
            while n > 0 && entry.times[n - 1] == 0.0 {
                n -= 1;
            }
            entry.subhashes[n] = subhash;
            entry.times[n] = 0.0;
            n
        };

        let counter: f64 = entry.times[n] as f64 + increment;
        if counter < 1.0 {
            entry.times[n] = counter as f32;
            false
        } else {
            self.reset(hash);
            true
        }
    }

    /// counter.py reset
    pub fn reset(&mut self, hash: u64) {
        let key = Self::_get_index(hash);
        let subhash = Self::_get_subhash(hash);
        if let Some(entry) = self.entries.get_mut(&key) {
            for i in 0..ASSOCIATIVITY {
                if entry.subhashes[i] == subhash {
                    entry.times[i] = 0.0;
                }
            }
        }
    }

    /// counter.py decay_all_counters — no-op for deterministic counter.
    pub fn decay_all_counters(&mut self) {}

    /// counter.py _clear_all
    pub fn _clear_all(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static DECAY_GENERATION_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn advance_decay_generation(intervals: usize) {
        DECAY_GENERATION.fetch_add(intervals, Ordering::Relaxed);
    }

    fn counter_time(counter: &JitCounter, hash: u64) -> f32 {
        let index = counter._get_index(hash);
        let subhash = JitCounter::_get_subhash(hash);
        let entry = &counter.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                return entry.times[i];
            }
        }
        0.0
    }

    #[test]
    fn test_basic_counting() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        assert!(!counter.tick(42, increment));
        assert!(!counter.tick(42, increment));
        assert!(counter.tick(42, increment));
    }

    #[test]
    fn test_different_hashes() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        let shift = counter.shift;
        let h1 = 1u64 << shift;
        let h2 = 2u64 << shift;
        assert!(!counter.tick(h1, increment));
        assert!(!counter.tick(h2, increment));
        assert!(!counter.tick(h1, increment));
        assert!(counter.tick(h1, increment));
        assert!(!counter.tick(h2, increment));
    }

    #[test]
    fn test_reset() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        let h = 1u64 << counter.shift;
        counter.tick(h, increment);
        counter.tick(h, increment);
        counter.reset(h);
        assert!(!counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
    }

    #[test]
    fn test_decay() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(10);
        let h = 1u64 << counter.shift;
        for _ in 0..8 {
            counter.tick(h, increment);
        }
        // default decay_by_mult = 1.0 (no decay). Set decay first.
        counter.set_decay(40); // decay_by_mult = 0.96
        // time ≈ 8 * (1/10) = 0.8, decay by 0.96 → 0.768
        counter.decay_all_counters();
        // Verify via a tick that doesn't fire (need ~0.232 more to reach 1.0)
        let index = counter._get_index(h);
        let subhash = JitCounter::_get_subhash(h);
        let entry = &counter.timetable[index];
        let mut time = 0.0f32;
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                time = entry.times[i];
                break;
            }
        }
        assert!(time > 0.7 && time < 0.8, "time={}", time);
    }

    #[test]
    fn test_tick_applies_every_pending_decay_generation() {
        let _generation_guard = DECAY_GENERATION_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        counter.set_decay(40);
        let h = 3u64 << counter.shift;
        counter.change_current_fraction(h, 0.5);

        advance_decay_generation(2);
        let increment = 0.001;
        assert!(!counter.tick(h, increment));

        // pypy__decay_jit_counters narrows the multiplier once and multiplies
        // in single precision; two elapsed intervals are two such multiplies.
        let mult = 0.96f64 as f32;
        let expected = ((0.5f32 * mult * mult) as f64 + increment) as f32;
        let actual = counter_time(&counter, h);
        assert!((actual - expected).abs() < 1.0e-6, "actual={actual}");
    }

    #[test]
    fn test_change_current_fraction_is_not_retro_decayed() {
        let _generation_guard = DECAY_GENERATION_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        counter.set_decay(40);
        let h = 4u64 << counter.shift;
        counter.change_current_fraction(h, 0.5);

        advance_decay_generation(1);
        counter.change_current_fraction(h, 0.98);
        let increment = 0.001;
        assert!(!counter.tick(h, increment));

        let expected = (0.98f32 as f64 + increment) as f32;
        let actual = counter_time(&counter, h);
        assert!((actual - expected).abs() < 1.0e-6, "actual={actual}");
    }

    #[test]
    fn test_auto_reset_on_fire() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        let h = 1u64 << counter.shift;
        assert!(!counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
    }

    #[test]
    fn test_fetch_next_hash() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let h1 = counter.fetch_next_hash();
        let h2 = counter.fetch_next_hash();
        assert_ne!(h1, h2);
        assert_ne!(counter._get_index(h1), counter._get_index(h2));
    }

    #[test]
    fn test_change_current_fraction() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(100);
        let h = 1u64 << counter.shift;
        counter.change_current_fraction(h, 0.98);
        // 0.98 + ~0.01 = ~0.99, not enough; two more ticks → ~1.0
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
    }

    #[test]
    fn test_size_parameter() {
        let counter = JitCounter::new(1024);
        assert_eq!(counter.size, 1024);
        // 0xFFFFFFFF >> shift = 1023 → shift = 22
        assert_eq!(counter.shift, 22);
    }
}
