/// counter.py: JitCounter — float-based 5-way associative timetable.
///
/// Direct port of rpython/jit/metainterp/counter.py.
/// Uses f32 time values (0.0 to 1.0) instead of integer counts.
/// tick(hash, increment) adds increment; fires when >= 1.0.
/// 5-way associative cache indexed by _get_index(hash), matched by
/// _get_subhash(hash). MRU promotion via _swap.

/// counter.py:82 DEFAULT_SIZE = 2048
const DEFAULT_SIZE: usize = 2048;
const DEFAULT_SIZE_MASK: usize = DEFAULT_SIZE - 1;

/// counter.py:12-13 ENTRY: 5 (f32 time, u16 subhash) pairs per bucket.
const ASSOCIATIVITY: usize = 5;

/// counter.py:84 shift = 16 for DEFAULT_SIZE = 2048 (2048 = 2^11, 32-11=21 → shift=21? No.)
/// RPython: hash32 >> shift == size-1. For size=2048: UINT32MAX >> shift = 2047.
/// 2^32 - 1 >> shift = 2047 → shift = 21. Actually for size=2048:
/// (0xFFFFFFFF >> 21) = 0x7FF = 2047 = 2048-1. So shift=21.
const SHIFT: u32 = 21;

/// One timetable entry: 5-way associative (time, subhash) pairs.
/// counter.py:11-13 ENTRY struct.
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

/// counter.py:16 JitCounter
pub struct JitCounter {
    /// counter.py:97 timetable
    timetable: Vec<Entry>,
    /// counter.py:103 celltable — compiled code hints per bucket.
    celltable: Vec<bool>,
    /// counter.py:100 _nexthash
    next_hash: u64,
    /// counter.py:264 decay_by_mult — f64 (Python float).
    decay_by_mult: f64,
    /// Pre-computed increment for the default threshold.
    /// counter.py:122-126 compute_threshold(threshold) — f64 (Python float).
    /// RPython: increment is computed as Python float (f64), used in tick()
    /// as f64 addition with f32→f64 promoted time value, stored back as f32.
    increment: f64,
    /// Original threshold value (for external queries).
    threshold: u32,
}

impl JitCounter {
    pub fn new(threshold: u32) -> Self {
        JitCounter {
            timetable: vec![Entry::default(); DEFAULT_SIZE],
            celltable: vec![false; DEFAULT_SIZE],
            next_hash: 1,
            decay_by_mult: 0.96_f64, // counter.py default decay=40 → 1.0-40*0.001=0.96
            increment: Self::compute_threshold_static(threshold),
            threshold,
        }
    }

    /// counter.py:122-126 compute_threshold
    /// Returns f64 (Python float) — same as RPython.
    pub fn compute_threshold_static(threshold: u32) -> f64 {
        if threshold <= 0 {
            return 0.0;
        }
        1.0_f64 / (threshold as f64 - 0.001)
    }

    /// counter.py:122-126 compute_threshold
    pub fn compute_threshold(&self, threshold: u32) -> f64 {
        Self::compute_threshold_static(threshold)
    }

    /// counter.py:128-136 _get_index
    #[inline(always)]
    fn get_index(hash: u64) -> usize {
        let hash32 = hash as u32;
        (hash32 >> SHIFT) as usize
    }

    /// counter.py:138-140 _get_subhash
    #[inline(always)]
    fn get_subhash(hash: u64) -> u16 {
        (hash & 0xFFFF) as u16
    }

    /// counter.py:142-153 fetch_next_hash
    pub fn fetch_next_hash(&mut self) -> u64 {
        let result = self.next_hash;
        // counter.py:151-152: increment by value that changes both
        // subhash and index portions.
        self.next_hash = result.wrapping_add(1 | (1u64 << SHIFT) | (1u64 << (SHIFT - 16)));
        result
    }

    /// counter.py:155-166 _swap
    #[inline(always)]
    fn swap(entry: &mut Entry, n: usize) -> usize {
        if entry.times[n] > entry.times[n + 1] {
            n + 1
        } else {
            entry.times.swap(n, n + 1);
            entry.subhashes.swap(n, n + 1);
            n
        }
    }

    /// counter.py:168-183 _tick_slowpath
    fn tick_slowpath(entry: &mut Entry, subhash: u16) -> usize {
        if entry.subhashes[1] == subhash {
            Self::swap(entry, 0)
        } else if entry.subhashes[2] == subhash {
            Self::swap(entry, 1)
        } else if entry.subhashes[3] == subhash {
            Self::swap(entry, 2)
        } else if entry.subhashes[4] == subhash {
            Self::swap(entry, 3)
        } else {
            // Not found. Find first empty slot from the end.
            let mut n = 4;
            while n > 0 && entry.times[n - 1] == 0.0 {
                n -= 1;
            }
            entry.subhashes[n] = subhash;
            entry.times[n] = 0.0;
            n
        }
    }

    /// Check if counter would fire without modifying state.
    pub fn would_fire(&self, hash: u64) -> bool {
        self.would_fire_with_increment(hash, self.increment)
    }

    /// Check if counter would fire against a specific threshold.
    pub fn would_fire_with_threshold(&self, hash: u64, threshold: u32) -> bool {
        self.would_fire_with_increment(hash, Self::compute_threshold_static(threshold))
    }

    pub fn would_fire_with_increment(&self, hash: u64, increment: f64) -> bool {
        let index = Self::get_index(hash);
        let subhash = Self::get_subhash(hash);
        let entry = &self.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                return entry.times[i] as f64 + increment >= 1.0;
            }
        }
        increment >= 1.0
    }

    /// counter.py:185-202 tick(hash, increment)
    ///
    /// Adds `increment` to the time value for `hash`.
    /// Returns true when the counter reaches 1.0 (auto-resets).
    #[inline(always)]
    pub fn tick(&mut self, hash: u64) -> bool {
        self.tick_with_increment(hash, self.increment)
    }

    /// counter.py:185-202 tick(hash, increment) with explicit increment.
    /// RPython: counter = float(p_entry.times[n]) + increment
    /// float() promotes f32→f64. increment is f64. Addition in f64.
    /// Store back: p_entry.times[n] = r_singlefloat(counter) (f64→f32).
    #[inline(always)]
    pub fn tick_with_increment(&mut self, hash: u64, increment: f64) -> bool {
        let index = Self::get_index(hash);
        let subhash = Self::get_subhash(hash);
        let entry = &mut self.timetable[index];

        let n = if entry.subhashes[0] == subhash {
            0
        } else {
            Self::tick_slowpath(entry, subhash)
        };

        // counter.py:194: counter = float(p_entry.times[n]) + increment
        let counter: f64 = entry.times[n] as f64 + increment;
        if counter < 1.0 {
            // counter.py:196: p_entry.times[n] = r_singlefloat(counter)
            entry.times[n] = counter as f32;
            false
        } else {
            // counter.py:199-200: self.reset(hash); return True
            self.reset(hash);
            true
        }
    }

    /// counter.py:185-202 tick with integer threshold (convenience).
    pub fn tick_with_threshold(&mut self, hash: u64, threshold: u32) -> bool {
        let increment = Self::compute_threshold_static(threshold);
        self.tick_with_increment(hash, increment)
    }

    /// counter.py:204-230 change_current_fraction(hash, new_fraction)
    pub fn change_current_fraction(&mut self, hash: u64, new_fraction: f64) {
        let index = Self::get_index(hash);
        let subhash = Self::get_subhash(hash);
        let entry = &mut self.timetable[index];

        // Find slot to overwrite: first matching subhash or empty time.
        let mut n = 0;
        while n < 4 && entry.subhashes[n] != subhash && entry.times[n] != 0.0 {
            n += 1;
        }
        // Shift right all elements [n-1, n-2, ..., 0]
        while n > 0 {
            n -= 1;
            entry.subhashes[n + 1] = entry.subhashes[n];
            entry.times[n + 1] = entry.times[n];
        }
        // Insert at position 0. r_singlefloat(new_fraction) = f64→f32.
        entry.subhashes[0] = subhash;
        entry.times[0] = new_fraction as f32;
    }

    /// counter.py:232-237 reset(hash)
    pub fn reset(&mut self, hash: u64) {
        let index = Self::get_index(hash);
        let subhash = Self::get_subhash(hash);
        let entry = &mut self.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                entry.times[i] = 0.0;
            }
        }
    }

    /// counter.py:258-264 set_decay(decay)
    pub fn set_decay(&mut self, decay: i32) {
        let clamped = decay.clamp(0, 1000);
        self.decay_by_mult = 1.0_f64 - (clamped as f64 * 0.001);
    }

    /// counter.py:266-278 decay_all_counters()
    /// RPython C code: each f32 time *= f64 decay_by_mult → f32.
    pub fn decay_all_counters(&mut self) {
        let mult = self.decay_by_mult;
        for entry in &mut self.timetable {
            for time in &mut entry.times {
                *time = (*time as f64 * mult) as f32;
            }
        }
    }

    /// Decay with explicit factor.
    pub fn decay_all_counters_by(&mut self, factor: f64) {
        for entry in &mut self.timetable {
            for time in &mut entry.times {
                *time = (*time as f64 * factor) as f32;
            }
        }
    }

    /// counter.py:239-240 lookup_chain(hash)
    /// O(1) celltable check.
    #[inline(always)]
    pub fn has_compiled_hint(&self, hash: u64) -> bool {
        self.celltable[Self::get_index(hash)]
    }

    /// counter.py:246-256 install_new_cell(hash, newcell)
    pub fn set_compiled_hint(&mut self, hash: u64, compiled: bool) {
        self.celltable[Self::get_index(hash)] = compiled;
    }

    /// counter.py:242-244 cleanup_chain(hash)
    pub fn cleanup_chain(&mut self, hash: u64) {
        self.reset(hash);
        self.set_compiled_hint(hash, false);
    }

    /// Get current time for a hash (for testing/introspection).
    pub fn get_time(&self, hash: u64) -> f32 {
        let index = Self::get_index(hash);
        let subhash = Self::get_subhash(hash);
        let entry = &self.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                return entry.times[i];
            }
        }
        0.0
    }

    /// Get the current count as integer (legacy compatibility).
    pub fn get(&self, hash: u64) -> u32 {
        let time = self.get_time(hash) as f64;
        (time / self.increment) as u32
    }

    /// Get the threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Set the threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
        self.increment = Self::compute_threshold_static(threshold);
    }

    /// Install a specific count for a hash (legacy compatibility).
    pub fn install(&mut self, hash: u64, count: u32) {
        let fraction = count as f64 * self.increment;
        self.change_current_fraction(hash, fraction.min(0.999));
    }

    /// Total number of active entries (non-zero times).
    pub fn num_active(&self) -> usize {
        self.timetable
            .iter()
            .flat_map(|e| e.times.iter())
            .filter(|&&t| t > 0.0)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_counting() {
        let mut counter = JitCounter::new(3);
        assert!(!counter.tick(42));
        assert!(!counter.tick(42));
        assert!(counter.tick(42)); // reaches threshold
    }

    #[test]
    fn test_different_hashes() {
        // Use hashes that map to different indices
        let h1 = 1u64 << SHIFT; // index=1, subhash=0
        let h2 = 2u64 << SHIFT; // index=2, subhash=0
        let mut counter = JitCounter::new(3);
        assert!(!counter.tick(h1));
        assert!(!counter.tick(h2));
        assert!(!counter.tick(h1));
        assert!(counter.tick(h1)); // h1 reaches threshold
        assert!(!counter.tick(h2)); // h2 hasn't yet
    }

    #[test]
    fn test_reset() {
        let h = 1u64 << SHIFT;
        let mut counter = JitCounter::new(3);
        counter.tick(h);
        counter.tick(h);
        counter.reset(h);
        assert!(!counter.tick(h)); // restarted from 0
        assert!(!counter.tick(h));
        assert!(counter.tick(h)); // now reaches threshold
    }

    #[test]
    fn test_decay() {
        let h = 1u64 << SHIFT;
        let mut counter = JitCounter::new(10);
        for _ in 0..8 {
            counter.tick(h);
        }
        // time ≈ 8 * (1/10) = 0.8, decay by 0.96 → 0.768
        counter.decay_all_counters();
        let time = counter.get_time(h);
        assert!(time > 0.7 && time < 0.8, "time={}", time);
    }

    #[test]
    fn test_auto_reset_on_fire() {
        let h = 1u64 << SHIFT;
        let mut counter = JitCounter::new(3);
        assert!(!counter.tick(h));
        assert!(!counter.tick(h));
        assert!(counter.tick(h)); // fires and auto-resets
        // After auto-reset, counter starts from 0 again
        assert!(!counter.tick(h));
        assert!(!counter.tick(h));
        assert!(counter.tick(h)); // fires again
    }
}

/// counter.py: DeterministicJitCounter for testing.
pub struct DeterministicJitCounter {
    counts: std::collections::HashMap<u64, f64>,
    threshold: u32,
    increment: f64,
}

impl DeterministicJitCounter {
    pub fn new(threshold: u32) -> Self {
        DeterministicJitCounter {
            counts: std::collections::HashMap::new(),
            threshold,
            increment: JitCounter::compute_threshold_static(threshold),
        }
    }

    pub fn tick(&mut self, hash: u64) -> bool {
        let count = self.counts.entry(hash).or_insert(0.0);
        *count += self.increment;
        if *count >= 1.0 {
            *count = 0.0;
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self, hash: u64) {
        self.counts.insert(hash, 0.0);
    }

    pub fn get(&self, hash: u64) -> u32 {
        let time = self.counts.get(&hash).copied().unwrap_or(0.0);
        (time / self.increment) as u32
    }

    pub fn threshold(&self) -> u32 {
        self.threshold
    }
}
