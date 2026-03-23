/// Hot counter for detecting frequently-executed loops.
///
/// Translated from rpython/jit/metainterp/counter.py.
/// Uses a hash table to track execution counts per green key (bytecode position).
/// When a counter reaches the threshold, JIT compilation is triggered.

/// Size of the counter table (must be a power of 2).
const TABLE_SIZE: usize = 1 << 14; // 16384 entries
const TABLE_MASK: usize = TABLE_SIZE - 1;

/// Number of entries to check in each hash probe (5-way associative).
const ASSOCIATIVITY: usize = 5;

/// A hot counter using a hash table with linear probing.
pub struct JitCounter {
    /// Counter table: pairs of (hash, count).
    table: Vec<(u64, u32)>,
    /// Threshold for triggering compilation.
    threshold: u32,
    /// counter.py: _nexthash — monotonically increasing hash generator.
    next_hash: u64,
    /// counter.py: decay_by_mult — multiplier for decay (1.0 = no decay).
    decay_mult: f64,
}

impl JitCounter {
    pub fn new(threshold: u32) -> Self {
        JitCounter {
            table: vec![(0, 0); TABLE_SIZE],
            threshold,
            next_hash: 0,
            decay_mult: 1.0,
        }
    }

    /// Check if counter would fire without modifying state.
    #[inline]
    pub fn would_fire(&self, hash: u64) -> bool {
        if self.threshold == 0 {
            return true;
        }
        let base = (hash as usize) & TABLE_MASK;
        for i in 0..ASSOCIATIVITY {
            let idx = (base + i) & TABLE_MASK;
            if self.table[idx].0 == hash {
                return self.table[idx].1 + 1 >= self.threshold;
            }
        }
        1 >= self.threshold
    }

    /// Increment the counter for the given hash. Returns true if threshold is reached.
    /// RPython counter.py tick: always_inline for the fast path (slot 0 hit).
    #[inline(always)]
    pub fn tick(&mut self, hash: u64) -> bool {
        let base = (hash as usize) & TABLE_MASK;

        // RPython counter.py tick / _tick_slowpath parity:
        // 5-way associative lookup with MRU swap to slot 0.
        // Found at slot 0 (fast path):
        if self.table[base].0 == hash {
            self.table[base].1 += 1;
            return self.table[base].1 >= self.threshold;
        }
        // Slowpath: check slots 1..ASSOCIATIVITY
        // RPython counter.py _tick_slowpath + _swap parity.
        for i in 1..ASSOCIATIVITY {
            let idx = (base + i) & TABLE_MASK;
            if self.table[idx].0 == hash {
                self.table[idx].1 += 1;
                let fired = self.table[idx].1 >= self.threshold;
                // RPython _swap(p_entry, n): swap with slot n only if
                // slot n has a lower count (conditional promotion).
                let prev_idx = (base + i - 1) & TABLE_MASK;
                if self.table[prev_idx].1 <= self.table[idx].1 {
                    self.table.swap(idx, prev_idx);
                }
                return fired;
            }
        }

        // Not found — RPython: find first empty slot (count == 0),
        // or use the last slot (n=4). Existing entries are preserved.
        let mut insert_idx = (base + ASSOCIATIVITY - 1) & TABLE_MASK;
        for i in (0..ASSOCIATIVITY).rev() {
            let idx = (base + i) & TABLE_MASK;
            if self.table[idx].1 == 0 {
                insert_idx = idx;
                break;
            }
        }
        self.table[insert_idx] = (hash, 1);
        false
    }

    /// Reset the counter for the given hash.
    pub fn reset(&mut self, hash: u64) {
        let base = (hash as usize) & TABLE_MASK;
        for i in 0..ASSOCIATIVITY {
            let idx = (base + i) & TABLE_MASK;
            if self.table[idx].0 == hash {
                self.table[idx].1 = 0;
                return;
            }
        }
    }

    /// Evict the counter entry for the given hash.
    /// Sets hash and count to 0, so the key must re-accumulate from zero
    /// in a (possibly different) slot.
    pub fn evict(&mut self, hash: u64) {
        let base = (hash as usize) & TABLE_MASK;
        for i in 0..ASSOCIATIVITY {
            let idx = (base + i) & TABLE_MASK;
            if self.table[idx].0 == hash {
                self.table[idx] = (0, 0);
                return;
            }
        }
    }

    /// Decay all counters by the given factor (0.0 = clear, 1.0 = keep).
    /// counter.py:266 decay_all_counters.
    /// RPython default decay=40, so decay_by_mult = 1.0 - 40*0.001 = 0.96.
    /// Each call reduces all counters by 4%.
    pub fn decay_all_counters(&mut self) {
        self.decay_all_counters_by(0.96);
    }

    /// Decay all counters by a multiplicative factor.
    /// counter.py: set_decay(decay) — accepts 0..1000, computes per-call multiplier.
    pub fn decay_all_counters_by(&mut self, factor: f64) {
        for entry in &mut self.table {
            entry.1 = (entry.1 as f64 * factor) as u32;
        }
    }

    /// Get the current count for a hash (for testing/introspection).
    /// counter.py: get() — returns the current count.
    pub fn get(&self, hash: u64) -> u32 {
        let base = (hash as usize) & TABLE_MASK;
        for i in 0..ASSOCIATIVITY {
            let idx = (base + i) & TABLE_MASK;
            if self.table[idx].0 == hash {
                return self.table[idx].1;
            }
        }
        0
    }

    /// Install a specific count for a hash (for bridge eagerness).
    /// counter.py: install_new_cell(hash, count)
    pub fn install(&mut self, hash: u64, count: u32) {
        let base = (hash as usize) & TABLE_MASK;
        // Find existing or lowest entry
        for i in 0..ASSOCIATIVITY {
            let idx = (base + i) & TABLE_MASK;
            if self.table[idx].0 == hash {
                self.table[idx].1 = count;
                return;
            }
        }
        // Not found — evict lowest
        let mut min_idx = base;
        let mut min_count = self.table[base].1;
        for i in 1..ASSOCIATIVITY {
            let idx = (base + i) & TABLE_MASK;
            if self.table[idx].1 < min_count {
                min_count = self.table[idx].1;
                min_idx = idx;
            }
        }
        self.table[min_idx] = (hash, count);
    }

    /// Compute the threshold for bridge compilation based on guard eagerness.
    /// counter.py: compute_threshold(scale)
    /// Returns `threshold * scale` capped at threshold.
    pub fn compute_threshold(&self, scale: f64) -> u32 {
        let result = (self.threshold as f64 * scale) as u32;
        result.min(self.threshold)
    }

    /// counter.py: fetch_next_hash()
    /// Generate a unique hash for a new green key.
    /// Each call returns a different hash to avoid collisions.
    pub fn fetch_next_hash(&mut self) -> u64 {
        let result = self.next_hash;
        // Increment by a value that changes both the index portion
        // and the sub-hash portion of the hash, avoiding patterns.
        self.next_hash = result.wrapping_add(1 | (1 << 16) | (1 << 32));
        result
    }

    /// counter.py: change_current_fraction(hash, new_fraction)
    /// Set the counter for a hash to a fraction of the threshold.
    /// Used for bridge compilation eagerness: a bridge that fails
    /// frequently gets its counter boosted closer to the threshold.
    pub fn change_current_fraction(&mut self, hash: u64, fraction: f64) {
        let new_count = (self.threshold as f64 * fraction) as u32;
        self.install(hash, new_count);
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Set the threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
    }

    /// Total number of active entries (non-zero counts).
    pub fn num_active(&self) -> usize {
        self.table.iter().filter(|(_, c)| *c > 0).count()
    }

    /// counter.py: set_decay(decay)
    /// Set the decay factor from 0 (none) to 1000 (max).
    /// The decay multiplier is `1.0 - decay * 0.001`.
    pub fn set_decay(&mut self, decay: i32) {
        let clamped = decay.clamp(0, 1000);
        self.decay_mult = 1.0 - (clamped as f64 * 0.001);
    }

    /// counter.py: lookup_chain(hash)
    /// Look up the JitCell chain head for this hash bucket.
    /// Returns the current count (simplified — no cell chain yet).
    pub fn lookup_chain(&self, hash: u64) -> u32 {
        self.get(hash)
    }

    /// counter.py: cleanup_chain(hash)
    /// Reset counter and clean up the cell chain for this hash.
    pub fn cleanup_chain(&mut self, hash: u64) {
        self.reset(hash);
    }

    /// counter.py: install_new_cell(hash, newcell)
    ///
    /// Walk the linked list at celltable[index], remove dead cells
    /// (should_remove_jitcell), and prepend the new cell.
    /// In our implementation, this is a no-op since we use WarmEnterState's
    /// HashMap<u64, BaseJitCell> instead of a per-bucket chain.
    /// The method exists for API parity.
    pub fn install_new_cell(&mut self, _hash: u64) {
        // In RPython, celltable is a parallel array of JitCell linked lists.
        // In majit, WarmEnterState.cells HashMap handles this directly.
    }

    /// counter.py: _get_index(hash) — hash to bucket index.
    fn get_index(&self, hash: u64) -> usize {
        (hash as usize) & TABLE_MASK
    }

    /// counter.py: _get_subhash(hash) — lower 16 bits for associative match.
    fn get_subhash(hash: u64) -> u16 {
        (hash & 0xFFFF) as u16
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
        let mut counter = JitCounter::new(3);
        assert!(!counter.tick(1));
        assert!(!counter.tick(2));
        assert!(!counter.tick(1));
        assert!(counter.tick(1)); // hash 1 reaches threshold
        assert!(!counter.tick(2)); // hash 2 hasn't yet
    }

    #[test]
    fn test_reset() {
        let mut counter = JitCounter::new(3);
        counter.tick(42);
        counter.tick(42);
        counter.reset(42);
        assert!(!counter.tick(42)); // restarted from 1
        assert!(!counter.tick(42));
        assert!(counter.tick(42)); // now reaches threshold
    }

    #[test]
    fn test_decay() {
        let mut counter = JitCounter::new(10);
        for _ in 0..8 {
            counter.tick(42);
        }
        // count=8, decay by 0.96 → floor(8*0.96)=7
        counter.decay_all_counters();
        assert_eq!(counter.get(42), 7);
        // Need 3 more to reach 10
        assert!(!counter.tick(42)); // 8
        assert!(!counter.tick(42)); // 9
        assert!(counter.tick(42)); // 10
    }
}

/// counter.py: DeterministicJitCounter — deterministic counter for testing.
///
/// Uses a HashMap instead of a lossy hash table, so no eviction occurs.
/// This makes test behavior fully deterministic regardless of hash collisions.
pub struct DeterministicJitCounter {
    counts: std::collections::HashMap<u64, u32>,
    threshold: u32,
}

impl DeterministicJitCounter {
    pub fn new(threshold: u32) -> Self {
        DeterministicJitCounter {
            counts: std::collections::HashMap::new(),
            threshold,
        }
    }

    /// Increment and check threshold. Deterministic: no eviction.
    pub fn tick(&mut self, hash: u64) -> bool {
        let count = self.counts.entry(hash).or_insert(0);
        *count += 1;
        *count >= self.threshold
    }

    /// Reset the counter for a hash.
    pub fn reset(&mut self, hash: u64) {
        self.counts.insert(hash, 0);
    }

    /// Get the current count.
    pub fn get(&self, hash: u64) -> u32 {
        self.counts.get(&hash).copied().unwrap_or(0)
    }

    /// Check without modifying.
    #[inline]
    pub fn would_fire(&self, hash: u64) -> bool {
        let count = self.counts.get(&hash).copied().unwrap_or(0);
        count + 1 >= self.threshold
    }

    /// Threshold getter.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }
}

#[cfg(test)]
mod deterministic_tests {
    use super::*;

    #[test]
    fn test_deterministic_basic() {
        let mut counter = DeterministicJitCounter::new(3);
        assert!(!counter.tick(42));
        assert!(!counter.tick(42));
        assert!(counter.tick(42));
    }

    #[test]
    fn test_deterministic_no_eviction() {
        let mut counter = DeterministicJitCounter::new(3);
        // Insert many different hashes — no eviction
        for i in 0..100u64 {
            counter.tick(i);
        }
        // All should still be tracked at count 1
        for i in 0..100u64 {
            assert_eq!(counter.get(i), 1);
        }
    }

    #[test]
    fn test_deterministic_would_fire() {
        let mut counter = DeterministicJitCounter::new(3);
        counter.tick(42);
        assert!(!counter.would_fire(42)); // count=1, need 2 more
        counter.tick(42);
        assert!(counter.would_fire(42)); // count=2, one more tick fires
    }

    #[test]
    fn test_deterministic_reset() {
        let mut counter = DeterministicJitCounter::new(2);
        counter.tick(42);
        counter.reset(42);
        assert_eq!(counter.get(42), 0);
        assert!(!counter.tick(42)); // count=1, not yet
        assert!(counter.tick(42)); // count=2, fires
    }

    #[test]
    fn test_jit_counter_install() {
        let mut counter = JitCounter::new(10);
        counter.install(42, 8);
        assert_eq!(counter.get(42), 8);
        assert!(!counter.tick(42)); // 9, not yet
        assert!(counter.tick(42)); // 10, fires
    }

    #[test]
    fn test_jit_counter_compute_threshold() {
        let counter = JitCounter::new(100);
        assert_eq!(counter.compute_threshold(0.5), 50);
        assert_eq!(counter.compute_threshold(1.0), 100);
        assert_eq!(counter.compute_threshold(2.0), 100); // capped
    }

    #[test]
    fn test_jit_counter_num_active() {
        let mut counter = JitCounter::new(10);
        assert_eq!(counter.num_active(), 0);
        counter.tick(1);
        counter.tick(2);
        assert_eq!(counter.num_active(), 2);
    }
}
