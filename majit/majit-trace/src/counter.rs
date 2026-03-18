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
}

impl JitCounter {
    pub fn new(threshold: u32) -> Self {
        JitCounter {
            table: vec![(0, 0); TABLE_SIZE],
            threshold,
        }
    }

    /// Check if counter would fire without modifying state.
    pub fn would_fire(&self, hash: u64) -> bool {
        if self.threshold == 0 { return true; }
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
    pub fn tick(&mut self, hash: u64) -> bool {
        let base = (hash as usize) & TABLE_MASK;

        // 5-way associative lookup
        for i in 0..ASSOCIATIVITY {
            let idx = (base + i) & TABLE_MASK;
            if self.table[idx].0 == hash {
                self.table[idx].1 += 1;
                return self.table[idx].1 >= self.threshold;
            }
        }

        // Not found — find the entry with the lowest count to evict
        let mut min_idx = base;
        let mut min_count = self.table[base].1;
        for i in 1..ASSOCIATIVITY {
            let idx = (base + i) & TABLE_MASK;
            if self.table[idx].1 < min_count {
                min_count = self.table[idx].1;
                min_idx = idx;
            }
        }

        // Evict and insert
        self.table[min_idx] = (hash, 1);
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

    /// Decay all counters by the given factor (0.0 = clear, 1.0 = keep).
    /// counter.py: decay_all_counters with configurable decay factor.
    /// For backwards compatibility, the default halves all counters.
    pub fn decay_all_counters(&mut self) {
        self.decay_all_counters_by(0.5);
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
        counter.decay_all_counters();
        // Count should be halved (8 -> 4), so need 6 more to reach 10
        for _ in 0..5 {
            assert!(!counter.tick(42));
        }
        assert!(counter.tick(42)); // 4 + 6 = 10
    }
}
