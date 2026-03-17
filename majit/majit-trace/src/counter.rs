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

    /// Decay all counters (halve them). Called periodically.
    pub fn decay_all_counters(&mut self) {
        for entry in &mut self.table {
            entry.1 /= 2;
        }
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Set the threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
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
