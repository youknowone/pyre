//! Memory management for compiled loops.
//!
//! Mirrors RPython's `memmgr.py` MemoryManager: generation-based loop aging
//! to evict loops that haven't been accessed recently.

use std::collections::HashMap;

/// Generation-based loop aging. Loops not accessed for `max_age`
/// generations are candidates for eviction.
///
/// Reference: rpython/jit/metainterp/memmgr.py MemoryManager.
pub struct LoopAging {
    generation: u64,
    max_age: u64,
    /// loop_key → last access generation.
    loop_generations: HashMap<u64, u64>,
}

impl LoopAging {
    /// Create a new LoopAging with the given max_age.
    /// `max_age == 0` disables eviction.
    pub fn new(max_age: u64) -> Self {
        LoopAging {
            generation: 0,
            max_age,
            loop_generations: HashMap::new(),
        }
    }

    /// Set the maximum age before eviction.
    pub fn set_max_age(&mut self, max_age: u64) {
        self.max_age = max_age;
    }

    /// Get the current max_age setting.
    pub fn max_age(&self) -> u64 {
        self.max_age
    }

    /// Mark a loop as alive in the current generation.
    pub fn keep_loop_alive(&mut self, loop_key: u64) {
        self.loop_generations.insert(loop_key, self.generation);
    }

    /// Register a new loop at the current generation.
    pub fn register_loop(&mut self, loop_key: u64) {
        self.loop_generations.insert(loop_key, self.generation);
    }

    /// Advance the generation counter. Returns the set of loop keys
    /// that are now too old and should be evicted.
    pub fn next_generation(&mut self) -> Vec<u64> {
        self.generation += 1;

        if self.max_age == 0 {
            return vec![];
        }

        let threshold = self.generation.saturating_sub(self.max_age);
        let evicted: Vec<u64> = self
            .loop_generations
            .iter()
            .filter(|&(_, &last_access)| last_access < threshold)
            .map(|(&key, _)| key)
            .collect();

        for key in &evicted {
            self.loop_generations.remove(key);
        }

        evicted
    }

    /// Get the current generation number.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of tracked loops.
    pub fn alive_count(&self) -> usize {
        self.loop_generations.len()
    }

    /// Release all tracked loops. After this, no loops are tracked.
    ///
    /// Reference: memmgr.py MemoryManager.release_all_loops
    pub fn release_all_loops(&mut self) {
        self.loop_generations.clear();
    }
}
