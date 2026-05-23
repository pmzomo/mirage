// SPDX-License-Identifier: Apache-2.0
use std::collections::HashSet;

/// Fixed cost model (microseconds / bytes). Deterministic by construction.
pub const EXPERT_BYTES: u64 = 1_500_000; // ~1.5 MB per expert at Q4 (synthetic)
pub const PCIE_US_PER_MB: u32 = 31;      // ~32 GB/s
pub const LAYER_COMPUTE_US: u32 = 120;
pub const STALL_US_PER_MISS: u32 = 1_900;

pub struct FakeVram {
    budget_bytes: u64,
    resident: HashSet<u16>,
    per_expert: u64,
}

impl FakeVram {
    pub fn new(budget_mb: u32) -> Self {
        FakeVram { budget_bytes: budget_mb as u64 * 1_000_000,
                   resident: HashSet::new(), per_expert: EXPERT_BYTES }
    }
    pub fn used_bytes(&self) -> u64 { self.resident.len() as u64 * self.per_expert }
    pub fn used_mb(&self) -> u32 { (self.used_bytes() / 1_000_000) as u32 }

    /// Make `experts` resident. Returns the count that were NOT already
    /// resident (i.e. misses requiring a PCIe transfer). Invariant I7:
    /// never exceed the budget — evict in deterministic id order first.
    pub fn ensure_resident(&mut self, experts: &[u16]) -> u32 {
        let misses = experts.iter().filter(|e| !self.resident.contains(e)).count() as u32;
        for &e in experts { self.resident.insert(e); }
        while self.used_bytes() > self.budget_bytes {
            let victim = self.resident.iter()
                .filter(|e| !experts.contains(e))
                .min()
                .copied()
                .or_else(|| self.resident.iter().min().copied())
                .expect("residency non-empty when over budget");
            self.resident.remove(&victim);
        }
        assert!(self.used_bytes() <= self.budget_bytes, "I7: residency budget exceeded");
        misses
    }
}

pub struct LatencyModel;

impl LatencyModel {
    pub fn transfer_us(misses: u32) -> u32 {
        let mb = (misses as u64 * EXPERT_BYTES / 1_000_000) as u32;
        mb * PCIE_US_PER_MB
    }
    pub fn stall_us(misses: u32) -> u32 { misses * STALL_US_PER_MISS }
    pub fn compute_us(layers: u16) -> u32 { layers as u32 * LAYER_COMPUTE_US }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn second_residency_has_no_misses() {
        let mut v = FakeVram::new(12_000);
        assert_eq!(v.ensure_resident(&[1, 2, 3]), 3);
        assert_eq!(v.ensure_resident(&[1, 2, 3]), 0);
    }
    #[test]
    fn budget_is_never_exceeded() {
        let mut v = FakeVram::new(3); // 3 MB => only 2 experts fit
        v.ensure_resident(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(v.used_bytes() <= 3_000_000);
    }
}
