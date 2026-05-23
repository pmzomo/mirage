// SPDX-License-Identifier: Apache-2.0
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionTier {
    Q2,
    Q4,
    Q6,
    FP8,
    FP16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChainPhase {
    NotReasoning,
    Early,
    Middle,
    Ending,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifficultyClass {
    Trivial,
    Normal,
    ReasoningCritical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Gpu,
    Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenKind {
    Punctuation,
    Formatting,
    CommonWord,
    Content,
    ReasoningMarker,
}

/// Bitset of active attention heads, one bit per head (max 64 heads/layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadMask(pub u64);

impl HeadMask {
    pub fn all(n_heads: u16) -> Self {
        debug_assert!(n_heads <= 64);
        HeadMask(if n_heads == 64 {
            u64::MAX
        } else {
            (1u64 << n_heads) - 1
        })
    }
    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }
    pub fn is_set(&self, head: u16) -> bool {
        self.0 & (1u64 << head) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn head_mask_all_counts_correctly() {
        assert_eq!(HeadMask::all(32).count(), 32);
        assert_eq!(HeadMask::all(64).count(), 64);
        assert!(HeadMask::all(8).is_set(7));
        assert!(!HeadMask::all(8).is_set(8));
    }
}
