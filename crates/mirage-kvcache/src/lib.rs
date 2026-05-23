// SPDX-License-Identifier: Apache-2.0
//! KV compression policy — pure decision logic (spec §8.3). No GPU.
use mirage_core::control::PrecisionTier;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvBlock { pub position: u32, pub precision: PrecisionTier, pub generation: u32 }

pub struct KvCachePolicy { pub demote_age: u32 }

impl KvCachePolicy {
    pub fn new(demote_age: u32) -> Self { KvCachePolicy { demote_age } }

    /// Age-based demotion: blocks older than `demote_age` step down one tier.
    /// Invariant I4: never lose the generation counter; never go below Q2.
    pub fn demote(&self, block: KvBlock, current_pos: u32) -> KvBlock {
        if current_pos.saturating_sub(block.position) <= self.demote_age {
            return block;
        }
        let precision = match block.precision {
            PrecisionTier::FP16 => PrecisionTier::Q6,
            PrecisionTier::FP8 => PrecisionTier::Q4,
            PrecisionTier::Q6 => PrecisionTier::Q4,
            PrecisionTier::Q4 => PrecisionTier::Q2,
            PrecisionTier::Q2 => PrecisionTier::Q2,
        };
        KvBlock { precision, generation: block.generation + 1, ..block }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recent_block_is_not_demoted() {
        let pol = KvCachePolicy::new(64);
        let b = KvBlock { position: 100, precision: PrecisionTier::FP16, generation: 0 };
        assert_eq!(pol.demote(b, 120), b);
    }
    #[test]
    fn old_block_steps_down_and_bumps_generation() {
        let pol = KvCachePolicy::new(64);
        let b = KvBlock { position: 0, precision: PrecisionTier::FP16, generation: 0 };
        let d = pol.demote(b, 1000);
        assert_eq!(d.precision, PrecisionTier::Q6);
        assert_eq!(d.generation, 1);
    }
    #[test]
    fn q2_is_the_floor() {
        let pol = KvCachePolicy::new(0);
        let b = KvBlock { position: 0, precision: PrecisionTier::Q2, generation: 5 };
        assert_eq!(pol.demote(b, 1000).precision, PrecisionTier::Q2);
    }
}
