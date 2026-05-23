// SPDX-License-Identifier: Apache-2.0
//! Phase 1 n-gram branch predictor (spec §9). Prefetch-mode only.
use mirage_core::context::RoutingContext;
use mirage_core::decision::{ActualRouting, RoutingPrediction};
use mirage_core::traits::BranchPredictor;
use std::collections::HashMap;

/// Maps a recent-expert signature -> the expert set that followed it.
pub struct NgramBranchPredictor {
    table: HashMap<u64, Vec<Vec<u16>>>,
    order: usize,
}

impl NgramBranchPredictor {
    pub fn new(order: usize) -> Self {
        NgramBranchPredictor {
            table: HashMap::new(),
            order,
        }
    }
    fn sig(&self, recent: &[u16]) -> u64 {
        let take = recent.len().saturating_sub(self.order);
        let mut h = 1469598103934665603u64;
        for &e in &recent[take..] {
            h ^= e as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h
    }
}

impl BranchPredictor for NgramBranchPredictor {
    fn predict(&self, ctx: &RoutingContext) -> RoutingPrediction {
        let experts = self
            .table
            .get(&self.sig(&ctx.recent_experts))
            .cloned()
            .unwrap_or_default();
        RoutingPrediction {
            experts_per_moe_layer: experts,
            likely_exit_layer: 0,
        }
    }
    fn update(&mut self, _predicted: &RoutingPrediction, actual: &ActualRouting) {
        // Index the flattened actual experts by their own leading signature so
        // a repeated context predicts the same continuation next time.
        let flat: Vec<u16> = actual
            .experts_per_moe_layer
            .iter()
            .flatten()
            .copied()
            .collect();
        let sig = self.sig(&flat);
        self.table.insert(sig, actual.experts_per_moe_layer.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learns_a_repeated_routing_pattern() {
        let mut p = NgramBranchPredictor::new(4);
        let actual = ActualRouting {
            experts_per_moe_layer: vec![vec![3, 7, 9, 1]],
            exit_layer: 1,
        };
        p.update(&RoutingPrediction::default(), &actual);
        let flat = vec![3u16, 7, 9, 1];
        let pred = p.predict(&RoutingContext {
            position: 1,
            recent_experts: flat,
        });
        assert_eq!(pred.experts_per_moe_layer, vec![vec![3, 7, 9, 1]]);
    }
    #[test]
    fn unknown_context_predicts_empty_not_panic() {
        let p = NgramBranchPredictor::new(4);
        let pred = p.predict(&RoutingContext {
            position: 0,
            recent_experts: vec![],
        });
        assert!(pred.experts_per_moe_layer.is_empty());
    }
}
