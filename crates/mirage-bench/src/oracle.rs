// SPDX-License-Identifier: Apache-2.0
use mirage_core::context::{RoutingContext, TokenContext};
use mirage_core::control::{ChainPhase, DifficultyClass, PrecisionTier};
use mirage_core::decision::{ActualRouting, ComputeDecision, RoutingPrediction};
use mirage_core::traits::{BranchPredictor, CognitiveProfiler};

/// Full-depth FP16 every token. No early exit, no precision adaptation.
/// This is the baseline the adaptive scheduler is measured against.
pub struct OracleProfiler { pub n_layers: u16 }

impl OracleProfiler {
    pub fn new(n_layers: u16) -> Self { OracleProfiler { n_layers } }
}

impl CognitiveProfiler for OracleProfiler {
    fn predict(&self, ctx: &TokenContext) -> ComputeDecision {
        ComputeDecision {
            difficulty: DifficultyClass::ReasoningCritical,
            target_depth: self.n_layers,
            precision_per_group: vec![PrecisionTier::FP16],
            head_budget: 64,
            burst: false,
            chain_phase: ctx.chain_phase,
        }
    }
}

/// Empty prediction → cold-cache execution every token. This isolates
/// the *plan* contribution from the *prefetch* contribution in the bench.
pub struct OraclePredictor;

impl BranchPredictor for OraclePredictor {
    fn predict(&self, _ctx: &RoutingContext) -> RoutingPrediction {
        RoutingPrediction::default()
    }
    fn update(&mut self, _predicted: &RoutingPrediction, _actual: &ActualRouting) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirage_core::control::TokenKind;
    fn ctx() -> TokenContext {
        TokenContext { token_id: 0, position: 0, recent_logit_margin: 1.0,
            recent_entropy: 0.2, token_kind: TokenKind::Content,
            chain_phase: ChainPhase::Middle }
    }
    #[test]
    fn oracle_profiler_returns_full_depth_fp16() {
        let p = OracleProfiler::new(48);
        let d = p.predict(&ctx());
        assert_eq!(d.target_depth, 48);
        assert_eq!(d.precision_per_group, vec![PrecisionTier::FP16]);
    }
    #[test]
    fn oracle_predictor_is_empty() {
        let p = OraclePredictor;
        let pred = p.predict(&RoutingContext { position: 5, recent_experts: vec![1, 2, 3] });
        assert!(pred.experts_per_moe_layer.is_empty());
        assert_eq!(pred.likely_exit_layer, 0);
    }
}
