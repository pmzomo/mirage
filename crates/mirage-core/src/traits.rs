// SPDX-License-Identifier: Apache-2.0
use crate::context::{ExecOutcome, RoutingContext, TokenContext};
use crate::decision::{ActualRouting, ComputeDecision, RoutingPrediction};
use crate::plan::{ExecPlan, ModelShape};

pub trait CognitiveProfiler {
    fn predict(&self, ctx: &TokenContext) -> ComputeDecision;
}

pub trait BranchPredictor {
    fn predict(&self, ctx: &RoutingContext) -> RoutingPrediction;
    fn update(&mut self, predicted: &RoutingPrediction, actual: &ActualRouting);
}

pub trait ExecutionBackend {
    fn execute(&mut self, plan: &ExecPlan, shape: &ModelShape) -> ExecOutcome;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::*;

    struct NullProfiler;
    impl CognitiveProfiler for NullProfiler {
        fn predict(&self, _: &TokenContext) -> ComputeDecision {
            ComputeDecision {
                difficulty: DifficultyClass::Normal,
                target_depth: 1,
                precision_per_group: vec![PrecisionTier::Q4],
                head_budget: 32,
                burst: false,
                chain_phase: ChainPhase::NotReasoning,
            }
        }
    }

    #[test]
    fn trait_object_is_usable() {
        let p: Box<dyn CognitiveProfiler> = Box::new(NullProfiler);
        let ctx = TokenContext {
            token_id: 1, position: 0, recent_logit_margin: 1.0,
            recent_entropy: 0.1, token_kind: TokenKind::Content,
            chain_phase: ChainPhase::NotReasoning,
        };
        assert_eq!(p.predict(&ctx).target_depth, 1);
    }
}
