// SPDX-License-Identifier: Apache-2.0
//! Counterfactual replay (spec §16): re-feed recorded routing contexts to a
//! candidate predictor and measure its hit rate WITHOUT re-running the GPU.
use mirage_core::context::RoutingContext;
use mirage_core::decision::ActualRouting;
use mirage_core::traits::BranchPredictor;
use mirage_telemetry::TokenTrace;

pub struct ReplayResult {
    pub tokens: usize,
    pub hits: usize,
}
impl ReplayResult {
    pub fn hit_rate(&self) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            self.hits as f64 / self.tokens as f64
        }
    }
}

/// Replay recorded traces against a fresh predictor. A "hit" = predicted
/// expert set equals the recorded actual expert set for that token.
pub fn replay<B: BranchPredictor>(traces: &[TokenTrace], pred: &mut B) -> ReplayResult {
    let mut hits = 0;
    for t in traces {
        let rctx = RoutingContext {
            position: t.position,
            recent_experts: t.predicted_experts.clone(),
        };
        let predicted = pred.predict(&rctx);
        let flat: Vec<u16> = predicted
            .experts_per_moe_layer
            .iter()
            .flatten()
            .copied()
            .collect();
        if flat == t.actual_experts {
            hits += 1;
        }
        pred.update(
            &predicted,
            &ActualRouting {
                experts_per_moe_layer: vec![t.actual_experts.clone()],
                exit_layer: 0,
            },
        );
    }
    ReplayResult {
        tokens: traces.len(),
        hits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirage_predictor::NgramBranchPredictor;
    use mirage_telemetry::tests_support::sample_trace;

    #[test]
    fn replay_counts_all_tokens() {
        let traces: Vec<TokenTrace> = (0..10).map(sample_trace).collect();
        let mut p = NgramBranchPredictor::new(4);
        let r = replay(&traces, &mut p);
        assert_eq!(r.tokens, 10);
        assert!(r.hit_rate() >= 0.0 && r.hit_rate() <= 1.0);
    }
}
