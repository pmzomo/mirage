// SPDX-License-Identifier: Apache-2.0
//! Phase 1 hand-tuned profiler (spec §7 L3, §13).
use mirage_core::control::{ChainPhase, DifficultyClass, PrecisionTier, TokenKind};
use mirage_core::context::TokenContext;
use mirage_core::decision::ComputeDecision;
use mirage_core::traits::CognitiveProfiler;

pub struct HeuristicProfiler { pub n_layers: u16 }

impl HeuristicProfiler {
    pub fn new(n_layers: u16) -> Self { HeuristicProfiler { n_layers } }
}

impl CognitiveProfiler for HeuristicProfiler {
    fn predict(&self, ctx: &TokenContext) -> ComputeDecision {
        // Trivial tokens: shallow + low precision. Reasoning: full depth + FP16.
        let trivial = matches!(ctx.token_kind, TokenKind::Punctuation | TokenKind::Formatting)
            && ctx.recent_logit_margin > 3.0;
        let reasoning = matches!(ctx.token_kind, TokenKind::ReasoningMarker)
            || ctx.chain_phase == ChainPhase::Middle
            || ctx.recent_logit_margin < 0.5;

        let (difficulty, depth, tier, burst) = if trivial {
            (DifficultyClass::Trivial, self.n_layers / 3, PrecisionTier::Q2, false)
        } else if reasoning {
            (DifficultyClass::ReasoningCritical, self.n_layers, PrecisionTier::FP16, true)
        } else {
            (DifficultyClass::Normal, self.n_layers, PrecisionTier::Q4, false)
        };

        ComputeDecision {
            difficulty,
            target_depth: depth.max(1),
            precision_per_group: vec![tier],
            head_budget: 32,
            burst,
            chain_phase: ctx.chain_phase,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ctx(kind: TokenKind, margin: f32, phase: ChainPhase) -> TokenContext {
        TokenContext { token_id: 1, position: 0, recent_logit_margin: margin,
            recent_entropy: 0.2, token_kind: kind, chain_phase: phase }
    }
    #[test]
    fn trivial_token_runs_shallow_and_low_precision() {
        let p = HeuristicProfiler::new(48);
        let d = p.predict(&ctx(TokenKind::Punctuation, 5.0, ChainPhase::NotReasoning));
        assert_eq!(d.difficulty, DifficultyClass::Trivial);
        assert!(d.target_depth < 48);
        assert_eq!(d.precision_per_group, vec![PrecisionTier::Q2]);
    }
    #[test]
    fn reasoning_token_runs_full_depth_fp16_burst() {
        let p = HeuristicProfiler::new(48);
        let d = p.predict(&ctx(TokenKind::ReasoningMarker, 2.0, ChainPhase::Middle));
        assert_eq!(d.difficulty, DifficultyClass::ReasoningCritical);
        assert_eq!(d.target_depth, 48);
        assert!(d.burst);
    }
}
