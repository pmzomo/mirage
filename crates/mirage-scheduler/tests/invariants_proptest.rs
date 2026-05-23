// SPDX-License-Identifier: Apache-2.0
//! System Invariants as property tests (spec §23).
use mirage_core::context::{RoutingContext, TokenContext};
use mirage_core::control::{ChainPhase, TokenKind};
use mirage_core::plan::ModelShape;
use mirage_predictor::NgramBranchPredictor;
use mirage_profiler::HeuristicProfiler;
use mirage_scheduler::{Anomaly, FallbackController, Scheduler};
use mirage_sim::backend::SimBackend;
use mirage_telemetry::validate;
use proptest::prelude::*;

proptest! {
    // I8: every emitted trace validates against its schema version.
    // I7: residency budget never exceeded (SimBackend asserts internally).
    #[test]
    fn i7_i8_traces_valid_and_budget_respected(
        positions in prop::collection::vec(0u32..5000, 1..120),
        margin in 0.0f32..6.0,
    ) {
        let shape = ModelShape::qwen3_30b_a3b();
        let mut sched = Scheduler::new(
            HeuristicProfiler::new(shape.n_layers),
            NgramBranchPredictor::new(8), 1, 42);
        let mut backend = SimBackend::new(12_000);
        let fb = FallbackController::new();
        for (i, &pos) in positions.iter().enumerate() {
            let tctx = TokenContext { token_id: i as u32, position: pos,
                recent_logit_margin: margin, recent_entropy: 0.2,
                token_kind: TokenKind::Content, chain_phase: ChainPhase::Middle };
            let rctx = RoutingContext { position: pos, recent_experts: vec![] };
            let tr = sched.step(&tctx, &rctx, &shape, &mut backend, &fb);
            prop_assert!(validate(&tr).is_ok());
            prop_assert!(tr.vram_usage_mb <= 12_000);
        }
    }

    // I5: fallback never escalates past 3 and baseline stays reachable
    // under any anomaly sequence.
    #[test]
    fn i5_fallback_capped_and_baseline_reachable(
        seq in prop::collection::vec(0u8..5, 0..50)
    ) {
        let mut fb = FallbackController::new();
        for code in seq {
            let a = match code {
                0 => Anomaly::ReasoningCollapse,
                1 => Anomaly::NbpFloorBreach,
                2 => Anomaly::KvIntegrity,
                3 => Anomaly::VramPressure,
                _ => Anomaly::KernelNaN,
            };
            fb.on_anomaly(a);
            prop_assert!(fb.level() <= 3);
            prop_assert!(fb.baseline_reachable());
        }
    }
}
