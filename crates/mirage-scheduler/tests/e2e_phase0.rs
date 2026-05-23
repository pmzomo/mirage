// SPDX-License-Identifier: Apache-2.0
//! Phase 0 EXIT CRITERION (spec §27): synthetic Qwen3-30B-shaped model runs
//! the full moat in sim; invariants hold; NBP hit rate rises as the predictor
//! warms; all traces validate.
use mirage_core::context::{RoutingContext, TokenContext};
use mirage_core::control::{ChainPhase, TokenKind};
use mirage_core::plan::ModelShape;
use mirage_predictor::NgramBranchPredictor;
use mirage_profiler::HeuristicProfiler;
use mirage_scheduler::{FallbackController, Scheduler};
use mirage_sim::backend::SimBackend;
use mirage_telemetry::validate;

#[test]
fn phase0_full_moat_runs_and_predictor_warms() {
    let shape = ModelShape::qwen3_30b_a3b();
    let mut sched = Scheduler::new(
        HeuristicProfiler::new(shape.n_layers),
        NgramBranchPredictor::new(8),
        99,
        42,
    );
    let mut backend = SimBackend::new(12_000);
    let fb = FallbackController::new();

    let mut recent: Vec<u16> = Vec::new();
    let mut early_stall = 0u64;
    let mut late_stall = 0u64;

    for i in 0..512u32 {
        // Coherent passage: position advances so topic windows repeat (locality).
        let tctx = TokenContext {
            token_id: i,
            position: i,
            recent_logit_margin: 1.0,
            recent_entropy: 0.2,
            token_kind: TokenKind::Content,
            chain_phase: ChainPhase::Middle,
        };
        let rctx = RoutingContext {
            position: i,
            recent_experts: recent.clone(),
        };
        let tr = sched.step(&tctx, &rctx, &shape, &mut backend, &fb);

        assert!(validate(&tr).is_ok(), "I8: trace must validate");
        assert!(tr.vram_usage_mb <= 12_000, "I7: budget");
        assert_eq!(tr.fallback_level, 0, "no anomalies injected");

        recent = tr.actual_experts.clone();
        if i < 64 {
            early_stall += tr.latency.stall_us as u64;
        }
        if i >= 448 {
            late_stall += tr.latency.stall_us as u64;
        }
    }

    // The warm cache + predictor must reduce stalls over the run
    // (NBP/locality is doing real work — spec §9).
    assert!(
        late_stall < early_stall,
        "expected stalls to fall as cache/predictor warm: early={early_stall} late={late_stall}"
    );
}
