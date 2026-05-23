// SPDX-License-Identifier: Apache-2.0
//! Invariant I6 (spec §20): identical (input, run, shape) => byte-identical traces.
use mirage_core::context::{RoutingContext, TokenContext};
use mirage_core::control::{ChainPhase, TokenKind};
use mirage_core::plan::ModelShape;
use mirage_predictor::NgramBranchPredictor;
use mirage_profiler::HeuristicProfiler;
use mirage_scheduler::{FallbackController, Scheduler};
use mirage_sim::backend::SimBackend;
use mirage_telemetry::frame::write_record;

fn run_once() -> Vec<u8> {
    let shape = ModelShape::qwen3_30b_a3b();
    let mut sched = Scheduler::new(
        HeuristicProfiler::new(shape.n_layers),
        NgramBranchPredictor::new(8), 7, 42);
    let mut backend = SimBackend::new(12_000);
    let fb = FallbackController::new();
    let mut buf = Vec::new();
    for i in 0..200u32 {
        let tctx = TokenContext { token_id: i, position: i, recent_logit_margin: 1.0,
            recent_entropy: 0.2, token_kind: TokenKind::Content,
            chain_phase: ChainPhase::NotReasoning };
        let rctx = RoutingContext { position: i, recent_experts: vec![] };
        let tr = sched.step(&tctx, &rctx, &shape, &mut backend, &fb);
        write_record(&mut buf, &tr).unwrap();
    }
    buf
}

#[test]
fn two_runs_are_byte_identical() {
    assert_eq!(run_once(), run_once());
}
