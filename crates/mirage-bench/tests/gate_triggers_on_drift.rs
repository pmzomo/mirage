// SPDX-License-Identifier: Apache-2.0
//! Proves the 2% plan-divergence gate triggers when the adaptive config
//! is intentionally broken. A BadProfiler always returns depth=1, which
//! deviates from the oracle's full-depth on every reasoning token.
use mirage_bench::oracle::{OracleProfiler, OraclePredictor};
use mirage_bench::runner::BenchRunner;
use mirage_bench::scorer::{PlanDivergence, Scorer};
use mirage_bench::task::Task;
use mirage_bench::tasks::gsm_toy::GsmToy;
use mirage_core::context::TokenContext;
use mirage_core::control::{ChainPhase, DifficultyClass, PrecisionTier};
use mirage_core::decision::ComputeDecision;
use mirage_core::traits::CognitiveProfiler;
use mirage_predictor::NgramBranchPredictor;
use mirage_core::plan::ModelShape;

struct BadProfiler;
impl CognitiveProfiler for BadProfiler {
    fn predict(&self, _: &TokenContext) -> ComputeDecision {
        ComputeDecision {
            difficulty: DifficultyClass::Trivial,
            target_depth: 1,
            precision_per_group: vec![PrecisionTier::Q2],
            head_budget: 4,
            burst: false,
            chain_phase: ChainPhase::NotReasoning,
        }
    }
}

#[test]
fn manufactured_drift_blows_the_gate() {
    let shape = ModelShape::qwen3_30b_a3b();
    let runner = BenchRunner::new(shape, 12_000, 7, 42);
    let samples = GsmToy.samples();
    let oracle = runner.run(OracleProfiler::new(shape.n_layers), OraclePredictor, &samples);
    let bad = runner.run(BadProfiler, NgramBranchPredictor::new(8), &samples);
    let div = PlanDivergence.score(&oracle, &bad);
    // Every token diverges (depth differs every time). Score should be exactly 1.0.
    assert!(
        div.value > 0.02,
        "expected drift > 2% gate, got {} (this gate would silently let regressions ship)",
        div.value,
    );
}
