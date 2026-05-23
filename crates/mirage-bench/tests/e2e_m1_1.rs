// SPDX-License-Identifier: Apache-2.0
//! M1.1 exit criterion: oracle + adaptive both run on GsmToy, both produce
//! valid traces, plan-divergence stays inside the gate, and a Markdown
//! report renders cleanly.
use mirage_bench::oracle::{OraclePredictor, OracleProfiler};
use mirage_bench::render::to_markdown;
use mirage_bench::report::BenchReport;
use mirage_bench::runner::BenchRunner;
use mirage_bench::scorer::{PlanDivergence, Scorer};
use mirage_bench::task::Task;
use mirage_bench::tasks::gsm_toy::GsmToy;
use mirage_core::plan::ModelShape;
use mirage_predictor::NgramBranchPredictor;
use mirage_profiler::HeuristicProfiler;
use mirage_telemetry::validate;

#[test]
fn m1_1_exit_criterion() {
    let shape = ModelShape::qwen3_30b_a3b();
    let runner = BenchRunner::new(shape, 12_000, 1, 42);
    let task = GsmToy;
    let samples = task.samples();

    let oracle = runner.run(
        OracleProfiler::new(shape.n_layers),
        OraclePredictor,
        &samples,
    );
    let adaptive = runner.run(
        HeuristicProfiler::new(shape.n_layers),
        NgramBranchPredictor::new(8),
        &samples,
    );

    assert!(!oracle.is_empty(), "oracle produced traces");
    assert_eq!(
        oracle.len(),
        adaptive.len(),
        "same input → same trace count"
    );
    for t in oracle.iter().chain(adaptive.iter()) {
        assert!(validate(t).is_ok(), "I8: every emitted trace must validate");
        assert!(t.vram_usage_mb <= 12_000, "I7: budget respected");
    }

    let div = PlanDivergence.score(&oracle, &adaptive);
    assert!(
        div.value <= 0.02,
        "M1.1 gate: divergence {} > 2%",
        div.value
    );

    let report = BenchReport::from_traces("adaptive", task.name(), &adaptive, Some(div));
    let md = to_markdown(&[report]);
    assert!(md.starts_with("# mirage-bench report"));
}
