// SPDX-License-Identifier: Apache-2.0
//! Reasoning-retention CI gate. Runs the GsmToy task through:
//!   1. the sim-oracle config (full-depth FP16, no adaptivity)
//!   2. the production-default adaptive config (HeuristicProfiler + NgramBranchPredictor)
//! and prints a Markdown report. Exits with code 1 if plan-divergence
//! exceeds the threshold (default 2.0 %).
use mirage_bench::oracle::{OracleProfiler, OraclePredictor};
use mirage_bench::render::to_markdown;
use mirage_bench::report::BenchReport;
use mirage_bench::runner::BenchRunner;
use mirage_bench::scorer::{PlanDivergence, Scorer};
use mirage_bench::task::Task;
use mirage_bench::tasks::gsm_toy::GsmToy;
use mirage_core::plan::ModelShape;
use mirage_predictor::NgramBranchPredictor;
use mirage_profiler::HeuristicProfiler;

const THRESHOLD: f64 = 0.02; // 2 %

fn main() {
    let shape = ModelShape::qwen3_30b_a3b();
    let task = GsmToy;
    let samples = task.samples();
    let runner = BenchRunner::new(shape, 12_000, 99, 42);

    let oracle = runner.run(OracleProfiler::new(shape.n_layers), OraclePredictor, &samples);
    let adaptive = runner.run(
        HeuristicProfiler::new(shape.n_layers),
        NgramBranchPredictor::new(8),
        &samples,
    );

    let div = PlanDivergence.score(&oracle, &adaptive);
    let r_o = BenchReport::from_traces("oracle", task.name(), &oracle, None);
    let r_a = BenchReport::from_traces("adaptive", task.name(), &adaptive, Some(div.clone()));
    print!("{}", to_markdown(&[r_o, r_a]));

    if div.value > THRESHOLD {
        eprintln!(
            "FAIL: plan-divergence {:.4} exceeds gate threshold {:.4}",
            div.value, THRESHOLD,
        );
        std::process::exit(1);
    }
    eprintln!("OK: plan-divergence {:.4} within gate {:.4}", div.value, THRESHOLD);
}
