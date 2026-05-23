// SPDX-License-Identifier: Apache-2.0
use crate::task::BenchSample;
use mirage_core::context::{RoutingContext, TokenContext};
use mirage_core::plan::ModelShape;
use mirage_core::traits::{BranchPredictor, CognitiveProfiler};
use mirage_scheduler::{FallbackController, Scheduler};
use mirage_sim::backend::SimBackend;
use mirage_telemetry::TokenTrace;

pub struct BenchRunner {
    pub shape: ModelShape,
    pub vram_budget_mb: u32,
    pub run_id: u128,
    pub model_id: u64,
}

impl BenchRunner {
    pub fn new(shape: ModelShape, vram_budget_mb: u32, run_id: u128, model_id: u64) -> Self {
        BenchRunner { shape, vram_budget_mb, run_id, model_id }
    }

    pub fn run<P: CognitiveProfiler, B: BranchPredictor>(
        &self,
        profiler: P,
        predictor: B,
        samples: &[BenchSample],
    ) -> Vec<TokenTrace> {
        let mut sched = Scheduler::new(profiler, predictor, self.run_id, self.model_id);
        let mut backend = SimBackend::new(self.vram_budget_mb);
        let fb = FallbackController::new();
        let mut out = Vec::with_capacity(samples.iter().map(|s| s.steps.len()).sum());
        let mut recent: Vec<u16> = Vec::new();
        let mut position = 0u32;

        for sample in samples {
            for step in &sample.steps {
                let tctx = TokenContext {
                    token_id: step.token_id,
                    position,
                    recent_logit_margin: step.recent_logit_margin,
                    recent_entropy: step.recent_entropy,
                    token_kind: step.token_kind,
                    chain_phase: step.chain_phase,
                };
                let rctx = RoutingContext { position, recent_experts: recent.clone() };
                let tr = sched.step(&tctx, &rctx, &self.shape, &mut backend, &fb);
                recent = tr.actual_experts.clone();
                position += 1;
                out.push(tr);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::gsm_toy::GsmToy;
    use crate::task::Task;
    use mirage_profiler::HeuristicProfiler;
    use mirage_predictor::NgramBranchPredictor;
    use mirage_telemetry::validate;

    #[test]
    fn runner_emits_one_trace_per_step() {
        let shape = ModelShape::qwen3_30b_a3b();
        let runner = BenchRunner::new(shape, 12_000, 1, 42);
        let samples = GsmToy.samples();
        let expected: usize = samples.iter().map(|s| s.steps.len()).sum();
        let traces = runner.run(
            HeuristicProfiler::new(shape.n_layers),
            NgramBranchPredictor::new(8),
            &samples,
        );
        assert_eq!(traces.len(), expected);
        for t in &traces { assert!(validate(t).is_ok()); }
    }
}
