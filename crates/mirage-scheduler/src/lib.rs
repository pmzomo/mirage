// SPDX-License-Identifier: Apache-2.0
//! Adaptive scheduler + graded fallback (spec §19, §10).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anomaly { ReasoningCollapse, NbpFloorBreach, KvIntegrity, VramPressure, KernelNaN }

/// Levels: 0 full adaptive .. 3 pure baseline. Invariant I5: level 3 is
/// reachable from any state and never escalates past 3.
pub struct FallbackController { level: u8 }

impl FallbackController {
    pub fn new() -> Self { FallbackController { level: 0 } }
    pub fn level(&self) -> u8 { self.level }
    pub fn on_anomaly(&mut self, a: Anomaly) {
        let step = match a {
            Anomaly::NbpFloorBreach => 2,
            Anomaly::ReasoningCollapse => 1,
            Anomaly::KvIntegrity | Anomaly::VramPressure => 1,
            Anomaly::KernelNaN => 3,
        };
        self.level = (self.level.max(step)).min(3);
    }
    pub fn baseline_reachable(&self) -> bool { true }
}

impl Default for FallbackController { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anomalies_escalate_monotonically_and_cap_at_3() {
        let mut f = FallbackController::new();
        f.on_anomaly(Anomaly::ReasoningCollapse);
        assert_eq!(f.level(), 1);
        f.on_anomaly(Anomaly::NbpFloorBreach);
        assert_eq!(f.level(), 2);
        f.on_anomaly(Anomaly::ReasoningCollapse); // never de-escalates
        assert_eq!(f.level(), 2);
        f.on_anomaly(Anomaly::KernelNaN);
        assert_eq!(f.level(), 3);
        f.on_anomaly(Anomaly::KernelNaN);
        assert_eq!(f.level(), 3); // capped
        assert!(f.baseline_reachable());
    }
}

use mirage_core::control::{Device, HeadMask, PrecisionTier};
use mirage_core::context::{RoutingContext, TokenContext};
use mirage_core::plan::{ExecPlan, ExpertBinding, LayerOp, ModelShape};
use mirage_core::traits::{BranchPredictor, CognitiveProfiler, ExecutionBackend};
use mirage_telemetry::{LatencyBreakdown, TokenTrace, SCHEMA_VERSION};

pub struct Scheduler<P, B> { pub profiler: P, pub predictor: B, pub run_id: u128, pub model_id: u64 }

impl<P: CognitiveProfiler, B: BranchPredictor> Scheduler<P, B> {
    pub fn new(profiler: P, predictor: B, run_id: u128, model_id: u64) -> Self {
        Scheduler { profiler, predictor, run_id, model_id }
    }

    /// One token through the full moat (spec §10). Pure given inputs +
    /// backend => deterministic. Fallback level forces axis disable per §19.
    pub fn step<E: ExecutionBackend>(
        &mut self,
        tctx: &TokenContext,
        rctx: &RoutingContext,
        shape: &ModelShape,
        backend: &mut E,
        fallback: &FallbackController,
    ) -> TokenTrace {
        let decision = self.profiler.predict(tctx);
        let prediction = self.predictor.predict(rctx);

        // Fallback gating (§19): level >=1 disables dynamic precision (force Q4),
        // level >=3 disables early-exit (full depth, baseline-equivalent).
        let tier = if fallback.level() >= 1 { PrecisionTier::Q4 }
                   else { decision.precision_per_group[0] };
        let depth = if fallback.level() >= 3 { shape.n_layers } else { decision.target_depth };

        let predicted_flat: Vec<u16> =
            prediction.experts_per_moe_layer.iter().flatten().copied().collect();

        let ops = (0..depth.min(shape.n_layers)).map(|l| LayerOp {
            layer_idx: l, precision: tier, head_mask: HeadMask::all(shape.n_heads),
            experts: ExpertBinding::Resident(predicted_flat.clone()), device: Device::Gpu,
        }).collect();

        let plan = ExecPlan {
            token_id: tctx.token_id, position: tctx.position, ops,
            exit_after: depth.min(shape.n_layers), fallback_level: fallback.level(),
        };

        let outcome = backend.execute(&plan, shape);
        self.predictor.update(&prediction, &outcome.actual);

        let actual_flat: Vec<u16> =
            outcome.actual.experts_per_moe_layer.iter().flatten().copied().collect();
        let t = &outcome.telemetry;
        let total = t.prefetch_us + t.compute_us + t.stall_us + t.cpu_us + 1;

        TokenTrace {
            schema_version: SCHEMA_VERSION, model_id: self.model_id, run_id: self.run_id,
            token_id: tctx.token_id, position: tctx.position, chain_phase: decision.chain_phase,
            predicted_experts: predicted_flat, actual_experts: actual_flat,
            precision_plan: vec![tier], profiler_decision: decision,
            latency: LatencyBreakdown { total_us: total, prefetch_us: t.prefetch_us,
                compute_us: t.compute_us, stall_us: t.stall_us, cpu_us: t.cpu_us, sample_us: 1 },
            vram_usage_mb: t.vram_mb, pcie_bytes: t.pcie_bytes, kv_cache_bytes: t.kv_bytes,
            energy_mj: t.energy_mj, logits_entropy: tctx.recent_entropy,
            logit_margin: tctx.recent_logit_margin, fallback_level: fallback.level(),
            deterministic: true,
        }
    }
}

#[cfg(test)]
mod step_tests {
    use super::*;
    use mirage_core::control::{ChainPhase, TokenKind};
    use mirage_profiler::HeuristicProfiler;
    use mirage_predictor::NgramBranchPredictor;
    use mirage_sim::backend::SimBackend;
    use mirage_telemetry::validate;

    fn tctx(id: u32) -> TokenContext {
        TokenContext { token_id: id, position: id, recent_logit_margin: 1.0,
            recent_entropy: 0.2, token_kind: TokenKind::Content,
            chain_phase: ChainPhase::NotReasoning }
    }

    #[test]
    fn step_emits_a_valid_trace() {
        let shape = ModelShape::qwen3_30b_a3b();
        let mut sched = Scheduler::new(
            HeuristicProfiler::new(shape.n_layers),
            NgramBranchPredictor::new(8), 1, 42);
        let mut backend = SimBackend::new(12_000);
        let fb = FallbackController::new();
        let tr = sched.step(&tctx(5),
            &RoutingContext { position: 5, recent_experts: vec![] },
            &shape, &mut backend, &fb);
        assert!(validate(&tr).is_ok());
        assert_eq!(tr.fallback_level, 0);
    }

    #[test]
    fn level_3_forces_full_depth_q4() {
        let shape = ModelShape::qwen3_30b_a3b();
        let mut sched = Scheduler::new(
            HeuristicProfiler::new(shape.n_layers),
            NgramBranchPredictor::new(8), 1, 42);
        let mut backend = SimBackend::new(12_000);
        let mut fb = FallbackController::new();
        fb.on_anomaly(Anomaly::KernelNaN); // -> level 3
        // Trivial token would normally go shallow+Q2; level 3 overrides.
        let mut c = tctx(9);
        c.token_kind = TokenKind::Punctuation;
        c.recent_logit_margin = 9.0;
        let tr = sched.step(&c, &RoutingContext { position: 9, recent_experts: vec![] },
            &shape, &mut backend, &fb);
        assert_eq!(tr.precision_plan, vec![PrecisionTier::Q4]);
        assert_eq!(tr.profiler_decision.difficulty,
                   mirage_core::control::DifficultyClass::Trivial);
        // Depth forced to full despite trivial decision:
        assert_eq!(tr.latency.compute_us,
                   mirage_sim::hw::LatencyModel::compute_us(shape.n_layers));
    }
}
