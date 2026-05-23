// SPDX-License-Identifier: Apache-2.0
use crate::hw::{FakeVram, LatencyModel};
use crate::model::true_experts;
use mirage_core::context::{ExecOutcome, ExecTelemetry};
use mirage_core::decision::ActualRouting;
use mirage_core::plan::{ExecPlan, ExpertBinding, ModelShape};
use mirage_core::traits::ExecutionBackend;

pub struct SimBackend {
    vram: FakeVram,
}

impl SimBackend {
    pub fn new(budget_mb: u32) -> Self {
        SimBackend {
            vram: FakeVram::new(budget_mb),
        }
    }
}

impl ExecutionBackend for SimBackend {
    fn execute(&mut self, plan: &ExecPlan, shape: &ModelShape) -> ExecOutcome {
        let mut total_misses = 0u32;
        let mut experts_per_layer = Vec::new();
        let executed = plan.exit_after.min(plan.ops.len() as u16);
        for op in plan.ops.iter().take(executed as usize) {
            // Honor prefetch hint from the plan's ExpertBinding.
            if let ExpertBinding::Resident(es) | ExpertBinding::Stream { experts: es, .. } =
                &op.experts
            {
                self.vram.ensure_resident(es);
            }
            // The TRUE routing the model actually needs (spec §10 step 7b).
            let actual = true_experts(shape, op.layer_idx, plan.position);
            if !actual.is_empty() {
                total_misses += self.vram.ensure_resident(&actual);
                experts_per_layer.push(actual);
            }
        }
        let compute = LatencyModel::compute_us(executed);
        let prefetch = LatencyModel::transfer_us(total_misses);
        let stall = LatencyModel::stall_us(total_misses);
        ExecOutcome {
            actual: ActualRouting {
                experts_per_moe_layer: experts_per_layer,
                exit_layer: executed,
            },
            telemetry: ExecTelemetry {
                prefetch_us: prefetch,
                compute_us: compute,
                stall_us: stall,
                cpu_us: 0,
                vram_mb: self.vram.used_mb(),
                pcie_bytes: total_misses as u64 * crate::hw::EXPERT_BYTES,
                kv_bytes: 0,
                energy_mj: (compute + stall) / 10,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirage_core::control::{Device, HeadMask, PrecisionTier};
    use mirage_core::plan::LayerOp;

    fn plan_for(position: u32, layers: u16) -> ExecPlan {
        let ops = (0..layers)
            .map(|l| LayerOp {
                layer_idx: l,
                precision: PrecisionTier::Q4,
                head_mask: HeadMask::all(32),
                experts: ExpertBinding::Resident(vec![]),
                device: Device::Gpu,
            })
            .collect();
        ExecPlan {
            token_id: position,
            position,
            ops,
            exit_after: layers,
            fallback_level: 0,
        }
    }

    #[test]
    fn warm_cache_eliminates_misses_on_repeat() {
        let s = ModelShape::qwen3_30b_a3b();
        let mut b = SimBackend::new(12_000);
        let cold = b.execute(&plan_for(100, 4), &s);
        let warm = b.execute(&plan_for(100, 4), &s);
        assert!(warm.telemetry.stall_us < cold.telemetry.stall_us);
        assert_eq!(warm.telemetry.prefetch_us, 0);
    }

    #[test]
    fn execute_is_deterministic() {
        let s = ModelShape::qwen3_30b_a3b();
        let mut b1 = SimBackend::new(12_000);
        let mut b2 = SimBackend::new(12_000);
        assert_eq!(
            b1.execute(&plan_for(50, 8), &s),
            b2.execute(&plan_for(50, 8), &s)
        );
    }
}
